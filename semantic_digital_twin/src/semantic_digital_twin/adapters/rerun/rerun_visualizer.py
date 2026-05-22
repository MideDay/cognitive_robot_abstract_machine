from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import rerun as rr
from typing_extensions import Optional

from semantic_digital_twin.adapters.rerun.rerun_logger import (
    DEFAULT_ROOT_ENTITY_PATH,
    MeshFileResolver,
    log_model,
    log_state,
)
from semantic_digital_twin.callbacks.callback import (
    ModelChangeCallback,
    StateChangeCallback,
)


class RerunSink(Enum):
    """Where the Rerun recording stream sends its data."""

    SPAWN = "spawn"
    """Spawn and stream to a local Rerun viewer."""
    CONNECT = "connect"
    """Stream to an already-running viewer over gRPC (uses ``sink_target`` URL)."""
    SAVE = "save"
    """Write an ``.rrd`` recording file (uses ``sink_target`` path)."""
    NONE = "none"
    """Do not configure a sink; the caller manages the recording's output."""


@dataclass(eq=False)
class _RerunStateCallback(StateChangeCallback):
    """Re-logs the per-body transforms on every world state change."""

    visualizer: RerunVisualizer = field(kw_only=True)
    """The owning visualizer whose recording and configuration are used."""

    def _notify(self, **kwargs) -> None:
        self.visualizer.log_current_state()


@dataclass(eq=False)
class RerunVisualizer(ModelChangeCallback):
    """
    Live Rerun view of a world that updates automatically as the world changes.

    Mirrors the ROS visualization adapter: a model-change callback that re-logs
    the static geometry, owning a state-change callback that re-logs the per-body
    transforms. Logs the geometry and current state on construction.
    """

    root_entity_path: str = DEFAULT_ROOT_ENTITY_PATH
    """Entity path under which the kinematic tree is logged."""
    application_id: str = "semantic_digital_twin"
    """Rerun application id for the recording."""
    sink: RerunSink = field(default=RerunSink.SPAWN, kw_only=True)
    """Where the recording sends its data."""
    sink_target: Optional[str] = field(default=None, kw_only=True)
    """gRPC URL for ``CONNECT`` or file path for ``SAVE``."""
    timeline: str = field(default="state_version", kw_only=True)
    """Name of the Rerun timeline driven by the world state version."""
    mesh_file_resolver: Optional[MeshFileResolver] = field(default=None, kw_only=True)
    """Optional hook to load local mesh files for color."""
    state_history: bool = field(default=True, kw_only=True)
    """Keep a scrubbable state history (bounded by ``memory_limit``); if ``False``, keep only the current state."""
    memory_limit: str = field(default="75%", kw_only=True)
    """Spawned-viewer memory budget (e.g. ``"2GB"``); oldest data is dropped past it. Only used by the ``SPAWN`` sink."""

    _recording: rr.RecordingStream = field(init=False, repr=False)
    """The Rerun recording stream all data is logged to."""
    _state_callback: _RerunStateCallback = field(init=False, repr=False)
    """The owned callback that re-logs transforms on state changes."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self._recording = rr.RecordingStream(self.application_id)
        self._configure_sink()
        self._state_callback = _RerunStateCallback(_world=self._world, visualizer=self)
        self._notify()
        self.log_current_state()

    def _configure_sink(self) -> None:
        """Attach the configured output sink to the recording stream."""
        if self.sink is RerunSink.SPAWN:
            self._recording.spawn(memory_limit=self.memory_limit)
        elif self.sink is RerunSink.CONNECT:
            if self.sink_target is None:
                raise ValueError("RerunSink.CONNECT requires a sink_target gRPC URL.")
            self._recording.connect_grpc(self.sink_target)
        elif self.sink is RerunSink.SAVE:
            if self.sink_target is None:
                raise ValueError("RerunSink.SAVE requires a sink_target file path.")
            self._recording.save(self.sink_target)

    def _notify(self, **kwargs) -> None:
        log_model(
            self._world,
            self.root_entity_path,
            mesh_file_resolver=self.mesh_file_resolver,
            recording=self._recording,
        )

    def log_current_state(self) -> None:
        """Re-log the current per-body transforms, advancing the timeline if history is kept."""
        if self.state_history:
            rr.set_time(
                self.timeline,
                sequence=self._world.state.version,
                recording=self._recording,
            )
            log_state(self._world, self.root_entity_path, recording=self._recording)
        else:
            log_state(
                self._world,
                self.root_entity_path,
                static=True,
                recording=self._recording,
            )

    def stop(self) -> None:
        """Detach both the model and state callbacks from the world."""
        super().stop()
        self._state_callback.stop()
