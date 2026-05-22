"""Unit tests for the Rerun adapter (stateless logging and the live visualizer).

The tests build a tiny synthetic world from primitive geometry and a temporary
colored mesh, so they need no external mesh assets and run in CI. Logging targets
a sink-less recording stream (data is buffered, never displayed), and the
visualizer uses ``RerunSink.NONE`` so no viewer is spawned.
"""

import numpy as np
import rerun as rr
import trimesh

from semantic_digital_twin.adapters.rerun import (
    RerunSink,
    RerunVisualizer,
    log_model,
    log_state,
    log_world,
    shape_to_link_frame_mesh,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.geometry import Box, Color, Mesh, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body


def _make_box_world() -> World:
    """Build a world with a single box body fixed above the root."""
    world = World()
    root = Body(name=PrefixedName("root"))
    box = Box(
        origin=HomogeneousTransformationMatrix(),
        scale=Scale(0.2, 0.2, 0.2),
        color=Color(1.0, 0.0, 0.0, 1.0),
    )
    box_body = Body(
        name=PrefixedName("box"),
        visual=ShapeCollection([box]),
        collision=ShapeCollection([box]),
    )
    with world.modify_world():
        world.add_kinematic_structure_entity(root)
        world.add_connection(
            FixedConnection(
                parent=root,
                child=box_body,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    z=0.5
                ),
            )
        )
    return world


# --- Layer 1: stateless logging ---------------------------------------------


def test_shape_to_link_frame_mesh_returns_colored_geometry() -> None:
    """A primitive shape converts to a trimesh with geometry and vertex colors."""
    mesh = shape_to_link_frame_mesh(Box(scale=Scale(0.2, 0.2, 0.2)))
    assert len(mesh.vertices) > 0
    assert len(mesh.faces) > 0
    assert mesh.visual.vertex_colors.shape[1] == 4


def test_shape_to_link_frame_mesh_bakes_origin() -> None:
    """The shape's local origin is baked into the returned vertices."""
    offset = 5.0
    centered = shape_to_link_frame_mesh(Box(scale=Scale(1.0, 1.0, 1.0)))
    shifted = shape_to_link_frame_mesh(
        Box(
            origin=HomogeneousTransformationMatrix.from_xyz_rpy(x=offset),
            scale=Scale(1.0, 1.0, 1.0),
        )
    )
    assert np.isclose(
        shifted.vertices[:, 0].mean() - centered.vertices[:, 0].mean(), offset
    )


def test_log_functions_run_without_error() -> None:
    """Logging model, state (timeline and static), and a full snapshot does not raise."""
    world = _make_box_world()
    recording = rr.RecordingStream("test_semdt_rerun")
    log_model(world, recording=recording)
    log_state(world, recording=recording)
    log_state(world, static=True, recording=recording)
    log_world(world, recording=recording)


def test_mesh_color_survives_serialization(tmp_path) -> None:
    """A mesh's per-vertex color survives the to_json/_from_json wire round-trip.

    This is the path the distributed viewer relies on: color travels with the
    geometry, so no local mesh file is needed to render it.
    """
    source = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    source.visual.vertex_colors = np.tile([200, 50, 50, 255], (len(source.vertices), 1))
    mesh_path = tmp_path / "colored_box.ply"
    source.export(str(mesh_path))

    over_the_wire = Mesh.from_json(Mesh(filename=str(mesh_path)).to_json())
    rendered = shape_to_link_frame_mesh(over_the_wire)

    assert (rendered.visual.vertex_colors[:, :3] == [200, 50, 50]).all()


# --- Layer 2: the live visualizer -------------------------------------------


def test_rerun_visualizer_registers_handles_state_and_stops() -> None:
    """The visualizer attaches callbacks, handles a state change, and detaches on stop."""
    world = _make_box_world()
    state_callbacks_before = len(world.state.state_change_callbacks)

    visualizer = RerunVisualizer(_world=world, sink=RerunSink.NONE)
    assert len(world.state.state_change_callbacks) > state_callbacks_before

    world.notify_state_change()  # exercises the state callback path

    visualizer.stop()
    assert len(world.state.state_change_callbacks) == state_callbacks_before
