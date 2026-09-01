"""
Turning a signal that would end the process into an orderly halt of the robot.
"""

from __future__ import annotations

import signal
from dataclasses import dataclass, field
from enum import Enum
from types import FrameType
from typing import Any, Dict

from rclpy.signals import (
    SignalHandlerOptions,
    get_current_signal_handlers_options,
    install_signal_handlers,
    uninstall_signal_handlers,
)
from typing_extensions import Self


class ShutdownSignal(Enum):
    """
    A signal that asks the process to end.

    ..note:: Not an :class:`enum.IntEnum`, because an int survives the json round trip to
        the action client as a bare number instead of as this member.
    """

    INTERRUPT = signal.SIGINT
    """
    Sent when the user interrupts the process, for example with ``Ctrl+C``.
    """

    TERMINATE = signal.SIGTERM
    """
    Sent when a supervisor asks the process to end, for example a ros2 launch shutdown
    after its interrupt went unanswered.
    """


@dataclass
class ShutdownRequest:
    """
    Whether the process was asked to end, and by which signal.

    Written by a signal handler and read by the loops that command the robot, so that
    they can leave their cycle and halt it instead of being torn down mid-motion.
    """

    received_signal: ShutdownSignal | None = field(init=False, default=None)
    """
    The signal that asked the process to end, ``None`` while none arrived.
    """

    @property
    def is_pending(self) -> bool:
        """
        Whether the process was asked to end.
        """
        return self.received_signal is not None

    def receive(self, received_signal: ShutdownSignal) -> None:
        """
        Record the signal that asked the process to end.
        """
        self.received_signal = received_signal


@dataclass
class ShutdownSignalListener:
    """
    Records a :class:`ShutdownSignal` in a request instead of letting it end the process.

    Giskard commands the robot over velocity interfaces, which keep executing the last
    velocity they were sent. Both rclpy's handlers and Python's default one end the
    process where it stands, leaving that velocity running, so this takes them out of the
    way for as long as it is entered.

    ..note:: Only the first signal is recorded: the handlers that were installed before
        are put back right away, so a second interrupt still ends a process whose halt
        hangs.

    ..warning:: Signal handlers can only be installed from the main thread.
    """

    request: ShutdownRequest
    """
    The request the received signal is recorded in.
    """

    _replaced_handlers: Dict[ShutdownSignal, Any] = field(
        init=False, default_factory=dict
    )
    """
    The handler that was installed for each signal before, empty once they are back.
    """

    _replaced_rclpy_options: SignalHandlerOptions | None = field(
        init=False, default=None
    )
    """
    The signals rclpy handled before, ``None`` while this listener is not installed.
    """

    def __enter__(self) -> Self:
        self.install()
        return self

    def __exit__(self, exception_type, exception, traceback) -> None:
        self.uninstall()

    def install(self) -> None:
        """
        Take over the handling of every :class:`ShutdownSignal`.
        """
        self._replaced_rclpy_options = get_current_signal_handlers_options()
        uninstall_signal_handlers()
        for shutdown_signal in ShutdownSignal:
            self._replaced_handlers[shutdown_signal] = signal.signal(
                shutdown_signal.value, self._record_signal
            )

    def uninstall(self) -> None:
        """
        Give the handling of every :class:`ShutdownSignal` back.
        """
        self.restore_replaced_handlers()
        if self._replaced_rclpy_options is None:
            return
        install_signal_handlers(self._replaced_rclpy_options)
        self._replaced_rclpy_options = None

    def restore_replaced_handlers(self) -> None:
        """
        Put the handlers that were installed before this listener back.
        """
        for shutdown_signal, replaced_handler in self._replaced_handlers.items():
            signal.signal(shutdown_signal.value, replaced_handler)
        self._replaced_handlers.clear()

    def _record_signal(self, signal_number: int, frame: FrameType | None) -> None:
        """
        Record the received signal and step aside, so the next one ends the process.
        """
        self.restore_replaced_handlers()
        self.request.receive(ShutdownSignal(signal_number))
