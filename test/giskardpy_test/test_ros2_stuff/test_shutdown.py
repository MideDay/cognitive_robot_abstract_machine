import signal

import rclpy
from rclpy.signals import get_current_signal_handlers_options

from giskardpy.middleware.ros2.shutdown import (
    ShutdownRequest,
    ShutdownSignal,
    ShutdownSignalListener,
)

# %% recording the signal


class TestReceivedSignalIsRecorded:
    """
    A signal that would end the process is recorded instead, so the loops commanding the
    robot get the chance to halt it first.
    """

    def test_an_interrupt_is_recorded(self):
        request = ShutdownRequest()

        with ShutdownSignalListener(request=request):
            signal.raise_signal(signal.SIGINT)

        assert request.received_signal is ShutdownSignal.INTERRUPT
        assert request.is_pending

    def test_a_terminate_is_recorded(self):
        """
        A ros2 launch shutdown escalates from an interrupt to a terminate, which would
        otherwise kill the process outright.
        """
        request = ShutdownRequest()

        with ShutdownSignalListener(request=request):
            signal.raise_signal(signal.SIGTERM)

        assert request.received_signal is ShutdownSignal.TERMINATE

    def test_an_interrupt_raises_no_keyboard_interrupt(self):
        """
        The interrupt must not unwind the main thread, which is in the middle of a
        control cycle and still has to publish the zero velocities.
        """
        reached_the_end = False

        with ShutdownSignalListener(request=ShutdownRequest()):
            signal.raise_signal(signal.SIGINT)
            reached_the_end = True

        assert reached_the_end

    def test_nothing_is_recorded_without_a_signal(self):
        request = ShutdownRequest()

        with ShutdownSignalListener(request=request):
            pass

        assert request.received_signal is None
        assert not request.is_pending


# %% giving the signals back


class TestHandlersAreGivenBack:
    """
    The listener owns the signals only for as long as it is entered, and only until the
    first one arrives.
    """

    def test_the_previous_handler_is_back_after_the_first_signal(self):
        """
        A second interrupt has to end a process whose halt hangs, so the listener steps
        aside as soon as it recorded the first one.
        """
        handler_before = signal.getsignal(signal.SIGINT)

        with ShutdownSignalListener(request=ShutdownRequest()):
            signal.raise_signal(signal.SIGINT)

            assert signal.getsignal(signal.SIGINT) is handler_before

    def test_leaving_restores_every_handler(self):
        handlers_before = {
            shutdown_signal: signal.getsignal(shutdown_signal.value)
            for shutdown_signal in ShutdownSignal
        }

        with ShutdownSignalListener(request=ShutdownRequest()):
            pass

        for shutdown_signal, handler_before in handlers_before.items():
            assert signal.getsignal(shutdown_signal.value) is handler_before

    def test_leaving_restores_the_signals_rclpy_handles(self, init_rospy):
        options_before = get_current_signal_handlers_options()

        with ShutdownSignalListener(request=ShutdownRequest()):
            pass

        assert get_current_signal_handlers_options() == options_before


# %% keeping ros alive


def test_an_interrupt_leaves_ros_running(init_rospy):
    """
    The zero velocities are published over the very publishers a ros shutdown destroys,
    so the interrupt must not take the rclpy context down before they were sent.
    """
    with ShutdownSignalListener(request=ShutdownRequest()):
        signal.raise_signal(signal.SIGINT)

        assert rclpy.ok()
