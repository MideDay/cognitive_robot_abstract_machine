from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Any

from typing_extensions import TYPE_CHECKING

from giskardpy.motion_statechart.graph_node import MotionStatechartNode
from giskardpy.motion_statechart.motion_statechart import (
    MotionStatechart,
)
from coraplex.datastructures.enums import ExecutionType
from coraplex.plans.executables import GiskardExecutable

from semantic_digital_twin.world import World

if TYPE_CHECKING:
    from coraplex.plans.plan_node import PlanNode

logger = logging.getLogger(__name__)


@dataclass
class MotionExecutor:
    """
    Deprecated: The construction and execution of the motion state chart moved to
    :py:class:`pycram.plans.executables.GiskardExecutable`.
    This class only remains for the generated ORM interface.
    """

    motions: List[MotionStatechartNode]
    """
    The motions to execute.
    """

    world: World
    """
    The world in which the motions should be executed.
    """

    motion_state_chart: MotionStatechart = field(init=False)
    """
    Giskard's motion state chart that is created from the motions.
    """

    ros_node: Any = field(kw_only=True, default=None)
    """
    ROS node that should be used for communication.

    Only relevant for real execution.
    """

    plan_node: PlanNode = field(kw_only=True)
    """
    The plan node that created this executor.
    """


@dataclass
class ExecutionEnvironment:
    """
    Base class for managing execution context of all actions within.

    Instances of this class is to be used with a "with" context block

    Example:

        >>> with ExecutionEnvironment(ExecutionType.SIMULATED):
        >>>     SequentialPlan(context, NavigateActionDescription, ...)
    """

    execution_type: ExecutionType
    """
    The type of the execution environment.
    """

    previous_type: ExecutionType = field(init=False, default=None)
    """
    Type of the execution environment before setting it, used for nested
    environments.
    """

    def __enter__(self):
        """
        Entering function for 'with' scope, saves the previously set
        :py:attr:`~pycram.plans.executables.GiskardExecutable.execution_type` and sets it to the type of this
        environment.
        """
        self.previous_type = GiskardExecutable.execution_type
        GiskardExecutable.execution_type = self.execution_type

    def __exit__(self, _type, value, traceback):
        """
        Exit method for the 'with' scope, sets the
        :py:attr:`~pycram.plans.executables.GiskardExecutable.execution_type` to the previously used one.
        """
        GiskardExecutable.execution_type = self.previous_type

    def __call__(self):
        return self


# These are imported, so they don't have to be initialized when executing with
simulated_robot = ExecutionEnvironment(ExecutionType.SIMULATED)
real_robot = ExecutionEnvironment(ExecutionType.REAL)
semi_real_robot = ExecutionEnvironment(ExecutionType.SEMI_REAL)
no_execution = ExecutionEnvironment(ExecutionType.NO_EXECUTION)
