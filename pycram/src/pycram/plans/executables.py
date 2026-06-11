from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import List, Dict, TYPE_CHECKING

from giskardpy.motion_statechart.goals.templates import Parallel
from giskardpy.motion_statechart.graph_node import Task
from semantic_digital_twin.world_description.connections import Connection6DoF
from semantic_digital_twin.world_description.world_entity import Body

if TYPE_CHECKING:

    from plans.condition_nodes import ConditionNode
    from plans.plan_node import MotionNode
    from datastructures.dataclasses import Context


@dataclass
class Executable:
    """
    Base class for executable units.
    """

    execution_list: List[Executable] = field(default_factory=list)

    context: Context = field(kw_only=True)

    def execute(self) -> None:
        """
        Executes the unit.
        """
        for e in self.execution_list:
            e.execute()


@dataclass
class GiskardExecutable(Executable):
    motion_mappings: Dict[MotionNode, Task] = field(kw_only=True)

    def execute(self) -> None:
        pass


@dataclass
class LanguageExecutable(Executable):

    @property
    def motion_state_chart(self):
        return Parallel(
            nodes=[
                motion.motion_node.designator.motion_chart for motion in self.motions
            ]
        )


@dataclass
class ConditionExecutable(Executable):
    """
    An executable unit for a condition node.
    """

    condition_node: ConditionNode = field(kw_only=True)
    """
    The condition node to execute.
    """

    def execute(self) -> None:
        """
        Executes the condition node.
        """
        pass


@dataclass
class MotionExecutable(GiskardExecutable): ...


@dataclass
class ModelChangeExecutable(Executable):

    body: Body = field(kw_only=True)

    new_parent: Body = field(kw_only=True)

    def execute(self) -> None:
        obj_transform = self.context.world.compute_forward_kinematics(
            self.new_parent, self.body
        )
        with self.context.world.modify_world():
            self.context.world.remove_connection(self.body.parent_connection)
            connection = Connection6DoF.create_with_dofs(
                parent=self.new_parent, child=self.body, world=self.context.world
            )
            self.context.world.add_connection(connection)
            connection.origin = obj_transform
