from dataclasses import dataclass

from giskardpy.model.collision_world_syncer import CollisionWorldSynchronizer
from giskardpy.motion_statechart.auxilary_variable_manager import (
    AuxiliaryVariableManager,
)
from giskardpy.qp.qp_controller_config import QPControllerConfig
from semantic_digital_twin.world import World


@dataclass
class BuildContext:
    world: World
    auxiliary_variable_manager: AuxiliaryVariableManager
    collision_scene: CollisionWorldSynchronizer
    qp_controller_config: QPControllerConfig

    def to_execution_context(self):
        return ExecutionContext(world=self.world)


@dataclass
class ExecutionContext:
    world: World
