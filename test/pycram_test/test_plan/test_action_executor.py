import time

from pycram.datastructures.enums import Arms, ApproachDirection, VerticalAlignment
from pycram.datastructures.grasp import GraspDescription
from pycram.plans.factories import sequential
from pycram.robot_plans.actions.core.pick_up import PickUpAction


def test_sub_action_expansion(immutable_model_world):
    world, view, context = immutable_model_world

    plan = sequential(
        [
            PickUpAction(
                object_designator=world.get_body_by_name("milk.stl"),
                arm=Arms.RIGHT,
                grasp_description=GraspDescription(
                    ApproachDirection.FRONT,
                    vertical_alignment=VerticalAlignment.NoAlignment,
                    end_effector=view.right_arm.end_effector,
                ),
            )
        ],
        context=context,
    )

    pick_node = plan.children[0]
    pick_node.notify()

    parser = ActionGraphParser(pick_node)
    expanded_children = parser.parse_children(pick_node.children)
    assert len(expanded_children) > 1
