#! /usr/bin/env python3
import os

# %% Imports

from math import pi

from coraplex.alternative_motion_mappings.daisy_motion_mapping import (
    DAiSyGripMotion,
    DAiSyFlexGripMotion,
    SetDAiSyGripAction,
    SetDAiSyFlexGripAction,
)
from coraplex.datastructures.enums import (
    Arms,
    ExecutionType,
    CuttingTechnique,
    WPGGripPreset,
)
from coraplex.execution_environment import simulated_robot, real_robot, semi_real_robot
from coraplex.plans.factories import sequential
from coraplex.robot_plans import MoveGripperMotion, MoveJointsMotion
from coraplex.robot_plans.actions.composite.tool_based import CuttingAction
from coraplex.robot_plans.actions.core.robot_body import (
    ParkArmsAction,
    SetGripperAction,
)
from coraplex.testing import attach_tool
from define_real_daisy import setup_real_daisy
from define_sim_daisy import setup_sim_daisy
from experiments.tool_based_actions.simple_demo.demo_world import (
    parse_object,
    BREAD_COLOR,
)
from semantic_digital_twin.adapters.mesh import STLParser
from semantic_digital_twin.datastructures.definitions import GripperState
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.exceptions import (
    WorldEntityNotFoundError,
    SemanticAnnotationNotInWorldError,
)
from semantic_digital_twin.robots.daisy import DAiSy
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    CuttingKnife,
    Bread,
)
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.connections import FixedConnection

verbose = True
collision_avoidance = False
execution_mode = ExecutionType.REAL
# execution_mode = ExecutionType.SEMI_REAL

print(f"Running in: {execution_mode}")

# %% Robot and World Setup
if execution_mode == ExecutionType.REAL or execution_mode == ExecutionType.SEMI_REAL:
    node, world, robot_view, context = setup_real_daisy()
else:
    node, world, robot_view, context = setup_sim_daisy()

try:
    cable_post = world.get_bodies_by_name(PrefixedName("item_profile_8_40x40_720.stl"))[
        0
    ]
except (WorldEntityNotFoundError, IndexError):
    cable_post = STLParser(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "resources",
            "objects",
            "item_profile_8_40x40_720.stl",
        )
    ).parse()
    cable_post_root = cable_post.root

    with world.modify_world():
        world.merge_world(
            cable_post,
            FixedConnection(
                world.get_semantic_annotations_by_type(DAiSy)[0].root,
                cable_post_root,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=0.62,
                    y=0.42,
                    z=0.8477,
                    roll=pi / 2,
                    yaw=-pi / 2,
                    reference_frame=world.get_semantic_annotations_by_type(DAiSy)[
                        0
                    ].root,
                ),
            ),
        )

try:
    cable_hanger = world.get_bodies_by_name(PrefixedName("cable_hanger_2.stl"))[0]
except (WorldEntityNotFoundError, IndexError):
    cable_hanger = STLParser(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "resources",
            "objects",
            "cable_hanger_2.stl",
        )
    ).parse()
    cable_hanger_root = cable_hanger.root

    with world.modify_world():
        world.merge_world(
            cable_hanger,
            FixedConnection(
                cable_post_root,
                cable_hanger_root,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=0.0,
                    y=0.310,  # 720/2 - 50
                    z=0.02,
                    roll=-pi / 2,
                    # pitch=-pi / 2,
                    reference_frame=cable_post_root,
                ),
            ),
        )

try:
    bread_body = world.get_body_by_name(PrefixedName("bread.stl"))
except (WorldEntityNotFoundError, IndexError):
    bread_world = parse_object("bread.stl", color=BREAD_COLOR)
    with world.modify_world():
        world.merge_world(
            bread_world,
            FixedConnection(
                world.get_semantic_annotations_by_type(DAiSy)[0].root,
                bread_world.get_body_by_name("bread.stl"),
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_quaternion(
                    pos_x=0.37,
                    pos_y=0.45,
                    pos_z=0.655,
                    quat_w=0.707,
                    quat_x=0,
                    quat_y=0,
                    quat_z=-0.707,
                    reference_frame=world.get_semantic_annotations_by_type(DAiSy)[
                        0
                    ].root,
                ),
            ),
        )
    bread_body = world.get_body_by_name("bread.stl")

try:
    knife_body = world.get_body_by_name(PrefixedName("big-knife.stl"))
except (WorldEntityNotFoundError, IndexError):
    knife_body = attach_tool(
        world,
        robot_view,
        Arms.RIGHT,
        parse_object("big-knife.stl"),
        {"y": 0.08, "z": 0.02, "roll": pi, "pitch": 0, "yaw": pi / 2},
    )

knife_name = PrefixedName("knife")
bread_name = PrefixedName("bread")

try:
    knife = world.get_semantic_annotations_by_type(CuttingKnife)[0]
except (WorldEntityNotFoundError, IndexError):
    knife = CuttingKnife(root=knife_body, name=knife_name)
    with world.modify_world():
        world.add_semantic_annotations([knife])

try:
    bread = world.get_semantic_annotations_by_type(Bread)[0]
except (WorldEntityNotFoundError, IndexError):
    bread = Bread(root=bread_body, name=bread_name)
    with world.modify_world():
        world.add_semantic_annotations([bread])

context.evaluate_conditions = False

daisy_left_arm_names = [
    "left_shoulder_pan_joint",
    "left_shoulder_lift_joint",
    "left_elbow_joint",
    "left_wrist_1_joint",
    "left_wrist_2_joint",
    "left_wrist_3_joint",
]

daisy_safe_left_arm_positions = [
    -2.71,  # left_shoulder_pan_joint
    -1.01,  # left_shoulder_lift_joint
    -2.10,  # left_elbow_joint
    -1.59,  # left_wrist_1_joint
    1.53,  # left_wrist_2_joint
    -4.23,  # left_wrist_3_joint
]

daisy_right_arm_names = [
    "right_shoulder_pan_joint",
    "right_shoulder_lift_joint",
    "right_elbow_joint",
    "right_wrist_1_joint",
    "right_wrist_2_joint",
    "right_wrist_3_joint",
]

daisy_safe_right_arm_positions = [
    -0.02,  # right_shoulder_pan_joint
    -0.97,  # right_shoulder_lift_joint
    -2.00,  # right_elbow_joint
    -1.76,  # right_wrist_1_joint
    1.56,  # right_wrist_2_joint
    -0.83,  # right_wrist_3_joint
]

plan = sequential(
    [
        # SetDAiSyGripAction(motion=GripperState.OPEN, gripper=Arms.BOTH),
        ParkArmsAction(arm=Arms.RIGHT),
        MoveJointsMotion(
            names=daisy_left_arm_names, positions=daisy_safe_left_arm_positions
        ),
        # SetDAiSyFlexGripAction(
        #     motion=GripperState.FLEXCLOSE,
        #     gripper=Arms.RIGHT,
        #     grip_force=150,
        #     grip_position=2,
        #     grip_acceleration=1200,
        #     grip_speed=52,
        # ),
        CuttingAction(
            object_to_cut=bread_body,
            arm=Arms.RIGHT,
            tool=knife,
            technique=CuttingTechnique.SLICE,
            number_of_cuts_on_local_x_axis=3,
            slice_thickness=0.03,
        ),
        ParkArmsAction(arm=Arms.RIGHT),
        # SetDAiSyGripAction(motion=GripperState.OPEN, gripper=Arms.RIGHT),
    ],
    context,
)

if execution_mode == ExecutionType.REAL:
    with real_robot(collision_avoidance=collision_avoidance):
        plan.perform()
elif execution_mode == ExecutionType.SEMI_REAL:
    with semi_real_robot(collision_avoidance=collision_avoidance):
        plan.perform()
else:
    with simulated_robot:
        plan.perform()

print("Plan finished.")
