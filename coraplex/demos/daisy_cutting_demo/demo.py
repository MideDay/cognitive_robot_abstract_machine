#! /usr/bin/env python3

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
from coraplex.execution_environment import simulated_robot
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
from semantic_digital_twin.datastructures.definitions import GripperState
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    CuttingKnife,
    Bread,
)
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix

verbose = True
collision_avoidance = False
execution_mode = ExecutionType.SIMULATED
# execution_mode = ExecutionType.SEMI_REAL

print(f"Running in: {execution_mode}")

# %% Robot and World Setup
if execution_mode == ExecutionType.REAL or execution_mode == ExecutionType.SEMI_REAL:
    node, world, robot_view, context = setup_real_daisy()
else:
    node, world, robot_view, context = setup_sim_daisy()

bread_world = parse_object("bread.stl", color=BREAD_COLOR)
with world.modify_world():
    world.merge_world_at_pose(
        bread_world,
        HomogeneousTransformationMatrix.from_xyz_quaternion(
            pos_x=0.5,
            pos_y=0.5,
            pos_z=0.65,
            quat_w=0.707,
            quat_x=0,
            quat_y=0,
            quat_z=-0.707,
            reference_frame=world.root,
        ),
    )

knife_body = attach_tool(
    world,
    robot_view,
    Arms.RIGHT,
    parse_object("big-knife.stl"),
    {"y": 0.08, "z": 0.02, "roll": pi, "pitch": 0, "yaw": pi / 2},
)
bread_body = world.get_body_by_name("bread.stl")

knife = CuttingKnife(root=knife_body)
with world.modify_world():
    world.add_semantic_annotations([Bread(root=bread_body), knife])

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
        SetDAiSyGripAction(motion=GripperState.OPEN, gripper=Arms.BOTH),
        ParkArmsAction(arm=Arms.RIGHT),
        MoveJointsMotion(
            names=daisy_left_arm_names, positions=daisy_safe_left_arm_positions
        ),
        SetDAiSyGripAction(motion=GripperState.CLOSE, gripper=Arms.RIGHT),
        CuttingAction(
            object_to_cut=bread_body,
            arm=Arms.RIGHT,
            tool=knife,
            technique=CuttingTechnique.SLICE,
            number_of_cuts_on_local_x_axis=3,
            slice_thickness=0.03,
        ),
    ],
    context,
)

with simulated_robot:
    plan.perform()

print("Plan finished.")
