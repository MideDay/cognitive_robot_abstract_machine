#! /usr/bin/env python3
"""
DAiSy takes a knife out of its holder, slices bread with it and puts it back, places the
bread in a bowl, and mixes the bowl's contents with a second knife from another holder.

Run it from this folder::

    python demo.py
"""

from __future__ import annotations

# %% Imports

from dataclasses import dataclass, field
from datetime import timedelta
from enum import StrEnum
from math import pi
from pathlib import Path

from krrood.patterns.path_enumeration import PathEnumeration

from coraplex.datastructures.enums import CuttingTechnique
from coraplex.demonstrations import RobotDemonstration
from coraplex.plans.context_extensions import RobotAccess
from coraplex.plans.executors import KinematicallySimulatedPlanExecutor, PlanExecutor
from coraplex.robot_plans.actions.composite.tool_based import (
    CuttingAction,
    MixingAction,
)
from coraplex.robot_plans.actions.composite.tool_paths import (
    ToolPath,
    build_container_path,
    build_cutting_path,
)
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.actions.core.placing import PlaceAction
from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction
from cramph.composites import Sequence
from cramph.context import ContextExtension, StatechartContext
from cramph.statechart import Statechart
from experiments.tool_based_actions.simple_demo.demo_world import (
    BOWL_COLOR,
    BREAD_COLOR,
    OBJECTS_DIRECTORY,
)
from giskardpy.motion_statechart.tasks.cartesian_tasks import CartesianPose
from giskardpy.motion_statechart.tasks.joint_tasks import JointPositionList
from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.grasping.grasp_candidates import GraspCandidate
from semantic_digital_twin.robots.daisy import DAiSy
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Bowl,
    Bread,
    CuttingKnife,
    ToolWithHandle,
    Whisk,
)
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.spatial_types.spatial_types import Point3, Pose
from semantic_digital_twin.specifications.kinematic_structure_entities import (
    BodySpecification,
)
from semantic_digital_twin.specifications.robots import RobotSpecification
from semantic_digital_twin.specifications.semantic_annotations import (
    SemanticAnnotationWithRootSpecification,
)
from semantic_digital_twin.specifications.worlds import WorldSpecification
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body

# %% The scene


class SceneFile(PathEnumeration):
    """
    The meshes the scene is built from, under the coraplex resources.
    """

    CABLE_POST = Path(OBJECTS_DIRECTORY, "item_profile_8_40x40_720.stl")
    """
    The aluminium profile standing on the table.
    """

    CABLE_HANGER = Path(OBJECTS_DIRECTORY, "cable_hanger_2.stl")
    """
    The hanger mounted on the cable post.
    """

    TOOL_HOLDER = Path(OBJECTS_DIRECTORY, "tool_holder.stl")
    """
    A holder a knife stands in blade-down.
    """

    KNIFE = Path(OBJECTS_DIRECTORY, "big-knife.stl")
    """
    The knife both tools are made from.
    """

    BREAD = Path(OBJECTS_DIRECTORY, "bread.stl")
    """
    The loaf that is sliced.
    """

    BOWL = Path(OBJECTS_DIRECTORY, "bowl.stl")
    """
    The bowl the bread is put in and mixed.
    """


class SceneBody(StrEnum):
    """
    The names of the scene's bodies.
    """

    CABLE_POST = "cable_post"
    CABLE_HANGER = "cable_hanger"
    KNIFE_HOLDER = "knife_holder"
    MIXING_TOOL_HOLDER = "mixing_tool_holder"
    KNIFE = "knife"
    KNIFE_TOOL_FRAME = "knife_end_effector"
    """
    The cutting knife's tool frame: a :class:`ToolWithHandle` acts through the body
    named after its root with the suffix ``_end_effector``.
    """

    MIXING_TOOL = "mixing_tool"
    MIXING_TOOL_FRAME = "mixing_tool_end_effector"
    """
    The mixing tool's tool frame, named the way :attr:`KNIFE_TOOL_FRAME` is.
    """

    BREAD = "bread"
    BOWL = "bowl"


@dataclass
class ToolHolder:
    """
    A holder standing on DAiSy's table, with a knife standing blade-down in it.
    """

    name: SceneBody
    """
    The name of the holder's body.
    """

    x: float
    """
    Position of the holder along the x axis of DAiSy's table.
    """

    y: float
    """
    Position of the holder along the y axis of DAiSy's table.
    """

    table_top_height: float = 0.608
    """
    Height of the table top the holder stands on.
    """

    knife_height: float = 0.02 + 0.184
    """
    Height of the knife's origin above the holder's origin: the holder floor plus the
    distance from the knife's origin to its tip.
    """

    knife_centering_offset: float = -0.0045
    """
    Shift along the table's x axis that centres the blade, whose height is not symmetric
    about the knife's origin, in the holder opening.
    """

    @property
    def specification(self) -> BodySpecification:
        """
        The holder's body, standing on the table top.
        """
        return BodySpecification.mesh(
            self.name,
            str(SceneFile.TOOL_HOLDER),
            parent_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
                x=self.x, y=self.y, z=self.table_top_height
            ),
        )

    @property
    def holder_T_knife(self) -> HomogeneousTransformationMatrix:
        """
        The pose of a knife standing blade-down in this holder, relative to the holder.
        """
        return HomogeneousTransformationMatrix.from_xyz_rpy(
            x=self.knife_centering_offset, z=self.knife_height, pitch=pi / 2, yaw=pi
        )

    def knife_pose_in(self, world: World) -> Pose:
        """
        :param world: The world holding this holder's body.
        :return: :attr:`holder_T_knife`, relative to this holder's body in ``world``.
        """
        return self.holder_T_knife.copy_with_new_reference_frames(
            world.get_body_by_name(self.name), None
        ).pose


# %% The demonstration


@dataclass
class DaisyCuttingDemonstration(RobotDemonstration):
    """
    DAiSy slices bread with a knife, puts the bread into a bowl and mixes it there with
    a second knife, taking each knife from its holder and putting it back afterwards.

    The second knife stands in for a whisk: like a whisk it mixes pointing straight down.
    """

    used_robot: type[AbstractRobot] = DAiSy

    ros_node_name: str = "daisy_cutting_demo"

    executor_type: type[PlanExecutor] = KinematicallySimulatedPlanExecutor

    simulation_time_per_action: timedelta | None = timedelta(seconds=300)

    cable_post: BodySpecification = field(
        default_factory=lambda: BodySpecification.mesh(
            SceneBody.CABLE_POST,
            str(SceneFile.CABLE_POST),
            parent_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
                x=0.62, y=0.42, z=0.8477, roll=pi / 2, yaw=-pi / 2
            ),
            child_specifications=[
                BodySpecification.mesh(
                    SceneBody.CABLE_HANGER,
                    str(SceneFile.CABLE_HANGER),
                    parent_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
                        y=0.310,  # 720/2 - 50
                        z=0.02,
                        roll=-pi / 2,
                    ),
                )
            ],
        )
    )
    """
    The cable post on the table, with the cable hanger mounted on it.
    """

    bread: SemanticAnnotationWithRootSpecification = field(
        default_factory=lambda: SemanticAnnotationWithRootSpecification(
            name=SceneBody.BREAD,
            semantic_annotation_type=Bread,
            root_specification=BodySpecification.mesh(
                SceneBody.BREAD,
                str(SceneFile.BREAD),
                color=BREAD_COLOR,
                parent_T_self=HomogeneousTransformationMatrix.from_xyz_quaternion(
                    pos_x=0.37,
                    pos_y=0.45,
                    pos_z=0.655,
                    quat_w=0.707,
                    quat_x=0,
                    quat_y=0,
                    quat_z=-0.707,
                ),
            ),
        )
    )
    """
    The loaf that is sliced, lying on the table.
    """

    bowl: SemanticAnnotationWithRootSpecification = field(
        default_factory=lambda: SemanticAnnotationWithRootSpecification(
            name=SceneBody.BOWL,
            semantic_annotation_type=Bowl,
            root_specification=BodySpecification.mesh(
                SceneBody.BOWL,
                str(SceneFile.BOWL),
                color=BOWL_COLOR,
                parent_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=0.37, y=0.75, z=0.658
                ),
            ),
        )
    )
    """
    The bowl the bread is put in and mixed, standing on the table.
    """

    knife_holder: ToolHolder = field(
        default_factory=lambda: ToolHolder(SceneBody.KNIFE_HOLDER, x=0.55, y=0.30)
    )
    """
    The holder the cutting knife stands in.
    """

    mixing_tool_holder: ToolHolder = field(
        default_factory=lambda: ToolHolder(SceneBody.MIXING_TOOL_HOLDER, x=0.55, y=0.40)
    )
    """
    The holder the mixing knife stands in.
    """

    knife_tool_frame_along_blade: float = 0.1
    """
    Where the cutting knife's tool frame lies, measured from its origin along the blade.
    """

    mixing_tool_frame_along_blade: float = 0.15
    """
    Where the mixing knife's tool frame lies, measured from its origin along the blade,
    close enough to the tip that the blade stays above the bowl's floor while mixing.
    """

    knife_grasp_along_blade: float = -0.08
    """
    Where the fingers meet a knife's handle, measured from the knife's origin along the
    blade.
    """

    knife_grasp_across_blade: float = 0.0
    """
    Where the fingers meet a knife's handle, measured from the knife's origin along the
    blade's height.
    """

    knife_lift_height: float = 0.15
    """
    How far a knife rises above its holder when it is taken out or put back.
    """

    cutting_technique: CuttingTechnique = CuttingTechnique.SLICE
    """
    How the bread is cut.
    """

    number_of_cuts: int = 3
    """
    How many cuts are made along the bread's local x axis.
    """

    slice_thickness: float = 0.03
    """
    Thickness of every slice.
    """

    pre_cut_height: float = 0.10
    """
    How far above the start of the cut the knife waits, already turned the way it cuts.
    """

    pre_mix_height: float = 0.10
    """
    How far above the start of the mixing path the mixing knife waits, before it is
    lowered into the bowl.
    """

    bread_height_above_bowl_origin: float = 0.08
    """
    Height above the bowl's origin the bread is put down at.
    """

    safe_left_arm_positions: dict[str, float] = field(
        default_factory=lambda: {
            "left_shoulder_pan_joint": -2.71,
            "left_shoulder_lift_joint": -1.01,
            "left_elbow_joint": -2.10,
            "left_wrist_1_joint": -1.59,
            "left_wrist_2_joint": 1.53,
            "left_wrist_3_joint": -4.23,
        }
    )
    """
    Joint positions that keep the left arm out of the right arm's way.
    """

    # %% Scene

    def scene_specification(self) -> WorldSpecification:
        return WorldSpecification(
            objects=[
                self.cable_post,
                self.knife_holder.specification,
                self.mixing_tool_holder.specification,
                self.bread,
                self.bowl,
            ]
        )

    def robot_specification(self) -> RobotSpecification:
        return RobotSpecification(self.used_robot)

    def is_scene_populated(self, world: World) -> bool:
        return world.is_kinematic_structure_entity_in_world_by_name(SceneBody.BREAD)

    def populate_scene(self, world: World) -> None:
        """
        Stand the cutting knife and the mixing knife in their holders.
        """
        self._knife_specification(
            SceneBody.KNIFE,
            CuttingKnife,
            self.knife_holder,
            SceneBody.KNIFE_TOOL_FRAME,
            self.knife_tool_frame_along_blade,
        ).spawn(world, parent=world.get_body_by_name(self.knife_holder.name))
        self._knife_specification(
            SceneBody.MIXING_TOOL,
            Whisk,
            self.mixing_tool_holder,
            SceneBody.MIXING_TOOL_FRAME,
            self.mixing_tool_frame_along_blade,
        ).spawn(world, parent=world.get_body_by_name(self.mixing_tool_holder.name))

    def _knife_specification(
        self,
        name: SceneBody,
        tool_type: type[ToolWithHandle],
        holder: ToolHolder,
        tool_frame_name: SceneBody,
        tool_frame_along_blade: float,
    ) -> SemanticAnnotationWithRootSpecification:
        """
        :param name: The name of the knife and of its body.
        :param tool_type: The tool the knife is used as.
        :param holder: The holder the knife stands in.
        :param tool_frame_name: The name of the knife's tool frame.
        :param tool_frame_along_blade: Where the tool frame lies, measured from the
            knife's origin along the blade.
        :return: The knife standing in ``holder``, relative to the holder.
        """
        return SemanticAnnotationWithRootSpecification(
            name=name,
            semantic_annotation_type=tool_type,
            root_specification=BodySpecification.mesh(
                name,
                str(SceneFile.KNIFE),
                parent_T_self=holder.holder_T_knife,
                child_specifications=[
                    BodySpecification(
                        tool_frame_name,
                        parent_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
                            x=tool_frame_along_blade
                        ),
                    )
                ],
            ),
        )

    # %% Plan

    def build_context_extensions(self, world: World) -> list[ContextExtension]:
        return [RobotAccess(world.get_semantic_annotations_by_type(self.used_robot)[0])]

    def build_statechart(self, context: StatechartContext) -> Statechart:
        """
        Slice the bread with the cutting knife, put the knife back, move the bread into
        the bowl, and mix it there with the mixing knife before putting that back too.
        """
        world = context.world
        daisy = context.require_extension(RobotAccess).robot
        [knife] = world.get_semantic_annotations_by_type(CuttingKnife)
        [mixing_tool] = world.get_semantic_annotations_by_type(Whisk)
        [bread] = world.get_semantic_annotations_by_type(Bread)
        [bowl] = world.get_semantic_annotations_by_type(Bowl)
        pre_mix_pose = self._pre_mix_pose(world, bowl)
        statechart = Statechart(context=context)
        statechart.add_node(
            Sequence(
                nodes=[
                    ParkArmsAction(arms=[daisy.right_arm]),
                    JointPositionList(goal_state=self._safe_left_arm_state(world)),
                    PickUpAction(
                        grasp=self._knife_handle_grasp(knife),
                        arm=daisy.right_arm,
                        retreat_distance=self.knife_lift_height,
                    ),
                    CartesianPose(
                        root_link=daisy.root,
                        tip_link=knife.get_tool_frame(),
                        goal_pose=self._pre_cut_pose(bread),
                    ),
                    CuttingAction(
                        object_to_cut=bread.root,
                        arm=daisy.right_arm,
                        tool=knife,
                        technique=self.cutting_technique,
                        number_of_cuts_on_local_x_axis=self.number_of_cuts,
                        slice_thickness=self.slice_thickness,
                    ),
                    ParkArmsAction(arms=[daisy.right_arm]),
                    PlaceAction(
                        object_designator=knife,
                        target_location=self.knife_holder.knife_pose_in(world),
                        retreat_distance=self.knife_lift_height,
                    ),
                    ParkArmsAction(arms=[daisy.right_arm]),
                    PickUpAction(
                        grasp=GraspCandidate(
                            bread,
                            Pose.from_xyz_rpy(
                                pitch=pi / 2, yaw=pi, reference_frame=bread.root
                            ),
                        ),
                        arm=daisy.left_arm,
                    ),
                    PlaceAction(
                        object_designator=bread,
                        target_location=Pose.from_xyz_rpy(
                            z=self.bread_height_above_bowl_origin,
                            reference_frame=bowl.root,
                        ),
                    ),
                    ParkArmsAction(arms=[daisy.left_arm]),
                    PickUpAction(
                        grasp=self._knife_handle_grasp(mixing_tool),
                        arm=daisy.right_arm,
                        retreat_distance=self.knife_lift_height,
                    ),
                    CartesianPose(
                        root_link=daisy.root,
                        tip_link=mixing_tool.get_tool_frame(),
                        goal_pose=pre_mix_pose,
                    ),
                    MixingAction(
                        container=bowl.root, arm=daisy.right_arm, tool=mixing_tool
                    ),
                    CartesianPose(
                        root_link=daisy.root,
                        tip_link=mixing_tool.get_tool_frame(),
                        goal_pose=pre_mix_pose,
                    ),
                    PlaceAction(
                        object_designator=mixing_tool,
                        target_location=self.mixing_tool_holder.knife_pose_in(world),
                        retreat_distance=self.knife_lift_height,
                    ),
                    ParkArmsAction(arms=[daisy.right_arm]),
                ]
            )
        )
        return statechart

    def _safe_left_arm_state(self, world: World) -> JointState:
        """
        :param world: The world holding DAiSy.
        :return: The left arm at :attr:`safe_left_arm_positions`.
        """
        return JointState(
            connections=[
                world.get_connection_by_name(joint_name)
                for joint_name in self.safe_left_arm_positions
            ],
            target_values=list(self.safe_left_arm_positions.values()),
        )

    def _knife_handle_grasp(self, knife: ToolWithHandle) -> GraspCandidate:
        """
        :param knife: A knife standing in its holder.
        :return: The grasp that takes the knife by its handle from behind.
        """
        return GraspCandidate(
            knife,
            Pose.from_xyz_rpy(
                x=self.knife_grasp_along_blade,
                z=self.knife_grasp_across_blade,
                pitch=pi / 2,
                reference_frame=knife.root,
            ),
        )

    def _pre_cut_pose(self, bread: Bread) -> Pose:
        """
        :param bread: The bread to cut.
        :return: Where the knife's tool frame waits, :attr:`pre_cut_height` above the
            start of the cut, with its x axis along the bread's y axis and its z axis
            pointing up.
        """
        cut_start = self._path_start(
            build_cutting_path(
                bread.root,
                technique=self.cutting_technique,
                slice_thickness=self.slice_thickness,
                number_of_cuts_on_local_x_axis=self.number_of_cuts,
            ),
            bread.root,
        )
        return Pose.from_xyz_rpy(
            x=cut_start.x,
            y=cut_start.y,
            z=cut_start.z + self.pre_cut_height,
            yaw=pi / 2,
            reference_frame=bread.root,
        )

    def _pre_mix_pose(self, world: World, bowl: Bowl) -> Pose:
        """
        :param world: The world holding the bowl and the mixing tool holder.
        :param bowl: The bowl to mix in.
        :return: Where the mixing knife's tool frame waits, :attr:`pre_mix_height` above
            the start of the mixing path, blade down as it stands in its holder.
        """
        mix_start = world.transform(
            self._path_start(build_container_path(bowl.root), bowl.root), world.root
        )
        mix_start.z += self.pre_mix_height
        return Pose(
            position=mix_start,
            orientation=world.transform(
                self.mixing_tool_holder.knife_pose_in(world), world.root
            ).orientation,
            reference_frame=world.root,
        )

    @staticmethod
    def _path_start(tool_path: ToolPath, path_body: Body) -> Point3:
        """
        :param tool_path: A tool path laid out on ``path_body``.
        :param path_body: The body the tool path is laid out on.
        :return: The first point the tool is sent to, relative to ``path_body``.
        """
        _, points, _ = tool_path.sample(frame=HomogeneousTransformationMatrix())
        return Point3.from_iterable(points[0], reference_frame=path_body)


def main(
    executor_type: type[PlanExecutor] = KinematicallySimulatedPlanExecutor,
) -> World:
    """
    Run the demonstration.

    :param executor_type: The executor running the plan; a
        :class:`~coraplex.plans.executors.RealRobotPlanExecutor` drives the real DAiSy.
    :return: The world the demonstration acted on.
    """
    return DaisyCuttingDemonstration(executor_type=executor_type).run()


if __name__ == "__main__":
    main()
