"""Point cloud filtering based on semantic regions.

This module provides an annotator for:

* Filtering point clouds using regions from object/region knowledge
* Creating region hypotheses with poses
* Visualizing filtered regions and clouds

.. note::
   Regions are defined in a shared semantic_digital_twin World.
"""

from timeit import default_timer

import open3d as o3d
import py_trees

import robokudo.annotators
import robokudo.annotators.core
import robokudo.types.scene
import robokudo.utils.annotator_helper
import robokudo.utils.error_handling
import robokudo.utils.knowledge
from robokudo.cas import CASViews
from robokudo.utils import module_loader
from robokudo.utils.region import region_obb_in_cam_coordinates, region_pose_annotation
from semantic_digital_twin.world_description.world_entity import Region


class RegionFilter(robokudo.annotators.core.ThreadedAnnotator):
    """Point cloud filtering using world regions.

    The RegionFilter can be used to filter point clouds based on an environment model
    that provides semantic_digital_twin Regions via a shared World.
    """

    class Descriptor(robokudo.annotators.core.BaseAnnotator.Descriptor):
        """Configuration descriptor for region filtering."""

        class Parameters:
            """Parameters for configuring region filtering.
            , defaults to "map"
            :ivar object_knowledge_ros_package: , defaults to "robokudo"
            :ivar object_knowledge_name: , defaults to "object_knowledge_iai_kitchen20"
            :ivar active_region: , empty for all regions
            """

            def __init__(self) -> None:
                self.world_frame_name: str = "map"
                """Name of the world coordinate frame"""

                self.object_knowledge_base_ros_package: str = "robokudo"
                """If you use SDT object knowledge to generate the detection, provide the knowledge base package name here"""

                self.object_knowledge_base_name: str = "object_knowledge_iai_kitchen20"
                """If you use SDT object knowledge to generate the detection, provide the knowledge base name here"""

                self.active_region: str = ""
                """Name of active region to filter. Does not define a specific region but can be used to check the active regions."""

        # Overwrite the parameters explicitly to enable auto-completion
        parameters = Parameters()

    def __init__(
            self,
            name: str = "RegionFilter",
            descriptor: "RegionFilter.Descriptor" = Descriptor(),
    ) -> None:
        """Initialize the region filter.

        :param name: Name of this annotator instance, defaults to "RegionFilter"
        :param descriptor: Configuration descriptor, defaults to Descriptor()
        """
        super().__init__(name=name, descriptor=descriptor)

        self.world_frame_name: str = self.descriptor.parameters.world_frame_name
        """Name of the world coordinate frame read from parameters"""

        self.object_knowledge_base = None
        """The object/region knowledge base instance"""

        self.load_object_knowledge_base()

        self.active_region = self.descriptor.parameters.active_region
        """Name of the active region to filter read from parameters"""

    def load_object_knowledge_base(self) -> None:
        """Load object/region knowledge from configured package and module.
        """
        self.object_knowledge_base = robokudo.utils.knowledge.load_object_knowledge_base(self)

    @robokudo.utils.error_handling.catch_and_raise_to_blackboard
    def compute(self) -> py_trees.common.Status:
        """Filter point cloud using world regions.

        The method:

        * Loads point cloud and query from CAS
        * Loads object/region knowledge and active regions
        * For each active region:
          * Transforms region to camera frame
          * Creates oriented bounding box
          * Filters points within box
          * Creates region hypothesis with pose
        * Updates CAS with filtered cloud

        :return: SUCCESS after processing
        :raises Exception: If queried region not found in map
        """
        start_timer = default_timer()
        cloud = self.get_cas().get(CASViews.CLOUD)

        query = None
        if self.get_cas().contains(CASViews.QUERY):
            query = self.get_cas().get(CASViews.QUERY)

        self.load_object_knowledge_base()

        regions = self.object_knowledge_base.world.get_kinematic_structure_entity_by_type(
            Region
        )
        all_regions = {str(region.name): region for region in regions}
        active_regions = all_regions

        # Overwrite active_regions, if location is explicitly mentioned
        if query is not None:
            queried_location = query.obj.location
            if queried_location != "" and queried_location != self.active_region:
                self.rk_logger.info(
                    f"Setting region filter to check for location '{queried_location}'"
                )
                active_regions = dict()
                try:
                    active_regions[queried_location] = all_regions[queried_location]
                except KeyError:
                    raise Exception(
                        f"Couldn't find requested location {queried_location} in object/region knowledge"
                    )

        visualized_geometries = []
        filtered_indices = set()

        self.rk_logger.debug(f"Analyzing {len(active_regions.keys())}")

        try:
            world_to_cam_transform = (
                robokudo.utils.annotator_helper.get_world_to_cam_transform_matrix(
                    self.get_cas()
                )
            )
        except KeyError as err:
            self.rk_logger.warning(f"Couldn't find viewpoint in the CAS: {err}")
            return py_trees.common.Status.FAILURE

        for key, region in active_regions.items():
            assert isinstance(region, Region)
            # Will be used for saving the indices of the cloud for this specific region
            filtered_indices_for_this_region = set()
            # RegionHypothetis for this specific region
            region_hypothesis = robokudo.types.scene.RegionHypothesis()

            obb = region_obb_in_cam_coordinates(
                self.object_knowledge_base.world, region, world_to_cam_transform
            )
            region_hypothesis.annotations.append(region_pose_annotation(region))

            # To avoid copying the whole points per region into a new cloud, we'll only collect and save
            # the indices of matching points
            region_indices = obb.get_point_indices_within_bounding_box(cloud.points)
            filtered_indices.update(region_indices)

            # Update the indices which belong to the region and save the cloud and name of region in RegionHypothesis
            filtered_indices_for_this_region.update(region_indices)
            filtered_cloud_for_this_region = cloud.select_by_index(
                list(filtered_indices_for_this_region)
            )
            region_hypothesis.points = filtered_cloud_for_this_region
            region_hypothesis.name = str(region.name)
            self.get_cas().annotations.append(region_hypothesis)

            visualized_geometries.append({"name": str(region.name), "geometry": obb})

        # Place the filtered PointCloud into the CAS, overwriting the previous one
        filtered_cloud = cloud.select_by_index(list(filtered_indices))
        self.get_cas().set(CASViews.CLOUD, filtered_cloud)

        visualized_geometries.append(
            {"name": "filtered cloud", "geometry": filtered_cloud}
        )

        world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
        visualized_geometries.append(
            {
                "name": "world_frame",
                "geometry": world_frame.transform(world_to_cam_transform),
            }
        )
        self.get_annotator_output_struct().set_geometries(visualized_geometries)

        end_timer = default_timer()
        self.feedback_message = f"Processing took {(end_timer - start_timer):.4f}s"
        return py_trees.common.Status.SUCCESS
