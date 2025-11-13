import os

import pytest

from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import Vector3, TransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    OmniDrive,
    RevoluteConnection,
    FixedConnection,
)
from semantic_digital_twin.world_description.world_entity import Body


@pytest.fixture
def pr2_world():
    urdf_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "resources",
        "urdf",
    )
    pr2 = os.path.join(urdf_dir, "pr2_kinematic_tree.urdf")
    pr2_parser = URDFParser.from_file(file_path=pr2)
    world_with_pr2 = pr2_parser.parse()
    with world_with_pr2.modify_world():
        pr2_root = world_with_pr2.root
        localization_body = Body(name=PrefixedName("odom_combined"))
        world_with_pr2.add_kinematic_structure_entity(localization_body)
        c_root_bf = OmniDrive.create_with_dofs(
            parent=localization_body, child=pr2_root, world=world_with_pr2
        )
        world_with_pr2.add_connection(c_root_bf)

    return world_with_pr2


@pytest.fixture()
def mini_world():
    world = World()
    with world.modify_world():
        body = Body(name=PrefixedName("root"))
        body2 = Body(name=PrefixedName("tip"))
        connection = RevoluteConnection.create_with_dofs(
            world=world, parent=body, child=body2, axis=Vector3.Z()
        )
        world.add_connection(connection)
    return world


@pytest.fixture()
def box_bot_world():
    world = World()
    with world.modify_world():
        body = Body(name=PrefixedName("map"))
        body2 = Body(name=PrefixedName("bot"))
        connection = OmniDrive.create_with_dofs(world=world, parent=body, child=body2)
        world.add_connection(connection)

        environment = Body(name=PrefixedName("environment"))
        env_connection = FixedConnection(
            parent=body,
            child=environment,
            parent_T_connection_expression=TransformationMatrix.from_xyz_rpy(1),
        )
        world.add_connection(env_connection)
    return world
