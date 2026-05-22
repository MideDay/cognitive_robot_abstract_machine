from __future__ import annotations

import os

import rerun as rr
import trimesh
from typing_extensions import Callable, List, Optional

from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.geometry import Mesh, Shape
from semantic_digital_twin.world_description.world_entity import Body

DEFAULT_ROOT_ENTITY_PATH = "world"
"""Default Rerun entity path under which the kinematic tree is logged."""

MeshFileResolver = Callable[[Body, Mesh], Optional[str]]
"""Hook mapping a ``(body, mesh)`` to a local mesh file path, or ``None``.
Used to recover material color when geometry was mirrored over ROS without it.
"""


def _entity_path(root_entity_path: str, body: Body) -> str:
    """Return the Rerun entity path for a body."""
    return f"{root_entity_path}/{body.name.name}"


def _visual_shapes(body: Body) -> List[Shape]:
    """Return a body's visual shapes, falling back to collision shapes."""
    return body.visual.shapes if body.visual.shapes else body.collision.shapes


def _colored_source_mesh(
    shape: Shape,
    body: Optional[Body],
    mesh_file_resolver: Optional[MeshFileResolver],
) -> trimesh.Trimesh:
    """Return the shape geometry, loading a local file for color when resolved."""
    if isinstance(shape, Mesh) and body is not None and mesh_file_resolver is not None:
        local_path = mesh_file_resolver(body, shape)
        if local_path is not None and os.path.exists(local_path):
            source_mesh = trimesh.load_mesh(local_path)
            source_mesh.apply_scale(shape.scale.to_np())
            return source_mesh
    return shape.mesh.copy()


def shape_to_link_frame_mesh(
    shape: Shape,
    body: Optional[Body] = None,
    mesh_file_resolver: Optional[MeshFileResolver] = None,
) -> trimesh.Trimesh:
    """
    Convert a shape to a trimesh expressed in its link frame.

    The shape's local origin is baked into the vertices and Collada
    ``TextureVisuals`` are normalized to per-vertex ``ColorVisuals``.

    :param shape: The shape to convert.
    :param body: The body owning the shape, required only when using a resolver.
    :param mesh_file_resolver: Optional hook to load a local file for color.
    :return: A trimesh whose vertices are expressed in the parent link frame.
    """
    link_frame_mesh = _colored_source_mesh(shape, body, mesh_file_resolver)
    link_frame_mesh.apply_transform(shape.origin.to_np())
    if hasattr(link_frame_mesh.visual, "to_color"):
        link_frame_mesh.visual = link_frame_mesh.visual.to_color()
    return link_frame_mesh


def log_model(
    world: World,
    root_entity_path: str = DEFAULT_ROOT_ENTITY_PATH,
    *,
    mesh_file_resolver: Optional[MeshFileResolver] = None,
    recording: Optional[rr.RecordingStream] = None,
) -> None:
    """
    Log the static visual geometry of every body in a world to Rerun.

    Geometry is logged as static :class:`rerun.Mesh3D` in each body's local
    frame, so only the per-body transforms need re-logging on a state change.

    :param world: The world whose geometry is logged.
    :param root_entity_path: Entity path under which the tree is logged.
    :param mesh_file_resolver: Optional hook to load local files for color.
    :param recording: Target recording stream; ``None`` uses the active one.
    """
    rr.log(
        root_entity_path,
        rr.ViewCoordinates.RIGHT_HAND_Z_UP,
        static=True,
        recording=recording,
    )
    for body in world.bodies:
        entity_path = _entity_path(root_entity_path, body)
        for index, shape in enumerate(_visual_shapes(body)):
            link_frame_mesh = shape_to_link_frame_mesh(shape, body, mesh_file_resolver)
            rr.log(
                f"{entity_path}/visual_{index}",
                rr.Mesh3D(
                    vertex_positions=link_frame_mesh.vertices,
                    triangle_indices=link_frame_mesh.faces,
                    vertex_normals=link_frame_mesh.vertex_normals,
                    vertex_colors=link_frame_mesh.visual.vertex_colors,
                ),
                static=True,
                recording=recording,
            )


def log_state(
    world: World,
    root_entity_path: str = DEFAULT_ROOT_ENTITY_PATH,
    *,
    static: bool = False,
    recording: Optional[rr.RecordingStream] = None,
) -> None:
    """
    Log the current forward-kinematics transform of every body to Rerun.

    When ``static`` is ``True`` the transforms overwrite the previous values
    with no timeline history, giving a constant-memory, current-state-only view.

    :param world: The world whose state is logged.
    :param root_entity_path: Entity path under which the tree is logged.
    :param static: Whether to log without timeline history (overwrite in place).
    :param recording: Target recording stream; ``None`` uses the active one.
    """
    for body in world.bodies:
        world_transform_body = world.compute_forward_kinematics_np(world.root, body)
        rr.log(
            _entity_path(root_entity_path, body),
            rr.Transform3D(
                translation=world_transform_body[:3, 3],
                mat3x3=world_transform_body[:3, :3],
            ),
            static=static,
            recording=recording,
        )


def log_world(
    world: World,
    root_entity_path: str = DEFAULT_ROOT_ENTITY_PATH,
    *,
    mesh_file_resolver: Optional[MeshFileResolver] = None,
    recording: Optional[rr.RecordingStream] = None,
) -> None:
    """
    Log a full snapshot of a world (geometry plus current state) to Rerun.

    :param world: The world to log.
    :param root_entity_path: Entity path under which the tree is logged.
    :param mesh_file_resolver: Optional hook to load local files for color.
    :param recording: Target recording stream; ``None`` uses the active one.
    """
    log_model(
        world,
        root_entity_path,
        mesh_file_resolver=mesh_file_resolver,
        recording=recording,
    )
    log_state(world, root_entity_path, recording=recording)
