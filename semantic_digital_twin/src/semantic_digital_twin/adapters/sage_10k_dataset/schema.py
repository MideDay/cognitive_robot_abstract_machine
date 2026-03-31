from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Self

import trimesh
from PIL import Image
from typing_extensions import Optional

from krrood.utils import get_full_class_name


from krrood.adapters.exceptions import JSON_TYPE_NAME
from krrood.adapters.json_serializer import SubclassJSONSerializer, to_json, from_json
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body


@dataclass
class HasXYZ(SubclassJSONSerializer):
    x: float
    y: float
    z: float

    def to_json(self) -> Dict[str, Any]:
        """
        Serialize to JSON.
        """
        return {
            **super().to_json(),
            "x": self.x,
            "y": self.y,
            "z": self.z,
        }

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Self:
        """
        Deserialize from JSON.
        """
        return cls(x=data["x"], y=data["y"], z=data["z"])


@dataclass
class Sage10kRotation(HasXYZ): ...


@dataclass
class Sage10kPosition(HasXYZ): ...


@dataclass
class Sage10kSize(SubclassJSONSerializer):
    height: float
    length: float
    width: float

    def to_json(self) -> Dict[str, Any]:
        """
        Serialize to JSON.
        """
        return {
            **super().to_json(),
            "height": self.height,
            "length": self.length,
            "width": self.width,
        }

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Self:
        """
        Deserialize from JSON.
        """
        return cls(height=data["height"], length=data["length"], width=data["width"])


@dataclass
class Sage10kPhysicallyBasedRendering(SubclassJSONSerializer):
    metallic: float
    roughness: float

    def to_json(self) -> Dict[str, Any]:
        """
        Serialize to JSON.
        """
        return {
            **super().to_json(),
            "metallic": self.metallic,
            "roughness": self.roughness,
        }

    @classmethod
    def _from_json(
        cls, data: Dict[str, Any], **kwargs
    ) -> Sage10kPhysicallyBasedRendering:
        """
        Deserialize from JSON.
        """
        return cls(metallic=data["metallic"], roughness=data["roughness"])


@dataclass
class Sage10kWall(SubclassJSONSerializer):
    id: str
    start_point: Sage10kPosition
    end_point: Sage10kPosition
    material: str
    height: float
    thickness: float

    def to_json(self) -> Dict[str, Any]:
        """
        Serialize to JSON.
        """
        return {
            **super().to_json(),
            "id": self.id,
            "start_point": to_json(self.start_point),
            "end_point": to_json(self.end_point),
            "material": self.material,
            "height": self.height,
            "thickness": self.thickness,
        }

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Sage10kWall:
        """
        Deserialize from JSON.
        """
        return cls(
            id=data["id"],
            start_point=Sage10kPosition._from_json(data["start_point"], **kwargs),
            end_point=Sage10kPosition._from_json(data["end_point"], **kwargs),
            material=data["material"],
            height=data["height"],
            thickness=data["thickness"],
        )


@dataclass
class Sage10kObject(SubclassJSONSerializer):
    id: str
    room_id: str
    type: str
    description: str
    source: str

    source_id: str
    """
    The prefix of the filenames in the objects folder that related to this object.
    """

    place_id: str
    place_guidance: str
    mass: float

    position: Sage10kPosition
    rotation: Sage10kRotation
    dimensions: Sage10kSize
    pbr_parameters: Sage10kPhysicallyBasedRendering

    def create_in_world(self, world: World, directory_path: Path) -> Body:
        ply_file = directory_path / "objects" / f"{self.source_id}.ply"
        texture_file = directory_path / "objects" / f"{self.source_id}_texture.png"

        # 1. Load the geometry from the PLY file
        mesh = trimesh.load(ply_file)

        # 2. Load the texture image
        tex_image = Image.open(texture_file)

        # 3. Create a material with the image
        # SimpleMaterial is the standard for basic diffuse textures
        material = trimesh.visual.texture.SimpleMaterial(image=tex_image)

        # 4. Apply the material to the mesh
        # Note: The mesh MUST have 'uv' coordinates already defined in the PLY
        mesh.visual = trimesh.visual.TextureVisuals(
            uv=mesh.visual.uv, material=material
        )

        # 5. Preview the result
        mesh.show()

        b = Body(name=PrefixedName(prefix=self.room_id, name=self.id))

    def to_json(self) -> Dict[str, Any]:
        """
        Serialize to JSON.
        """
        return {
            **super().to_json(),
            "id": self.id,
            "room_id": self.room_id,
            "type": self.type,
            "description": self.description,
            "source": self.source,
            "source_id": self.source_id,
            "place_id": self.place_id,
            "place_guidance": self.place_guidance,
            "mass": self.mass,
            "position": to_json(self.position),
            "rotation": to_json(self.rotation),
            "dimensions": to_json(self.dimensions),
            "pbr_parameters": to_json(self.pbr_parameters),
        }

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Sage10kObject:
        """
        Deserialize from JSON.
        """
        return cls(
            id=data["id"],
            room_id=data["room_id"],
            type=data["type"],
            description=data["description"],
            source=data["source"],
            source_id=data["source_id"],
            place_id=data["place_id"],
            place_guidance=data["place_guidance"],
            mass=data["mass"],
            position=Sage10kPosition._from_json(data["position"], **kwargs),
            rotation=Sage10kRotation._from_json(data["rotation"], **kwargs),
            dimensions=Sage10kSize._from_json(data["dimensions"], **kwargs),
            pbr_parameters=Sage10kPhysicallyBasedRendering._from_json(
                data["pbr_parameters"], **kwargs
            ),
        )


@dataclass
class Sage10kRoom(SubclassJSONSerializer):
    id: str
    room_type: str
    dimensions: Sage10kSize
    position: Sage10kPosition
    floor_material: str
    objects: List[Sage10kObject] = field(default_factory=list)
    walls: List[Sage10kWall] = field(default_factory=list)

    def to_json(self) -> Dict[str, Any]:
        """
        Serialize to JSON.
        """
        return {
            JSON_TYPE_NAME: get_full_class_name(self.__class__),
            "id": self.id,
            "room_type": self.room_type,
            "dimensions": to_json(self.dimensions),
            "position": to_json(self.position),
            "floor_material": self.floor_material,
            "objects": to_json(self.objects),
            "walls": to_json(self.walls),
        }

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Sage10kRoom:
        """
        Deserialize from JSON.
        """
        return cls(
            id=data["id"],
            room_type=data["room_type"],
            dimensions=Sage10kSize._from_json(data["dimensions"], **kwargs),
            position=Sage10kPosition._from_json(data["position"], **kwargs),
            floor_material=data["floor_material"],
            objects=[Sage10kObject._from_json(d, **kwargs) for d in data["objects"]],
            walls=[Sage10kWall._from_json(w, **kwargs) for w in data["walls"]],
        )


@dataclass
class Sage10kDoor(SubclassJSONSerializer):
    id: str
    wall_id: str
    position_on_wall: float
    width: float
    height: float
    door_type: str
    opens_inward: bool
    opening: bool
    door_material: str

    def to_json(self) -> Dict[str, Any]:
        """
        Serialize to JSON.
        """
        return {
            **super().to_json(),
            "id": self.id,
            "wall_id": self.wall_id,
            "position_on_wall": self.position_on_wall,
            "width": self.width,
            "height": self.height,
            "door_type": self.door_type,
            "opens_inward": self.opens_inward,
            "opening": self.opening,
            "door_material": self.door_material,
        }

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Sage10kDoor:
        """
        Deserialize from JSON.
        """
        return cls(
            id=data["id"],
            wall_id=data["wall_id"],
            position_on_wall=data["position_on_wall"],
            width=data["width"],
            height=data["height"],
            door_type=data["door_type"],
            opens_inward=data["opens_inward"],
            opening=data["opening"],
            door_material=data["door_material"],
        )


@dataclass
class Sage10kScene(SubclassJSONSerializer):
    id: str
    building_style: str
    description: str
    created_from_text: str
    total_area: float
    rooms: List[Sage10kRoom] = field(default_factory=list)
    directory_path: Optional[Path] = None

    def to_json(self) -> Dict[str, Any]:
        """
        Serialize to JSON.
        """
        return {
            **super().to_json(),
            "id": self.id,
            "building_style": self.building_style,
            "description": self.description,
            "created_from_text": self.created_from_text,
            "total_area": self.total_area,
            "rooms": to_json(self.rooms),
        }

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Sage10kScene:
        """
        Deserialize from JSON.
        """
        return cls(
            id=data["id"],
            building_style=data["building_style"],
            description=data["description"],
            created_from_text=data["created_from_text"],
            total_area=data["total_area"],
            rooms=[Sage10kRoom._from_json(r, **kwargs) for r in data["rooms"]],
        )

    def create_world(self) -> World:
        if self.directory_path is None:
            raise ValueError("Directory path is not set.")
        world = World()

        for room in self.rooms:
            for sage_object in room.objects:
                sage_object.create_in_world(world, self.directory_path)
