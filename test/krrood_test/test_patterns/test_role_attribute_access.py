from dataclasses import dataclass

from krrood.entity_query_language.predicate import Symbol
from krrood.patterns.role import Role, role_taker_field


@dataclass(eq=False)
class PersistentEntityWithName(Symbol):
    name: str

    def __hash__(self) -> int:
        return hash(self.name)


@dataclass(eq=False)
class RoleWithOwnField(Role[PersistentEntityWithName]):
    entity: PersistentEntityWithName = role_taker_field()
    own_field: str = ""


def test_reading_a_taker_attribute_through_a_role_delegates_to_the_taker():
    entity = PersistentEntityWithName(name="original")
    role = RoleWithOwnField(entity=entity)

    assert role.name == "original"


def test_assigning_a_taker_attribute_through_a_role_does_not_modify_the_taker():
    entity = PersistentEntityWithName(name="original")
    role = RoleWithOwnField(entity=entity)

    role.name = "role-local"

    assert entity.name == "original"
    assert role.name == "role-local"


def test_modifying_the_taker_requires_going_through_role_taker():
    entity = PersistentEntityWithName(name="original")
    role = RoleWithOwnField(entity=entity)

    role.role_taker.name = "changed"

    assert entity.name == "changed"
    assert role.name == "changed"


def test_assigning_a_role_native_field_sets_it_on_the_role():
    entity = PersistentEntityWithName(name="original")
    role = RoleWithOwnField(entity=entity)

    role.own_field = "value"

    assert role.own_field == "value"
    assert not hasattr(entity, "own_field")
