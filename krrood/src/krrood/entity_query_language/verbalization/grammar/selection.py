from __future__ import annotations

from abc import ABC

from typing_extensions import (
    Any,
    Callable,
    ClassVar,
    List,
    Optional,
    Sequence,
    Type,
    TypeVar,
)

_T = TypeVar("_T")


def most_specific(candidates: Sequence[_T], key: Callable[[_T], Any]) -> Optional[_T]:
    """
    :param candidates: Items already filtered to those that apply.
    :param key: Specificity key; the maximum wins.
    :return: The single most-specific candidate by *key*, or ``None`` when empty.
    """
    return max(candidates, key=key, default=None)


class SpecificityRule(ABC):
    """
    A guarded alternative selected by specificity: the shared base of the small rule
    registries (restriction folding, restriction-subject resolution).

    An alternative is a **subclass** that implements an ``applies(...)`` guard (its
    signature is the subfamily's concern) and a payload method; alternatives
    *self-register* as the family's direct subclasses and are ranked by ``priority``.

    Reference: production-rule selection; the systemic-functional "most delicate system wins"
    principle.
    """

    priority: ClassVar[int] = 0
    """Tiebreak when several alternatives apply (higher wins)."""

    @classmethod
    def alternatives(cls) -> List[Type[SpecificityRule]]:
        """:return: The alternative subclasses of this family (closed: direct subclasses)."""
        return list(cls.__subclasses__())

    @classmethod
    def most_applicable(cls, *args: Any) -> Optional[Type[SpecificityRule]]:
        """
        *args* are forwarded verbatim to each alternative's ``applies`` classmethod, so
        the subfamily fixes that signature (e.g. ``(item, subject)``).

        :return: The most-specific alternative whose ``applies(*args)`` holds, or ``None``.
        """
        applicable = [alt for alt in cls.alternatives() if alt.applies(*args)]
        return most_specific(applicable, key=lambda alt: alt.priority)
