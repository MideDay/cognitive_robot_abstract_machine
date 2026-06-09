"""
Linguistic **features** — the small typed inputs that a
:class:`~krrood.entity_query_language.verbalization.grammar.choices.base.Choice` reads to
select an alternative (the *entry conditions* of an SFL system).
"""

from __future__ import annotations

from enum import Enum, auto


class Number(Enum):
    """Grammatical number — the entry feature of the number-agreement systems."""

    SINGULAR = auto()
    PLURAL = auto()

    @classmethod
    def of(cls, is_plural: bool) -> "Number":
        """``PLURAL`` when *is_plural* else ``SINGULAR`` (bridges boolean plan features)."""
        return cls.PLURAL if is_plural else cls.SINGULAR
