from __future__ import annotations

from enum import Enum


class Number(Enum):
    """Grammatical number of a noun or verb (singular vs. plural)."""

    SINGULAR = "singular"
    """A single entity."""
    PLURAL = "plural"
    """More than one entity."""

    @classmethod
    def of(cls, is_plural: bool) -> Number:
        """
        :param is_plural: Whether the number is plural.
        :return: ``PLURAL`` when *is_plural* else ``SINGULAR``.
        """
        return cls.PLURAL if is_plural else cls.SINGULAR


class Definiteness(Enum):
    """Grammatical definiteness of a noun phrase (*"a/an"* vs. *"the"* vs. no determiner)."""

    BARE = "bare"
    """No determiner — a numbered label (*"Robot 2"*) or a bare predicate noun."""
    INDEFINITE = "indefinite"
    """First, non-specific mention — *"a/an Robot"*, or a bare plural *"Robots"*."""
    DEFINITE = "definite"
    """Identifiable or subsequent mention — *"the Robot"* / *"the Robots"*."""
    UNIQUE = "unique"
    """A uniqueness-quantified first mention — *"the unique Robot"*."""


class Glue(Enum):
    """Orthographic spacing of a token relative to its neighbours."""

    NONE = "none"
    """Spaced on both sides like a normal word (the default)."""
    LEFT = "left"
    """No space *before* this token — it hugs the preceding token (*","* → *"x,"*)."""
    RIGHT = "right"
    """No space *after* this token — it hugs the following token (*"("* → *"(x"*)."""
