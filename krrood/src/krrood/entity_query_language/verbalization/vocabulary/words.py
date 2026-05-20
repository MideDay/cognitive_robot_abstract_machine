"""
Base word-type dataclasses and VocabEnum mixin for EQL verbalization.

PlainWord / RoleWord are the two leaf types.  Fixed-role subclasses of RoleWord
pin _role_ as a ClassVar so callers never re-declare the semantic role.
VocabEnum is a thin Enum mixin that delegates as_fragment() and .text to the
dataclass stored in each member's value.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import ClassVar

from krrood.entity_query_language.verbalization.fragments.base import (
    RoleFragment,
    VerbFragment,
    WordFragment,
)
from krrood.entity_query_language.verbalization.fragments.roles import SemanticRole


@dataclass(frozen=True)
class PlainWord:
    """A neutral word/phrase with no semantic role — renders to WordFragment."""

    text: str

    def as_fragment(self) -> WordFragment:
        return WordFragment(text=self.text)


@dataclass(frozen=True)
class RoleWord:
    """A word/phrase carrying a fixed semantic role — renders to RoleFragment.

    Subclasses set _role_ as a ClassVar.  ClassVar fields are invisible to
    @dataclass so _role_ never appears in __init__; only text does.
    frozen=True is inherited by subclasses via the __setattr__ guard.
    """

    text: str
    _role_: ClassVar[SemanticRole]

    def as_fragment(self) -> RoleFragment:
        return RoleFragment(text=self.text, role=self._role_)


class KeywordWord(RoleWord):
    _role_ = SemanticRole.KEYWORD


class LogicalWord(RoleWord):
    _role_ = SemanticRole.LOGICAL


class AggregationWord(RoleWord):
    _role_ = SemanticRole.AGGREGATION


class OperatorWord(RoleWord):
    _role_ = SemanticRole.OPERATOR


@dataclass(frozen=True)
class OperatorPhrase:
    """All eight phrase variants for one comparison operator, co-located.

    Flags negated / compact / temporal select among the eight fields.
    select() returns the appropriate OperatorWord for the given flag combination.
    """

    standard: str
    compact: str
    negated: str
    negated_compact: str
    temporal: str = ""
    temporal_compact: str = ""
    temporal_negated: str = ""
    temporal_negated_compact: str = ""

    def select(
        self, *, negated: bool = False, compact: bool = False, temporal: bool = False
    ) -> OperatorWord:
        if temporal:
            if negated and compact:
                text = self.temporal_negated_compact
            elif negated:
                text = self.temporal_negated
            elif compact:
                text = self.temporal_compact
            else:
                text = self.temporal
        elif negated and compact:
            text = self.negated_compact
        elif negated:
            text = self.negated
        elif compact:
            text = self.compact
        else:
            text = self.standard
        return OperatorWord(text=text or self.standard)


class VocabEnum(Enum):
    """Mixin for Enums whose values are PlainWord / RoleWord instances.

    Delegates as_fragment() and .text to the dataclass stored in .value so
    callers never need to write .value explicitly for the common cases.
    """

    def as_fragment(self) -> VerbFragment:
        return self.value.as_fragment()

    @property
    def text(self) -> str:
        return self.value.text
