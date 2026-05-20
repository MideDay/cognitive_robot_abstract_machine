"""
English vocabulary for EQL verbalization.

Every English word or phrase used in the verbalization output is defined here
as a named constant on one of the namespace Enums.  No natural-language strings
exist outside this module.

Namespace Enums (Keywords, Logicals, …) have frozen-dataclass instances as
member values; the VocabEnum mixin exposes .as_fragment() and .text directly on
the member so callers never write .value explicitly.

Parameterised phrases (ExistentialPhrase, GroupKeyPhrases, FallbackNouns) add
delegation methods on the Enum that forward to build_phrase() / plural_fragment()
on the underlying dataclass value.

Operators uses OperatorPhrase values and exposes .select() + .from_callable().
"""

from __future__ import annotations

import operator as _operator
from dataclasses import dataclass
from enum import Enum

import inflect

from krrood.entity_query_language.verbalization.fragments.base import (
    PhraseFragment,
    RoleFragment,
    VerbFragment,
    WordFragment,
)
from krrood.entity_query_language.verbalization.fragments.roles import SemanticRole
from krrood.entity_query_language.verbalization.utils import _ensure_plural
from krrood.entity_query_language.verbalization.vocabulary.words import (
    AggregationWord,
    KeywordWord,
    LogicalWord,
    OperatorPhrase,
    OperatorWord,
    PlainWord,
    VocabEnum,
)

_engine = inflect.engine()


# ── English-specific word subtypes ─────────────────────────────────────────────
# These add behaviour for phrases that depend on a runtime type_name argument.


@dataclass(frozen=True)
class SingularExistential(PlainWord):
    """'there's a/an TypeName' — article is phonologically computed at call time."""

    def build_phrase(self, type_name: str) -> VerbFragment:
        article = _engine.a(type_name).split()[0]
        return PhraseFragment(
            parts=[
                WordFragment(text=f"{self.text} {article}"),
                RoleFragment(text=type_name, role=SemanticRole.VARIABLE),
            ],
            separator=" ",
        )


@dataclass(frozen=True)
class PluralExistential(PlainWord):
    """'there are TypeNames'."""

    def build_phrase(self, type_name: str) -> VerbFragment:
        return PhraseFragment(
            parts=[
                self.as_fragment(),
                RoleFragment(text=_ensure_plural(type_name), role=SemanticRole.VARIABLE),
            ],
            separator=" ",
        )


@dataclass(frozen=True)
class FallbackNounWord(PlainWord):
    """A noun used when no type information is available (singular + inflected plural)."""

    def plural_fragment(self) -> WordFragment:
        return WordFragment(text=_engine.plural(self.text))


@dataclass(frozen=True)
class CommonGroupKeyWord(PlainWord):
    """'the common <field> of the <plural_root>' — group-key binding phrase."""

    def build_phrase(self, field_name: str, plural_root: str) -> VerbFragment:
        return PhraseFragment(
            parts=[
                self.as_fragment(),
                WordFragment(text=field_name),
                Prepositions.OF_THE.as_fragment(),
                WordFragment(text=plural_root),
            ],
            separator=" ",
        )


# ── Namespace Enums ────────────────────────────────────────────────────────────


class Keywords(VocabEnum):
    IF           = KeywordWord("If")
    THEN         = KeywordWord("then")
    FIND         = KeywordWord("Find")
    FIND_SETS_OF = KeywordWord("Find sets of")
    SUCH_THAT    = KeywordWord("such that")
    WHERE        = KeywordWord("where")
    WHOSE        = KeywordWord("whose")
    GROUPED_BY   = KeywordWord("grouped by")
    GROUPED      = KeywordWord("grouped")
    HAVING       = KeywordWord("having")
    ORDERED_BY   = KeywordWord("ordered by")
    TRUE         = KeywordWord("true")


class Logicals(VocabEnum):
    NOT          = LogicalWord("not")
    EITHER       = LogicalWord("either")
    FOR_ALL      = LogicalWord("for all")
    THERE_EXISTS = LogicalWord("there exists")


class Aggregations(VocabEnum):
    COUNT      = AggregationWord("number of")
    COUNT_ALL  = AggregationWord("count of all")
    SUM        = AggregationWord("sum of")
    AVERAGE    = AggregationWord("average of")
    MAX        = AggregationWord("maximum")
    MIN        = AggregationWord("minimum")
    MODE       = AggregationWord("mode of")
    MULTI_MODE = AggregationWord("all modes of")


class Copulas(VocabEnum):
    # role = OPERATOR — copulas appear alongside comparison operators visually
    IS     = OperatorWord("is")
    IS_NOT = OperatorWord("is not")
    ARE    = OperatorWord("are")


class Prepositions(VocabEnum):
    OF     = PlainWord("of")
    OF_THE = PlainWord("of the")


class Conjunctions(VocabEnum):
    AND = PlainWord("and")
    OR  = PlainWord("or")


class SortDirections(VocabEnum):
    ASCENDING  = PlainWord("ascending")
    DESCENDING = PlainWord("descending")


class Articles(VocabEnum):
    THE        = PlainWord("the")
    THE_UNIQUE = PlainWord("the unique")

    @staticmethod
    def indefinite(following_word: str) -> WordFragment:
        """Return 'a' or 'an' based on the phonological context of following_word."""
        text = _engine.a(following_word).split()[0] if following_word else "a"
        return WordFragment(text=text)


class ExistentialPhrase(VocabEnum):
    THERE_IS_A = SingularExistential("there's")
    THERE_ARE  = PluralExistential("there are")

    def build_phrase(self, type_name: str) -> VerbFragment:
        return self.value.build_phrase(type_name)


class FallbackNouns(VocabEnum):
    ENTITY = FallbackNounWord("entity")

    def plural_fragment(self) -> WordFragment:
        return self.value.plural_fragment()


class GroupKeyPhrases(VocabEnum):
    COMMON_OF = CommonGroupKeyWord("the common")

    def build_phrase(self, field_name: str, plural_root: str) -> VerbFragment:
        return self.value.build_phrase(field_name, plural_root)


class Operators(Enum):
    """Comparison operator phrases.

    Each member's value is an OperatorPhrase holding all eight text variants.
    Use .select(negated, compact, temporal) to obtain the appropriate OperatorWord.
    Use .from_callable(fn) to convert a Python operator callable to the right member.
    """

    EQ = OperatorPhrase(
        standard="is",           compact="equals",
        negated="is not",        negated_compact="does not equal",
        temporal="is at",        temporal_compact="at",
        temporal_negated="is not at", temporal_negated_compact="not at",
    )
    NE = OperatorPhrase(
        standard="is not",       compact="does not equal",
        negated="is",            negated_compact="equals",
        temporal="is not at",    temporal_compact="not at",
        temporal_negated="is at", temporal_negated_compact="at",
    )
    LT = OperatorPhrase(
        standard="is less than",     compact="less than",
        negated="is not less than",  negated_compact="not less than",
        temporal="is before",        temporal_compact="before",
        temporal_negated="is no earlier than", temporal_negated_compact="no earlier than",
    )
    LE = OperatorPhrase(
        standard="is at most",   compact="at most",
        negated="is not at most", negated_compact="not at most",
        temporal="is no later than",  temporal_compact="no later than",
        temporal_negated="is after",  temporal_negated_compact="after",
    )
    GT = OperatorPhrase(
        standard="is greater than",     compact="greater than",
        negated="is not greater than",  negated_compact="not greater than",
        temporal="is after",            temporal_compact="after",
        temporal_negated="is no later than", temporal_negated_compact="no later than",
    )
    GE = OperatorPhrase(
        standard="is at least",   compact="at least",
        negated="is not at least", negated_compact="not at least",
        temporal="is no earlier than",  temporal_compact="no earlier than",
        temporal_negated="is before",   temporal_negated_compact="before",
    )
    CONTAINS = OperatorPhrase(
        standard="contains",          compact="contains",
        negated="does not contain",   negated_compact="does not contain",
    )
    NOT_CONTAINS = OperatorPhrase(
        standard="does not contain",  compact="does not contain",
        negated="contains",           negated_compact="contains",
    )

    def select(
        self, *, negated: bool = False, compact: bool = False, temporal: bool = False
    ) -> OperatorWord:
        return self.value.select(negated=negated, compact=compact, temporal=temporal)

    @classmethod
    def from_callable(cls, fn) -> "Operators":
        from krrood.entity_query_language.operators.comparator import not_contains as _nc
        _MAP = {
            _operator.eq:       cls.EQ,
            _operator.ne:       cls.NE,
            _operator.lt:       cls.LT,
            _operator.le:       cls.LE,
            _operator.gt:       cls.GT,
            _operator.ge:       cls.GE,
            _operator.contains: cls.CONTAINS,
            _nc:                cls.NOT_CONTAINS,
        }
        return _MAP[fn]
