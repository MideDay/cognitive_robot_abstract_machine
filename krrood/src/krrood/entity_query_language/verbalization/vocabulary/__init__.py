"""
EQL verbalization vocabulary.

Quick imports::

    from krrood.entity_query_language.verbalization.vocabulary import (
        Keywords, Logicals, Aggregations, Copulas,
        Prepositions, Conjunctions, Articles, SortDirections,
        ExistentialPhrase, FallbackNouns, GroupKeyPhrases, Operators,
    )
"""

from krrood.entity_query_language.verbalization.vocabulary.english import (
    Aggregations,
    Articles,
    Conjunctions,
    Copulas,
    ExistentialPhrase,
    FallbackNouns,
    GroupKeyPhrases,
    Keywords,
    Logicals,
    Operators,
    Prepositions,
    SortDirections,
)
from krrood.entity_query_language.verbalization.vocabulary.words import (
    AggregationWord,
    KeywordWord,
    LogicalWord,
    OperatorPhrase,
    OperatorWord,
    PlainWord,
    RoleWord,
    VocabEnum,
)

__all__ = [
    # Namespace enums
    "Aggregations",
    "Articles",
    "Conjunctions",
    "Copulas",
    "ExistentialPhrase",
    "FallbackNouns",
    "GroupKeyPhrases",
    "Keywords",
    "Logicals",
    "Operators",
    "Prepositions",
    "SortDirections",
    # Base word types
    "AggregationWord",
    "KeywordWord",
    "LogicalWord",
    "OperatorPhrase",
    "OperatorWord",
    "PlainWord",
    "RoleWord",
    "VocabEnum",
]
