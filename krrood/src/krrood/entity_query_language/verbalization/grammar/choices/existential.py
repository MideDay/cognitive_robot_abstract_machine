"""Existential-number system: *"there's a X"* vs *"there are Xs"*."""

from __future__ import annotations

from dataclasses import dataclass

from krrood.entity_query_language.verbalization.fragments.base import VerbFragment
from krrood.entity_query_language.verbalization.grammar.choices.base import Choice
from krrood.entity_query_language.verbalization.grammar.choices.features import Number
from krrood.entity_query_language.verbalization.vocabulary.english import (
    ExistentialPhrase,
)


@dataclass
class ExistentialForm(Choice):
    """System environment for the existential clause that introduces an antecedent/consequent."""

    number: Number
    """— entry feature — singular vs plural existential."""

    type_name: str
    """— realisation input — the introduced type's display name."""


@dataclass
class SingularExistential(ExistentialForm):
    """*"there's a X"*."""

    def applies(self) -> bool:
        return self.number is Number.SINGULAR

    def realize(self) -> VerbFragment:
        return ExistentialPhrase.THERE_IS_A.build_phrase(self.type_name)


@dataclass
class PluralExistential(ExistentialForm):
    """*"there are Xs"*."""

    def applies(self) -> bool:
        return self.number is Number.PLURAL

    def realize(self) -> VerbFragment:
        return ExistentialPhrase.THERE_ARE.build_phrase(self.type_name)
