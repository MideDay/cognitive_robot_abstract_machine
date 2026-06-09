"""Copula-number agreement system: *"is"* vs *"are"*."""

from __future__ import annotations

from dataclasses import dataclass

from krrood.entity_query_language.verbalization.fragments.base import VerbFragment
from krrood.entity_query_language.verbalization.grammar.choices.base import Choice
from krrood.entity_query_language.verbalization.grammar.choices.features import Number
from krrood.entity_query_language.verbalization.vocabulary.english import Copulas


@dataclass
class CopulaForm(Choice):
    """System environment for copula number agreement."""

    number: Number
    """— entry feature — singular vs plural copula."""


@dataclass
class SingularCopula(CopulaForm):
    """*"is"*."""

    def applies(self) -> bool:
        return self.number is Number.SINGULAR

    def realize(self) -> VerbFragment:
        return Copulas.IS.as_fragment()


@dataclass
class PluralCopula(CopulaForm):
    """*"are"*."""

    def applies(self) -> bool:
        return self.number is Number.PLURAL

    def realize(self) -> VerbFragment:
        return Copulas.ARE.as_fragment()
