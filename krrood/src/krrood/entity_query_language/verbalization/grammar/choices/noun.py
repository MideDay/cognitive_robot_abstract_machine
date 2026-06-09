"""Noun-number system: a role-tagged noun in singular vs pluralised surface form."""

from __future__ import annotations

from dataclasses import dataclass

from krrood.entity_query_language.verbalization import morphology
from krrood.entity_query_language.verbalization.fragments.base import VerbFragment
from krrood.entity_query_language.verbalization.fragments.factory import role
from krrood.entity_query_language.verbalization.fragments.roles import SemanticRole
from krrood.entity_query_language.verbalization.grammar.choices.base import Choice
from krrood.entity_query_language.verbalization.grammar.choices.features import Number


@dataclass
class NounForm(Choice):
    """System environment for a noun rendered in singular or plural form, with a role."""

    number: Number
    """— entry feature — singular vs plural surface form."""

    name: str
    """— realisation input — the singular noun (e.g. attribute / field name)."""

    semantic_role: SemanticRole
    """— realisation input — the role tag for colour/linking (e.g. ATTRIBUTE)."""


@dataclass
class SingularNoun(NounForm):
    """The noun in its given (singular) form."""

    def applies(self) -> bool:
        return self.number is Number.SINGULAR

    def realize(self) -> VerbFragment:
        return role(self.name, self.semantic_role)


@dataclass
class PluralNoun(NounForm):
    """The noun pluralised (without double-pluralising)."""

    def applies(self) -> bool:
        return self.number is Number.PLURAL

    def realize(self) -> VerbFragment:
        return role(morphology.ensure_plural(self.name), self.semantic_role)
