"""
Standalone unit tests for the Choice systems (the SFL feature→form decisions).

Each Choice is resolved directly from features — no verbalizer pipeline — pinning the
self-describing-alternative + most_specific selection.
"""

from __future__ import annotations

import pytest

from krrood.entity_query_language.verbalization.fragments.base import (
    flatten_fragment_to_plain_text,
)
from krrood.entity_query_language.verbalization.fragments.roles import SemanticRole
from krrood.entity_query_language.verbalization.grammar.choices.base import Choice
from krrood.entity_query_language.verbalization.grammar.choices.copula import CopulaForm
from krrood.entity_query_language.verbalization.grammar.choices.existential import (
    ExistentialForm,
)
from krrood.entity_query_language.verbalization.grammar.choices.features import Number
from krrood.entity_query_language.verbalization.grammar.choices.noun import NounForm


def _text(fragment) -> str:
    return flatten_fragment_to_plain_text(fragment)


def test_existential_number_selects_by_number():
    assert (
        _text(ExistentialForm.resolve(number=Number.SINGULAR, type_name="Robot"))
        == "there's a Robot"
    )
    assert (
        _text(ExistentialForm.resolve(number=Number.PLURAL, type_name="Robot"))
        == "there are Robots"
    )


def test_copula_number_selects_by_number():
    assert _text(CopulaForm.resolve(number=Number.SINGULAR)) == "is"
    assert _text(CopulaForm.resolve(number=Number.PLURAL)) == "are"


def test_noun_number_singular_keeps_form_plural_pluralises():
    singular = NounForm.resolve(
        number=Number.SINGULAR, name="container", semantic_role=SemanticRole.ATTRIBUTE
    )
    plural = NounForm.resolve(
        number=Number.PLURAL, name="drawer", semantic_role=SemanticRole.ATTRIBUTE
    )
    assert _text(singular) == "container"
    assert _text(plural) == "drawers"


def test_number_of_bridges_boolean_plan_features():
    assert Number.of(True) is Number.PLURAL
    assert Number.of(False) is Number.SINGULAR


def test_resolve_raises_when_no_alternative_applies():
    # A system with an alternative that never applies → no match.
    from dataclasses import dataclass

    @dataclass
    class NeverForm(Choice):
        pass

    @dataclass
    class NeverAlternative(NeverForm):
        def applies(self) -> bool:
            return False

        def realize(self):  # pragma: no cover - never reached
            raise AssertionError

    with pytest.raises(ValueError):
        NeverForm.resolve()
