"""
The English grammar — one :class:`~krrood.entity_query_language.verbalization.grammar.phrase_rule.PhraseRule`
per EQL construct, as data.

Each rule is a Montague rule-to-rule clause: *for this construct, build this phrase*
(Montague 1970; Bach 1976, the rule-to-rule hypothesis).  A rule only **combines** —
recursion is delegated to ``ctx.child`` (the fold), and cross-cutting decisions to the
microplanning services (``ctx.refer`` / ``ctx.scope`` / ``ctx.config``), morphology, the
coordination module, and the lexicon — so each rule is responsible for a single
construct's surface composition.

Families are ported here one at a time; until a construct's rule is present the engine
falls back to the legacy dispatcher (strangler migration).
"""

from __future__ import annotations

from typing_extensions import List

from krrood.entity_query_language.core.variable import (
    ExternallySetVariable,
    Literal,
    Variable,
)
from krrood.entity_query_language.operators.comparator import Comparator
from krrood.entity_query_language.verbalization.fragments.base import RoleFragment
from krrood.entity_query_language.verbalization.fragments.factory import phrase, role
from krrood.entity_query_language.verbalization.fragments.roles import SemanticRole
from krrood.entity_query_language.verbalization.grammar.phrase_rule import PhraseRule
from krrood.entity_query_language.verbalization.microplanning.referring import (
    ArticleSelection,
)
from krrood.entity_query_language.verbalization.operator_phrase import (
    comparator_operator,
)
from krrood.entity_query_language.verbalization.vocabulary.english import Articles

# ── comparator ───────────────────────────────────────────────────────────────

COMPARATOR_RULES: List[PhraseRule] = [
    PhraseRule(
        construct=Comparator,
        name="comparator",
        # "<left> <operator> <right>", e.g. "is greater than 50".
        build=lambda node, ctx: phrase(
            ctx.child(node.left),
            comparator_operator(node, ctx.context),
            ctx.child(node.right),
        ),
    ),
]


# ── variables ──────────────────────────────────────────────────────────────────


def _variable_build(node, ctx):
    """ "a/an Robot" (first mention), "the Robot" (subsequent), or "Robot N" (numbered)."""
    article, label = ctx.refer.noun_for_parts(node)
    label_fragment = RoleFragment.for_variable(label, node)
    if article == ArticleSelection.NONE:
        return label_fragment
    if article == ArticleSelection.DEFINITE:
        return phrase(Articles.THE.as_fragment(), label_fragment)
    return phrase(Articles.indefinite(label), label_fragment)


def _external_variable_build(node, ctx):
    """ "a/an TypeName" for an opaque externally-set variable (no coreference)."""
    type_name = node._type_.__name__ if getattr(node, "_type_", None) else "variable"
    return phrase(
        Articles.indefinite(type_name), role(type_name, SemanticRole.VARIABLE)
    )


VARIABLE_RULES: List[PhraseRule] = [
    PhraseRule(construct=Variable, name="variable", build=_variable_build),
    # Literal <: Variable — deeper construct wins via select.
    PhraseRule(
        construct=Literal,
        name="literal",
        build=lambda node, ctx: role(
            ctx.context.type_name_of_value(node._value_), SemanticRole.LITERAL
        ),
    ),
    PhraseRule(
        construct=ExternallySetVariable,
        name="external-variable",
        build=_external_variable_build,
    ),
]

# ── the full grammar ───────────────────────────────────────────────────────────

RULES: List[PhraseRule] = [
    *COMPARATOR_RULES,
    *VARIABLE_RULES,
]
