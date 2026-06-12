from __future__ import annotations

from typing_extensions import TYPE_CHECKING, Optional, Sequence

from krrood.entity_query_language.core.base_expressions import SymbolicExpression
from krrood.entity_query_language.verbalization.fragments.base import Fragment
from krrood.entity_query_language.verbalization.fragments.features import Number
from krrood.entity_query_language.verbalization.grammar.phrase_rule import (
    Ctx,
    PhraseRule,
    select,
)
from krrood.entity_query_language.verbalization.grammar.english import RULES

if TYPE_CHECKING:
    from krrood.entity_query_language.verbalization.context import VerbalizationContext


class UnverbalizableExpressionError(TypeError):
    """
    No grammar rule covers an EQL construct — a coverage gap, surfaced as an error rather
    than silently degrading the node to its class name.
    """


def fold(
    node: SymbolicExpression,
    context: VerbalizationContext,
    rules: Optional[Sequence[PhraseRule]] = None,
    number: Number = Number.SINGULAR,
) -> Fragment:
    """
    Verbalize *node* by dispatching it to its matching grammar rule and recursing — the single
    catamorphism (fold) over the EQL expression tree.

    A node carrying a pre-built binding override is returned directly, before any dispatch.
    When no rule covers the node, an ``UnverbalizableExpressionError`` is raised rather than
    degrading silently to the class name.

    The recursion is an F-algebra fold over the EQL algebra, with the grammar as the algebra
    (Meijer, Fokkinga & Paterson 1991, "Functional Programming with Bananas, Lenses, Envelopes
    and Barbed Wire"; Bird & de Moor 1997, "Algebra of Programming").

    :param node: Any EQL expression.
    :param context: The verbalization context (services and render config).
    :param rules: Grammar to dispatch over; defaults to the standard rule set.
    :param number: Grammatical number to build *node* under.
    :return: The fragment for *node*.
    :raises UnverbalizableExpressionError: when no grammar rule covers *node*.
    """
    rules = RULES if rules is None else rules

    node_id = getattr(node, "_id_", None)
    if node_id is not None:
        override = context.binding.binding_overrides.get(node_id)
        if override is not None:
            return override

    ctx = Ctx(
        child=lambda child_node, number=Number.SINGULAR: fold(
            child_node, context, rules, number=number
        ),
        context=context,
        number=number,
    )

    rule = select(node, rules, ctx)
    if rule is None:
        raise UnverbalizableExpressionError(
            f"No verbalization rule for {type(node).__name__!r} "
            f"(name={getattr(node, '_name_', None)!r}); add a PhraseRule in grammar/english.py."
        )
    if rule.enters_query_scope:
        # The rule's construct is a query body: everything built inside sees query_depth >= 1,
        # so a nested Entity renders as a noun phrase. Declared on the rule (not pushed by hand
        # in assemblers) so the policy lives in one place. ``when`` already ran outside.
        with context.config.query_depth_scope():
            return rule.build(node, ctx)
    return rule.build(node, ctx)
