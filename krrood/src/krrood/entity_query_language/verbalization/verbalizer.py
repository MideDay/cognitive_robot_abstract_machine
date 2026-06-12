from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import Optional

from krrood.entity_query_language.core.base_expressions import SymbolicExpression
from krrood.entity_query_language.verbalization.context import VerbalizationContext
from krrood.entity_query_language.verbalization.engine import fold
from krrood.entity_query_language.verbalization.fragments.base import Fragment
from krrood.entity_query_language.verbalization.grammar.english import RULES
from krrood.entity_query_language.verbalization.rendering.realization import (
    realize_tree,
)


@dataclass
class EQLVerbalizer:
    """
    Builds the natural-language fragment tree that represents an EQL expression.
    """

    def build(
        self,
        expression: SymbolicExpression,
        context: Optional[VerbalizationContext] = None,
    ) -> Fragment:
        """
        Translate *expression* into its natural-language fragment tree.

        A fresh context is created when *context* is ``None``; pass a shared context across calls
        so repeated mentions corefer (a Robot … the Robot).

        :param expression: Any EQL symbolic expression.
        :param context: Shared verbalization state; created automatically when omitted.
        :return: Root of the fragment tree representing *expression* in natural language.
        """
        if context is None:
            context = VerbalizationContext.from_expression(expression)
        # Referents already introduced by prior builds on this (shared) context, so the same
        # expression verbalized twice reads "a Robot" then "the Robot".  Snapshot BEFORE the
        # fold, which records this build's own mentions in the same set.
        already_seen = set(context.referring.seen)
        fragment = fold(expression, context, RULES)
        return realize_tree(fragment, already_seen=already_seen)
