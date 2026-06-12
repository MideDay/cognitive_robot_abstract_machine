from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing_extensions import TYPE_CHECKING, Callable, ClassVar, Optional, Sequence

from krrood.entity_query_language.core.base_expressions import SymbolicExpression
from krrood.entity_query_language.verbalization.fragments.base import Fragment
from krrood.entity_query_language.verbalization.fragments.features import Number
from krrood.entity_query_language.verbalization.grammar.selection import most_specific

if TYPE_CHECKING:
    from krrood.entity_query_language.verbalization.context import VerbalizationContext
    from krrood.entity_query_language.verbalization.microplanning.binding_scope import (
        BindingScope,
    )
    from krrood.entity_query_language.verbalization.microplanning.config import (
        RenderConfig,
    )
    from krrood.entity_query_language.verbalization.microplanning.referring import (
        ReferringExpressions,
    )


@dataclass
class Ctx:
    """
    Per-node context handed to a rule's ``build``.

    Bundles the recursion entry (``child``, the fold continuation) with the microplanning
    services, so a rule never recurses by hand nor reaches for cross-cutting state directly.
    """

    child: Callable[..., Fragment]
    """Recurse on a sub-expression — the fold continuation.  Accepts an optional ``number=`` to
    request a plural realisation of the child."""

    context: VerbalizationContext
    """The owning verbalization context."""

    number: Number = Number.SINGULAR
    """Grammatical number requested for this node.  A number-aware rule reads it to build a
    plural noun-phrase shape; every other rule ignores it (renders singular)."""

    @property
    def refer(self) -> ReferringExpressions:
        """:return: The referring-expression service (articles, coreference, pronouns)."""
        return self.context.referring

    @property
    def scope(self) -> BindingScope:
        """:return: The binding-scope service (deferred constraints + field overrides)."""
        return self.context.binding

    @property
    def config(self) -> RenderConfig:
        """:return: The render-mode flags (query depth, compact predicates)."""
        return self.context.config


class PhraseRule(ABC):
    """
    One Montague rule-to-rule clause: *for this construct, build this phrase.*

    This realises the rule-to-rule mapping of Montague grammar: each construct of the source
    algebra (an EQL expression) has one rule describing how it composes into the target
    (English) algebra. Rules are flat (all direct subclasses); specificity comes from
    ``construct``, never from the rule-class hierarchy.

    References:

    * Montague, R. (1970), "Universal Grammar", *Theoria* 36 — syntax algebra → semantics
      algebra as a homomorphism.
    * Bach, E. (1976) — the rule-to-rule hypothesis (one syntactic rule ↔ one semantic rule).
    * Stanford Encyclopedia of Philosophy, "Montague Semantics" / "Compositionality".
    """

    construct: ClassVar[type]
    """The EQL node class this rule handles (the ``isinstance`` gate)."""
    name: ClassVar[str] = ""
    """Stable identifier for querying or tracing the grammar."""
    tiebreak: ClassVar[int] = 0
    """Explicit ordering when two rules with the same ``construct`` are both guarded and
    overlap; higher wins."""
    enters_query_scope: ClassVar[bool] = False
    """``True`` on a rule whose construct is itself a query body, so an entity found anywhere
    within it renders as a nested noun phrase."""

    def when(self, node: SymbolicExpression, ctx: Ctx) -> bool:
        """
        Extra precondition beyond ``isinstance(node, construct)``.

        The default accepts everything; override to express the non-``isinstance`` part of the
        rule's applicability (a guarded rule outranks an unguarded one on the same construct).

        :param node: The candidate EQL expression.
        :param ctx: The per-node context (recursion and services).
        :return: ``True`` when the rule applies to *node*.
        """
        return True

    @abstractmethod
    def build(self, node: SymbolicExpression, ctx: Ctx) -> Fragment:
        """
        Build the fragment for *node*.

        :param node: The EQL expression to verbalize.
        :param ctx: The per-node context (recursion and services).
        :return: The fragment for *node*.
        """


def _mro_depth(cls: type) -> int:
    """
    :param cls: A construct (EQL node) class.
    :return: Specificity of *cls* — deeper in the MRO ⇒ more specific (``Literal`` > ``Variable``).
    """
    return len(cls.__mro__)


def _is_guarded(rule: PhraseRule) -> bool:
    """
    :param rule: A phrase rule instance.
    :return: ``True`` when *rule* overrides ``when`` (a guarded rule).
    """
    return type(rule).when is not PhraseRule.when


def select(
    node: SymbolicExpression, rules: Sequence[PhraseRule], ctx: Ctx
) -> Optional[PhraseRule]:
    """
    Specificity key, highest wins: ``(construct MRO depth, guarded over unguarded, explicit
    tiebreak)``.

    :param node: The EQL expression being dispatched.
    :param rules: The grammar.
    :param ctx: The per-node context, passed to each rule's ``when``.
    :return: The most-specific rule whose ``construct`` and ``when`` match *node*, or ``None``
        when none apply (the caller supplies the fallback).
    """
    candidates = [
        rule
        for rule in rules
        if isinstance(node, rule.construct) and rule.when(node, ctx)
    ]
    return most_specific(
        candidates,
        key=lambda rule: (
            _mro_depth(rule.construct),
            _is_guarded(rule),
            rule.tiebreak,
        ),
    )
