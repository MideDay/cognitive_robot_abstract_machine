from __future__ import annotations

import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing_extensions import Dict, List, Set

from krrood.entity_query_language.core.base_expressions import SymbolicExpression
from krrood.entity_query_language.core.variable import Variable, Literal
from krrood.entity_query_language.query.query import Entity, Query
from krrood.entity_query_language.verbalization.fragments.features import Definiteness
from krrood.entity_query_language.verbalization.subquery import (
    aggregation_source_root,
    selected_aggregator,
)


def _aggregation_source_ids(expression: SymbolicExpression) -> Set[uuid.UUID]:
    """
    Such a variable denotes a population to aggregate over, not a specific entity, so it must not
    consume an entity-disambiguation number — otherwise the outer subject would pick up a
    spurious *"1"* with no matching *"2"*, and a constrained aggregation scope would read *"among
    BankTransaction 2"* rather than *"among BankTransactions"*.

    :param expression: Root expression to scan.
    :return: The ``_id_`` of every variable that serves as the source population of an
        aggregation sub-query (e.g. the ``BankTransaction`` behind ``max(t.amount_details.amount)``),
        to exclude from numbering.
    """
    ids: Set[uuid.UUID] = set()
    for node in expression._all_expressions_:
        if isinstance(node, Entity) and selected_aggregator(node) is not None:
            root = aggregation_source_root(node)
            if root is not None:
                ids.add(root._id_)
    return ids


def _build_disambiguation_map(expression: SymbolicExpression) -> Dict[uuid.UUID, str]:
    """
    Types appearing once keep the plain type name; types appearing two or more times get
    "TypeName 1", "TypeName 2", … labels in encounter order. Literal nodes are excluded, as are
    variables that only serve as the source population of an aggregation sub-query.

    :param expression: Root expression to pre-scan.
    :return: A mapping of ``variable._id_`` → display label.
    """
    if isinstance(expression, Query):
        expression.build()

    suppressed = _aggregation_source_ids(expression)

    type_to_ids: Dict[str, List[uuid.UUID]] = defaultdict(list)
    seen_ids: Set[uuid.UUID] = set()

    for node in expression._all_expressions_:
        if isinstance(node, Variable) and not isinstance(node, Literal):
            if node._id_ in suppressed:
                continue
            type_name = (
                node._type_.__name__
                if getattr(node, "_type_", None)
                else node.__class__.__name__
            )
            if node._id_ not in seen_ids:
                seen_ids.add(node._id_)
                type_to_ids[type_name].append(node._id_)

    result: Dict[uuid.UUID, str] = {}
    for type_name, ids in type_to_ids.items():
        if len(ids) == 1:
            result[ids[0]] = type_name
        else:
            for ordinal, variable_id in enumerate(ids, 1):
                result[variable_id] = f"{type_name} {ordinal}"
    return result


@dataclass
class ReferringExpressions:
    """
    Pre-computed referring-expression state for a verbalization pass: the disambiguation labels
    (numbering colliding types) and the set of referents already introduced.

    This is the referring-expression generation subtask of microplanning: deciding between an
    indefinite first mention (*"a Robot"*), a definite subsequent mention (*"the Robot"*), a
    numbered form when one type occurs several times (*"Robot 1"* / *"Robot 2"*), and a pronoun
    (*"its …"*) when a chain is rooted at the current discourse subject.

    References:

    * Reiter, E. & Dale, R. (2000), "Building Natural Language Generation Systems", CUP —
      referring-expression generation as a microplanning subtask.
    * Dale, R. & Reiter, E. (1995), "Computational interpretations of the Gricean maxims in the
      generation of referring expressions", *Cognitive Science* 19(2).
    """

    seen: Set[uuid.UUID] = field(default_factory=set)
    """Referent ``_id_`` s introduced so far, used to seed coreference across builds sharing this
    context, so verbalizing the same expression twice reads *"a Robot"* then *"the Robot"*."""

    disambiguation_map: Dict[uuid.UUID, str] = field(default_factory=dict)
    """Maps variable ``_id_`` → display label, pre-computed before verbalization
    begins.  Single-type variables keep the plain type name; colliding types get
    ``"TypeName 1"``, ``"TypeName 2"`` labels."""

    @classmethod
    def from_expression(cls, expression: SymbolicExpression) -> ReferringExpressions:
        """
        :param expression: Root EQL expression or query to scan.
        :return: An instance with the disambiguation map pre-built for *expression*.
        """
        return cls(disambiguation_map=_build_disambiguation_map(expression))

    def mark_introduced(self, expression: SymbolicExpression) -> None:
        """Record *expression* as introduced (so a later build sharing this context seeds it)."""
        self.seen.add(expression._id_)

    def numbered_label(self, variable: Variable) -> tuple[str, bool]:
        """Records *variable* as introduced.

        The label is the pre-computed disambiguation label (*"Robot 2"* for a colliding type),
        else the plain type name; *is_numbered* is whether they differ.

        :param variable: A variable instance.
        :return: Tuple of ``(display_label, is_numbered)``.
        """
        type_name = (
            variable._type_.__name__
            if getattr(variable, "_type_", None)
            else variable.__class__.__name__
        )
        label = self.disambiguation_map.get(variable._id_, type_name)
        self.seen.add(variable._id_)
        return label, label != type_name

    def noun_for_parts(self, variable: Variable) -> tuple[Definiteness, str]:
        """
        :param variable: A variable instance.
        :return: The first-mention ``(Definiteness, label)`` for *variable* — a numbered variable
            (*"Robot 2"*) is ``BARE``, any other is ``INDEFINITE`` (*"a Robot"*).
        """
        label, is_numbered = self.numbered_label(variable)
        return (Definiteness.BARE if is_numbered else Definiteness.INDEFINITE), label
