from __future__ import annotations

from enum import Enum

from typing_extensions import Optional

from krrood.entity_query_language.core.base_expressions import SymbolicExpression
from krrood.entity_query_language.core.mapped_variable import MappedVariable
from krrood.entity_query_language.core.variable import Variable, Literal
from krrood.entity_query_language.operators.aggregators import Aggregator
from krrood.entity_query_language.operators.comparator import Comparator
from krrood.entity_query_language.operators.core_logical_operators import (
    LogicalOperator,
)
from krrood.entity_query_language.query.query import Entity, SetOf


class SemanticRole(Enum):
    """Semantic category of a fragment, determining its colour markup."""

    KEYWORD = "keyword"
    """EQL structure words — *If*, *Then*, *Find*, *Where*, *Such that*."""
    VARIABLE = "variable"
    """Type and instance names — *Robot*, *Employee 1*."""
    AGGREGATION = "aggregation"
    """Aggregation phrases — *sum of*, *number of*, *average of*."""
    OPERATOR = "operator"
    """Comparator phrases — *is greater than*, *equals*."""
    LOGICAL = "logical"
    """Logical connectives — *and*, *or*, *not*, *for all*, *there exists*."""
    LITERAL = "literal"
    """Literal values — ``42``, ``"hello"``, ``True``."""
    ATTRIBUTE = "attribute"
    """Attribute and field names — *battery*, *tasks*, *name*."""
    PLAIN = "plain"
    """Neutral connecting text with no special colour."""


#: Hex colour string (or ``None`` for no colour) for each semantic role, matching the
#: query-graph palette.
ROLE_COLORS: dict[SemanticRole, Optional[str]] = {
    SemanticRole.KEYWORD: "#eded18",  # ConclusionSelector yellow
    SemanticRole.VARIABLE: "cornflowerblue",
    SemanticRole.AGGREGATION: "#F54927",  # Aggregator red-orange
    SemanticRole.OPERATOR: "#ff7f0e",  # Comparator orange
    SemanticRole.LOGICAL: "#2ca02c",  # LogicalOperator green
    SemanticRole.LITERAL: "#949292",  # Literal gray
    SemanticRole.ATTRIBUTE: "#8FC7B8",  # MappedVariable teal
    SemanticRole.PLAIN: None,
}


def _build_role_map() -> dict[type, SemanticRole]:
    """:return: The mapping of EQL expression types to their semantic role."""
    return {
        LogicalOperator: SemanticRole.LOGICAL,
        Aggregator: SemanticRole.AGGREGATION,
        Comparator: SemanticRole.OPERATOR,
        MappedVariable: SemanticRole.ATTRIBUTE,
        Literal: SemanticRole.LITERAL,  # before Variable in MRO traversal
        Variable: SemanticRole.VARIABLE,
        Entity: SemanticRole.VARIABLE,
        SetOf: SemanticRole.VARIABLE,
    }


_role_map: dict[type, SemanticRole] = _build_role_map()


def role_for(expression: SymbolicExpression) -> SemanticRole:
    """
    Falls back to ``PLAIN`` when the expression's type matches no known role.

    :param expression: Any EQL expression instance.
    :return: The most-specific matching semantic role for an EQL expression instance.
    """
    for cls in type(expression).__mro__:
        if cls in _role_map:
            return _role_map[cls]
    return SemanticRole.PLAIN
