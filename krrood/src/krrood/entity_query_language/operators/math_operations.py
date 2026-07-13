"""
Arithmetic operators for the Entity Query Language.

An arithmetic node (see :mod:`krrood.entity_query_language.operators.arithmetic`) delegates its
computation to the :class:`MathOperator` it carries: each operator owns both its rendered symbol and the
Python callable that performs it, so the node stays decoupled from the concrete operation.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass
from enum import Enum

from typing_extensions import Any, Callable


@dataclass(frozen=True)
class MathOperatorSpec:
    """
    The symbol and callable that make up one :class:`MathOperator`.
    """

    symbol: str
    """The mathematical symbol used when rendering the operator."""
    function: Callable[..., Any]
    """The callable that performs the operation over already-resolved operand values."""


class MathOperator(Enum):
    """
    An arithmetic operator usable inside a query. Each member carries the symbol it renders as and the
    callable that computes it over already-resolved operand values.
    """

    ADD = MathOperatorSpec("+", operator.add)
    SUBTRACT = MathOperatorSpec("-", operator.sub)
    MULTIPLY = MathOperatorSpec("*", operator.mul)
    DIVIDE = MathOperatorSpec("/", operator.truediv)
    FLOOR_DIVIDE = MathOperatorSpec("//", operator.floordiv)
    MODULO = MathOperatorSpec("%", operator.mod)
    POWER = MathOperatorSpec("**", operator.pow)
    NEGATE = MathOperatorSpec("-", operator.neg)

    @property
    def symbol(self) -> str:
        """
        :return: The mathematical symbol used when rendering this operator.
        """
        return self.value.symbol

    @property
    def function(self) -> Callable[..., Any]:
        """
        :return: The callable that performs this operation over already-resolved operand values.
        """
        return self.value.function
