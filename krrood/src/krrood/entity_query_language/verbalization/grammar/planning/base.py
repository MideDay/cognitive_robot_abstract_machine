from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from typing_extensions import Generic, TypeVar

N = TypeVar("N")
"""The EQL node type the planner analyses."""

P = TypeVar("P")
"""The plan (data record) the planner produces."""


@dataclass
class Planner(ABC, Generic[N, P]):
    """
    Pure analysis of a single EQL *node* into a plan of type ``P``.

    A planner inspects an EQL node and produces a *plan*: a plain, frozen data record of the
    decisions about *what to say*. It does not build fragments, touch the verbalization context,
    or recurse — the content/structure stage of the Reiter & Dale microplanning model.

    Reference: Reiter, E. & Dale, R. (2000), "Building Natural Language Generation Systems", CUP —
    microplanning vs. surface realisation.
    """

    node: N
    """The EQL expression being analysed."""

    @abstractmethod
    def plan(self) -> P:
        """:return: The plan computed from the node."""
