"""
Choice — a declarative micro form-decision: the **system / chooser / realization** of
Systemic Functional Linguistics, as data.

A ``Choice`` is a decision among mutually-exclusive **alternatives** (its subclasses),
each of which *self-describes* its entry condition (:meth:`applies`) and its surface
:meth:`realize`-ation.  There is **no feature→form mapping table**: the alternatives are
the enumeration, and they select themselves.  Selection reuses the same
:func:`~krrood.entity_query_language.verbalization.grammar.phrase_rule.most_specific`
primitive that dispatches :class:`~krrood.entity_query_language.verbalization.grammar.phrase_rule.PhraseRule`
— a `Choice` is that production-rule primitive applied at the *form* level rather than
the *construct* level (the two share only this function, not a base class).

Convention (so ``resolve`` can construct every alternative uniformly):
the **shared ancestor carries the whole environment** — entry features (read by
``applies``) and realisation inputs (read by ``realize``) — as fields; the alternative
**subclasses are behaviour-only** (``applies``/``realize``), adding no new fields.

References: Halliday, *Introduction to Functional Grammar*; Mann & Matthiessen, the
Penman/Nigel system network + choosers + realization statements; Bateman, KPML;
Elhadad, FUF/SURGE (functional unification); Gatt & Reiter, SimpleNLG features.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing_extensions import ClassVar, List, Type

from krrood.entity_query_language.verbalization.fragments.base import VerbFragment
from krrood.entity_query_language.verbalization.grammar.phrase_rule import most_specific


@dataclass
class Choice(ABC):
    """
    One linguistic system: pick the applicable self-describing alternative and realise it.

    Subclass once to declare the system's **environment** (entry features + realisation
    inputs as fields); subclass again per **alternative** (implement :meth:`applies` and
    :meth:`realize`, no new fields).
    """

    priority: ClassVar[int] = 0
    """Tiebreak when several alternatives apply (higher wins); mirrors ``PhraseRule.tiebreak``."""

    @abstractmethod
    def applies(self) -> bool:
        """Return ``True`` when this alternative's entry condition holds (reads ``self`` fields)."""

    @abstractmethod
    def realize(self) -> VerbFragment:
        """Build this alternative's surface fragment (reads ``self`` fields)."""

    @classmethod
    def alternatives(cls) -> "List[Type[Choice]]":
        """The alternative subclasses of this system (closed: direct subclasses)."""
        return list(cls.__subclasses__())

    @classmethod
    def resolve(cls, **environment) -> VerbFragment:
        """
        Construct every alternative with *environment*, keep those whose :meth:`applies`
        holds, and realise the most-specific one (by :attr:`priority`).

        :param environment: The system's fields (entry features + realisation inputs).
        :return: The chosen alternative's fragment.
        :raises ValueError: when no alternative applies.
        """
        applicable = [
            alternative
            for alternative in (sub(**environment) for sub in cls.alternatives())
            if alternative.applies()
        ]
        chosen = most_specific(applicable, key=lambda alternative: alternative.priority)
        if chosen is None:
            raise ValueError(
                f"No applicable alternative for {cls.__name__}({environment})"
            )
        return chosen.realize()
