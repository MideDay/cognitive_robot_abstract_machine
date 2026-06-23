from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from typing_extensions import Iterable, Union

from krrood.entity_query_language.predicate import Field
from krrood.entity_query_language.verbalization import morphology
from krrood.entity_query_language.verbalization.fragments.base import (
    Fragment,
    PhraseFragment,
    RoleFragment,
    WordFragment,
)
from krrood.entity_query_language.verbalization.fragments.features import Number
from krrood.entity_query_language.verbalization.fragments.roles import SemanticRole
from krrood.entity_query_language.verbalization.microplanning.coordination import (
    MAX_SET_MEMBERS,
    one_of,
)
from krrood.entity_query_language.verbalization.vocabulary.english import (
    Copulas,
    SetMembership,
)
from krrood.entity_query_language.verbalization.vocabulary.words import (
    PlainWord,
    VocabEnum,
)


class ClauseElement(ABC):
    """One typed part-of-speech constituent of a predicate clause.

    A predicate's ``_verbalization_fragment_`` builds its clause from these elements rather than raw
    fragments, so the author writes the affirmative, present-tense form once and the realisation
    passes inflect it (verb agreement, copula suppletion) and negate it (do-support). The element
    only declares *what part of speech* a word is; how it is realised is the morphology pass's job.
    """

    @abstractmethod
    def as_fragment(self) -> Fragment:
        """:return: the fragment this element contributes to the clause."""


@dataclass(frozen=True)
class Noun(ClauseElement):
    """A noun constituent — a predicate :class:`~krrood.entity_query_language.predicate.Field`, an
    already-rendered fragment, or a literal noun word."""

    content: Union[str, Fragment, Field]
    """A field (its rendered fragment is used), a rendered fragment (passed through), or a literal
    noun string."""

    def as_fragment(self) -> Fragment:
        """:return: the field's / wrapped fragment, or a word leaf for a literal string.

        >>> Noun("department").as_fragment().text
        'department'
        """
        if isinstance(self.content, Field):
            return self.content.fragment
        if isinstance(self.content, Fragment):
            return self.content
        return WordFragment(text=self.content)


@dataclass(frozen=True)
class Verb(ClauseElement):
    """A lexical verb given as its lemma. The morphology pass realises it present-tense
    (*"work"* → *"works"*) and negates it with do-support (*"does not work"*)."""

    lemma: str
    """The verb's base form (*"work"*, *"contain"*, *"love"*)."""

    number: Number = Number.SINGULAR
    """The subject number the verb agrees with — ``PLURAL`` reads the bare *"work"* / *"have"* for a
    coordinated or plural subject."""

    def as_fragment(self) -> RoleFragment:
        """:return: a ``VERB``-role leaf carrying the lemma for the morphology pass to inflect.

        >>> Verb("work").as_fragment().role
        <SemanticRole.VERB: 'verb'>
        """
        return RoleFragment(text=self.lemma, role=SemanticRole.VERB, number=self.number)


@dataclass(frozen=True)
class Adjective(ClauseElement):
    """A predicative adjective complement after a copula (*"is **reachable**"*)."""

    word: str
    """The adjective's surface word."""

    def as_fragment(self) -> WordFragment:
        """:return: a plain word leaf for the adjective.

        >>> Adjective("reachable").as_fragment().text
        'reachable'
        """
        return WordFragment(text=self.word)


@dataclass(frozen=True)
class Copula(ClauseElement):
    """The copula *"is"* of a predicative clause — realised for number (*"is"* / *"are"*) and
    negation (*"is not"*) by the morphology pass."""

    def as_fragment(self) -> RoleFragment:
        """:return: the affirmative singular copula leaf the morphology pass inflects.

        >>> Copula().as_fragment().text
        'is'
        """
        return Copulas.IS.as_fragment()


@dataclass(frozen=True)
class OneOf(ClauseElement):
    """A bounded membership set — *"one of A, B, or C"* — over a collection of admissible values.

    This is the high-level element for a "the subject is one of these" clause (a tuple of admissible
    types, a small value domain), so an author never re-implements the listing: each member renders
    as a linked type reference when it is a class, else as a literal value, and a set larger than the
    cap is summarised by count (*"one of seven types"*) rather than spelled out — the same bounded
    surface a domain-constrained variable uses.
    """

    members: Union[Iterable, Field]
    """The admissible values — a predicate :class:`~krrood.entity_query_language.predicate.Field`
    bound to a collection, or a collection directly. Classes render as linked type references, other
    values as literals."""

    def as_fragment(self) -> Fragment:
        """:return: the membership phrase, or a count summary past the cap.

        >>> from krrood.entity_query_language.verbalization.fragments.base import (
        ...     flatten_fragment_to_plain_text,
        ... )
        >>> flatten_fragment_to_plain_text(OneOf((int, str)).as_fragment())
        'one of int or str'
        """
        members = list(
            self.members.value if isinstance(self.members, Field) else self.members
        )
        are_types = bool(members) and all(
            isinstance(member, type) for member in members
        )
        render = RoleFragment.for_type if are_types else RoleFragment.for_literal
        listed = one_of([render(member) for member in members[: MAX_SET_MEMBERS + 1]])
        if listed is not None:
            return listed
        return PhraseFragment(
            parts=[
                SetMembership.ONE_OF.as_fragment(),
                WordFragment(text=morphology.cardinal(len(members))),
                WordFragment(text="types" if are_types else "values"),
            ]
        )


class Preposition(VocabEnum):
    """The prepositions a clause links its constituents with (*"works **in** a department"*)."""

    IN = PlainWord("in")
    ON = PlainWord("on")
    OF = PlainWord("of")
    TO = PlainWord("to")
    BY = PlainWord("by")
    AT = PlainWord("at")
    WITH = PlainWord("with")
    FROM = PlainWord("from")


ClauseConstituent = Union[Fragment, Field, ClauseElement, Preposition]


def _as_fragment(constituent: ClauseConstituent) -> Fragment:
    """:return: the fragment a clause constituent contributes — a field's rendered fragment, a raw
    fragment as-is, or an element's :meth:`~ClauseElement.as_fragment`."""
    if isinstance(constituent, Field):
        return constituent.fragment
    if isinstance(constituent, Fragment):
        return constituent
    return constituent.as_fragment()


def clause(*constituents: ClauseConstituent) -> PhraseFragment:
    """
    Build a predicate clause from typed part-of-speech constituents.

    A predicate states its affirmative form once — *"<subject> works in <object>"* —
    ``clause(Noun(subject), Verb("work"), Preposition.IN, Noun(object))`` — and the realisation
    passes handle agreement and negation. A raw :class:`Fragment` is accepted too, so a rendered
    field fragment can be dropped in directly.

    :param constituents: The clause's elements in surface order.
    :return: The inline phrase fragment for the clause.

    >>> from krrood.entity_query_language.verbalization.fragments.base import (
    ...     flatten_fragment_to_plain_text, WordFragment,
    ... )
    >>> flatten_fragment_to_plain_text(
    ...     clause(Noun(WordFragment(text="an Employee")), Verb("work"), Preposition.IN,
    ...            Noun(WordFragment(text="a Department")))
    ... )
    'an Employee work in a Department'
    """
    return PhraseFragment(parts=[_as_fragment(constituent) for constituent in constituents])
