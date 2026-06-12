from __future__ import annotations

from typing_extensions import List, Optional, Tuple

from krrood.entity_query_language.verbalization.fragments.base import (
    PhraseFragment,
    RoleFragment,
    Fragment,
)
from krrood.entity_query_language.verbalization.fragments.roles import SemanticRole
from krrood.entity_query_language.verbalization.fragments.source_ref import SourceRef
from krrood.entity_query_language.verbalization.vocabulary.english import (
    Articles,
    Prepositions,
)

#: One hop of a navigation chain — ``(attribute_name, source_ref)``.
PathPart = Tuple[str, Optional[SourceRef]]


def _attribute_fragment(name: str, source_ref: Optional[SourceRef]) -> RoleFragment:
    """:return: A role-tagged attribute fragment for *name*."""
    return RoleFragment(text=name, role=SemanticRole.ATTRIBUTE, source_ref=source_ref)


def possessive_path(parts: List[PathPart], root_fragment: Fragment) -> Fragment:
    """:return: *"the <inner> of the <outer> of <root>"* (parts iterated innermost-first)."""
    if not parts:
        return root_fragment
    reversed_parts = list(reversed(parts))
    first_name, first_ref = reversed_parts[0]
    fragment_parts: List[Fragment] = [
        Articles.THE.as_fragment(),
        _attribute_fragment(first_name, first_ref),
    ]
    for attribute_name, attribute_reference in reversed_parts[1:]:
        fragment_parts.extend(
            [
                Prepositions.OF_THE.as_fragment(),
                _attribute_fragment(attribute_name, attribute_reference),
            ]
        )
    fragment_parts.extend([Prepositions.OF.as_fragment(), root_fragment])
    return PhraseFragment(parts=fragment_parts)


def pronominal_path(parts: List[PathPart], pronoun: Fragment) -> Fragment:
    """:return: *"its attr"* (single hop) or *"the attr of its foo"* (multi-hop)."""
    if not parts:
        return pronoun
    reversed_parts = list(reversed(parts))
    last = len(reversed_parts) - 1
    fragment_parts: List[Fragment] = []
    for index, (attribute_name, attribute_reference) in enumerate(reversed_parts):
        attribute_fragment = _attribute_fragment(attribute_name, attribute_reference)
        if index == 0 and index != last:
            fragment_parts.extend([Articles.THE.as_fragment(), attribute_fragment])
        elif index == 0:  # single attribute → "its booking_date"
            fragment_parts.extend([pronoun, attribute_fragment])
        elif index == last:  # adjacent to the elided root → "of its amount_details"
            fragment_parts.extend(
                [Prepositions.OF.as_fragment(), pronoun, attribute_fragment]
            )
        else:
            fragment_parts.extend(
                [Prepositions.OF_THE.as_fragment(), attribute_fragment]
            )
    return PhraseFragment(parts=fragment_parts)
