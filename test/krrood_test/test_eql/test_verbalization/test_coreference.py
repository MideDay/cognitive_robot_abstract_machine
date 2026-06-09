"""
Unit tests for the de-stringified coreference store (:class:`ReferringExpressions`):
labels are now **fragments**, registration is encapsulated, and ``seen_reference``
composes *"the <label>"* from the stored fragment.
"""

from __future__ import annotations

from krrood.entity_query_language.verbalization.fragments.base import (
    flatten_fragment_to_plain_text,
    RoleFragment,
)
from krrood.entity_query_language.verbalization.fragments.roles import SemanticRole
from krrood.entity_query_language.verbalization.microplanning.referring import (
    ReferringExpressions,
)


class _Ref:
    """Minimal stand-in carrying an ``_id_`` (what the coreference store keys on)."""

    def __init__(self, identifier: int) -> None:
        self._id_ = identifier


def _text(fragment) -> str:
    return flatten_fragment_to_plain_text(fragment)


def test_register_label_stores_a_fragment_and_composes_definite_reference():
    refer = ReferringExpressions()
    robot = _Ref(1)
    refer.register_label(robot, "Robot")
    assert isinstance(refer.label_of(robot), RoleFragment)
    assert _text(refer.seen_reference(robot)) == "the Robot"


def test_register_keeps_a_structured_label_fragment():
    refer = ReferringExpressions()
    aggregate = _Ref(2)
    label = RoleFragment(text="maximum amount", role=SemanticRole.AGGREGATION)
    refer.register(aggregate, label)
    # The stored label is the fragment itself (not a flattened string).
    assert refer.label_of(aggregate) is label
    assert _text(refer.seen_reference(aggregate)) == "the maximum amount"


def test_alias_shares_the_source_label():
    refer = ReferringExpressions()
    source, target = _Ref(3), _Ref(4)
    refer.register_label(source, "Cabinet")
    refer.alias(target, source)
    assert _text(refer.seen_reference(target)) == "the Cabinet"


def test_seen_reference_is_none_when_unseen():
    refer = ReferringExpressions()
    assert refer.seen_reference(_Ref(5)) is None
    refer.alias(_Ref(6), _Ref(7))  # aliasing an unseen source is a no-op
    assert refer.seen_reference(_Ref(6)) is None
