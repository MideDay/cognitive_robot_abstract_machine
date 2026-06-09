"""
MorphologyProcessor — the single realisation pass that **applies** grammatical number.

Assemblers *tag* a leaf with :attr:`~krrood.entity_query_language.verbalization.fragments.features.Number`
(a decision); this one pass walks the finished fragment tree (via
:func:`~krrood.entity_query_language.verbalization.fragments.base.map_fragment`) and
pluralises the text of every leaf tagged :attr:`Number.PLURAL` — so inflection is applied
in exactly one place instead of inline at every assembler.

Reference: Gatt & Reiter (2009), SimpleNLG — the MorphologyProcessor realisation stage.
"""

from __future__ import annotations

from dataclasses import replace

from krrood.entity_query_language.verbalization import morphology
from krrood.entity_query_language.verbalization.fragments.base import (
    flatten_fragment_to_plain_text,
    map_fragment,
    RoleFragment,
    VerbFragment,
)
from krrood.entity_query_language.verbalization.fragments.features import Number


class MorphologyProcessor:
    """Pluralise every ``RoleFragment`` tagged :attr:`Number.PLURAL`; leave the rest as-is."""

    def process(self, fragment: VerbFragment) -> VerbFragment:
        """Return a new tree with all plural-tagged leaves inflected (idempotent)."""
        return map_fragment(fragment, self._inflect)

    @staticmethod
    def _inflect(leaf: VerbFragment) -> VerbFragment:
        if isinstance(leaf, RoleFragment) and leaf.number is Number.PLURAL:
            return replace(
                leaf,
                text=morphology.ensure_plural(leaf.text),
                number=Number.SINGULAR,
            )
        return leaf


def realize_subtree(fragment: VerbFragment) -> str:
    """
    Fully realise a sub-tree to plain text — run the morphology pass, then flatten.

    For an **opaque leaf** (a user :class:`~krrood.entity_query_language.predicate.Verbalizable`
    template that string-formats its children): the template's content is opaque text, so it
    must realise its children *here*, locally, rather than deferring to the global pass.
    """
    return flatten_fragment_to_plain_text(MorphologyProcessor().process(fragment))
