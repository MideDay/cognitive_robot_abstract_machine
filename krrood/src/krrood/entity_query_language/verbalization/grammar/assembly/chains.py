from __future__ import annotations

from typing_extensions import Optional

from krrood.entity_query_language.core.base_expressions import SymbolicExpression
from krrood.entity_query_language.core.mapped_variable import (
    Attribute,
    Index,
    MappedVariable,
)
from krrood.entity_query_language.core.variable import Variable
from krrood.entity_query_language.query.query import Entity
from krrood.entity_query_language.verbalization import morphology
from krrood.entity_query_language.verbalization.chain_utils import (
    build_path_parts,
    ChainAnalysis,
)
from krrood.entity_query_language.verbalization.fragments.base import (
    NounPhrase,
    PhraseFragment,
    PossessiveChain,
    RoleFragment,
    Fragment,
    WordFragment,
)
from krrood.entity_query_language.verbalization.fragments.features import (
    Definiteness,
    Number,
)
from krrood.entity_query_language.verbalization.grammar.assembly.base import Assembler
from krrood.entity_query_language.verbalization.grammar.assembly.query import (
    QueryAssembler,
)
from krrood.entity_query_language.verbalization.microplanning.possessive import (
    possessive_path,
)
from krrood.entity_query_language.verbalization.subquery import (
    unwrap_result_quantifiers,
)
from krrood.entity_query_language.verbalization.vocabulary.english import (
    Articles,
    Copulas,
    Prepositions,
)


class ChainAssembler(Assembler[MappedVariable, None]):
    """Realise a ``MappedVariable`` chain (possessive / predicative / pronominal forms).

    Realisation-only (``planner = None``): a chain has no content to decide, only a surface form
    to render — a boolean terminal attribute → predicative *"<navigation> is [not] <attribute>"*;
    anything else → the possessive path *"the attribute of the Root"*, optionally pronominalised
    to *"its …"* when the root is the current coreference subject.

    Reference: Gatt & Reiter (2009), SimpleNLG — surface realisation.
    """

    def realize(self, node: MappedVariable, plan: None = None) -> Fragment:
        """
        :param node: The chain to render.
        :param plan: Unused (this assembler has no plan).
        :return: The default chain rendering.
        """
        return self.chain(node)

    # ── entry forms ──────────────────────────────────────────────────────────

    def chain(self, expression: MappedVariable, *, negated: bool = False) -> Fragment:
        """
        When a plural is requested and this is a single attribute on a variable, build the bare
        plural *"attrs of Roots"*; otherwise render singular.

        :param expression: The chain to render.
        :param negated: Whether to negate a boolean-terminal predicative.
        :return: A boolean terminal → predicative *"<navigation> is [not] <attribute>"*; else the
            possessive path *"the attribute of the Root"*.
        """
        analysis = ChainAnalysis.of(expression)
        if self.ctx.number is Number.PLURAL:
            plural = self._plural_attribute(analysis)
            if plural is not None:
                return plural
        if analysis.is_boolean_terminal:
            return self._boolean_predicative(analysis, negated)
        root_fragment = self._chain_root(analysis.root)
        if isinstance(analysis.root, Variable):
            # Defer the pronominal-vs-possessive choice to the coreference pass: it knows
            # whether the root is the current discourse subject.
            return PossessiveChain(
                parts=analysis.parts,
                root_fragment=root_fragment,
                root_referent_id=analysis.root._id_,
            )
        return possessive_path(analysis.parts, root_fragment)

    def _plural_attribute(self, analysis: ChainAnalysis) -> Optional[Fragment]:
        """
        :param analysis: The analysed chain.
        :return: *"attributes of Roots"* when the chain is a single attribute on a variable, else
            ``None``.
        """
        root = analysis.root
        if not (
            isinstance(root, Variable)
            and len(analysis.chain) == 1
            and isinstance(analysis.chain[0], Attribute)
        ):
            return None
        label, numbered = self.ctx.refer.numbered_label(root)
        attribute = analysis.chain[0]
        root_noun_phrase = NounPhrase(
            head=RoleFragment.for_variable(label, root),
            number=Number.SINGULAR if numbered else Number.PLURAL,
            definiteness=Definiteness.BARE if numbered else Definiteness.INDEFINITE,
            referent_id=root._id_,
        )
        return NounPhrase(
            head=RoleFragment.for_attribute(
                attribute._owner_class_, attribute._attribute_name_
            ),
            number=Number.PLURAL,
            definiteness=Definiteness.INDEFINITE,
            modifiers=[Prepositions.OF.as_fragment(), root_noun_phrase],
        )

    def _chain_root(self, root: SymbolicExpression) -> Fragment:
        """
        :param root: The chain root.
        :return: The noun phrase for the chain root (an inline noun for entity roots).
        """
        inner = unwrap_result_quantifiers(root)
        if isinstance(inner, Entity):
            return QueryAssembler(self.ctx).inline_noun(inner)
        return self.ctx.child(root)

    def _boolean_predicative(self, analysis: ChainAnalysis, negated: bool) -> Fragment:
        """
        Chains ending in an integer index get ordinal navigation.

        :param analysis: The analysed chain.
        :param negated: Whether to negate the predicative.
        :return: The predicative *"<navigation> is [not] <attribute>"*.
        """
        chain = analysis.chain
        root_fragment = self._chain_root(analysis.root)
        navigation_chain = chain[:-1]

        if not navigation_chain:
            navigation_fragment = root_fragment
        elif isinstance(navigation_chain[-1], Index) and isinstance(
            navigation_chain[-1]._key_, int
        ):
            ordinal = morphology.ordinal(navigation_chain[-1]._key_)
            prefix_fragment = possessive_path(
                build_path_parts(navigation_chain[:-1]), root_fragment
            )
            navigation_fragment = PhraseFragment(
                parts=[
                    Articles.THE.as_fragment(),
                    WordFragment(text=ordinal),
                    Prepositions.OF.as_fragment(),
                    prefix_fragment,
                ]
            )
        else:
            navigation_fragment = possessive_path(
                build_path_parts(navigation_chain), root_fragment
            )

        copula = Copulas.IS_NOT.as_fragment() if negated else Copulas.IS.as_fragment()
        terminal = chain[-1]
        return PhraseFragment(
            parts=[
                navigation_fragment,
                copula,
                RoleFragment.for_attribute(
                    terminal._owner_class_, terminal._attribute_name_
                ),
            ]
        )
