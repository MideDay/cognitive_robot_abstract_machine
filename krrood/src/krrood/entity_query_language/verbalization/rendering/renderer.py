from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from krrood.entity_query_language.verbalization.fragments.base import (
    BlockFragment,
    PhraseFragment,
    RoleFragment,
    VerbFragment,
    WordFragment,
)
from krrood.entity_query_language.verbalization.rendering.formatter import (
    BulletStyle,
    Formatter,
    IndentSize,
    PlainFormatter,
)


@dataclass
class FragmentRenderer(ABC):
    """Converts a VerbFragment tree into a string."""

    _formatter: Formatter = field(default_factory=PlainFormatter)
    """
    The formatter to use for rendering.
    """

    @abstractmethod
    def render(self, fragment: VerbFragment) -> str:
        """
        Render a VerbFragment tree into a string.

        :param fragment: The root of the fragment tree.
        :return: The rendered string.
        """
        ...


@dataclass
class ParagraphRenderer(FragmentRenderer):
    """
    Flattens the fragment tree into a single prose string.

    BlockFragment headers and items are joined inline; nesting adds no
    visual structure — only content.
    """

    def render(self, fragment: VerbFragment) -> str:
        match fragment:
            case WordFragment(text=text):
                return text
            case RoleFragment(text=text, role=role):
                return self._formatter.colorize(text, role)
            case PhraseFragment(parts=parts, separator=sep):
                rendered = [self.render(p) for p in parts]
                return sep.join(rendered)
            case BlockFragment(header=header, items=items):
                rendered_items = [self.render(i) for i in items]
                prose = ", ".join(rendered_items)
                if header is None:
                    return prose
                header_str = self.render(header)
                return f"{header_str}{self._formatter.space}{prose}" if prose else header_str
            case _:
                return ""


@dataclass
class HierarchicalRenderer(FragmentRenderer):
    """
    Renders BlockFragments as indented bullet lists.

    Each level of BlockFragment nesting adds one ``indent`` step.
    Non-block fragments are rendered inline using the same formatter.

    Example output (ANSI/plain)::

        If:
          - there's a Handle
          - there's a PrismaticConnection, whose child is …
        Then:
          - there's a Drawer
            - whose container is …
    """

    indent_size: IndentSize = field(default=IndentSize.TWO_SPACES)
    """
    The size of the indentation for each level of nesting.
    """
    bullet: BulletStyle = field(default=BulletStyle.DASH)
    """
    The bullet character to use for the list items.
    """

    def render(self, fragment: VerbFragment, depth: int = 0) -> str:
        match fragment:
            case BlockFragment(header=header, items=items):
                lines: list[str] = []
                if header is not None:
                    lines.append(self.formatted_indent * depth + self._inline(header))
                    depth = depth + 1
                for item in items:
                    lines.append(self._render_item(item, depth))
                return self._formatter.newline.join(lines)
            case _:
                return self.formatted_indent * depth + self._inline(fragment)

    @property
    def formatted_indent(self) -> str:
        """The indentation string, with spaces replaced by the formatter's space character."""
        return self.indent_size.value.replace(' ', self._formatter.space)

    def _render_item(self, fragment: VerbFragment, depth: int) -> str:
        """Render one item, prepending the bullet at its indentation level."""
        match fragment:
            case BlockFragment():
                return self.render(fragment, depth)
            case _:
                prefix = self.formatted_indent * depth + self.bullet.value + self._formatter.space
                return prefix + self._inline(fragment)

    def _inline(self, fragment: VerbFragment) -> str:
        """Render a non-block fragment as a flat inline string."""
        match fragment:
            case WordFragment(text=text):
                return text
            case RoleFragment(text=text, role=role):
                return self._formatter.colorize(text, role)
            case PhraseFragment(parts=parts, separator=sep):
                return sep.join(self._inline(p) for p in parts)
            case BlockFragment():
                return self.render(fragment, 0)
            case _:
                return ""
