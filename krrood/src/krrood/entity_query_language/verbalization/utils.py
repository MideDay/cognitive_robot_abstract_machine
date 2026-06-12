from __future__ import annotations

import re


def camel_case_to_words(name: str) -> str:
    """
    Convert a CamelCase class name to space-separated lowercase words.

    :param name: CamelCase identifier string.
    :return: Space-separated lowercase words.

    Examples::

        camel_case_to_words("HasRole")     # → "has role"
        camel_case_to_words("IsReachable") # → "is reachable"
    """
    return re.sub(r"([A-Z])", r" \1", name).strip().lower()
