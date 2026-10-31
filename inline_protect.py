#!/usr/bin/env python3
"""Inline protection/restoration — Phase 2 stub.

Placeholder format (decided in Phase 0b, see HANDOFF.md): `§X{n}§`
(U+00A7 section sign + 'X' + integer counter). It survives MarianMT verbatim
in all tested positions, unlike `__INLINE_CODE_0__`, bare digits, or
letter-based prefixes.

Phase 2 will implement extraction in the order links -> inline code -> HTML,
then restore the placeholders after translation (with the `◆X{n}◆` fallback
and collision handling documented in HANDOFF.md).

This module exists now so Phase 2 plugs in without changing main.py's shape:
`protect_line` returns `(protected_line, placeholders)` and `restore_line`
consumes the same mapping.
"""

from __future__ import annotations

import re

# Literal §X<digits>§ marker — reused by Phase 2 for count/set validation.
PLACEHOLDER_RE = re.compile(r"§X(\d+)§")


def protect_line(line: str) -> tuple[str, dict[str, str]]:
    """Stub. Phase 2: replace inline spans with `§X{n}§` placeholders.

    Returns the line unchanged and an empty placeholder->original mapping, so
    the Phase 1 pipeline is a transparent no-op.
    """
    return line, {}


def restore_line(line: str, placeholders: dict[str, str]) -> str:
    """Stub. Phase 2: substitute placeholders back into the translated line."""
    return line
