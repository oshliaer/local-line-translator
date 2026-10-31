#!/usr/bin/env python3
"""Inline protection/restoration (Phase 2).

Placeholder format (decided in Phase 0b, see HANDOFF.md): `§X{n}§`
(U+00A7 section sign + 'X' + integer counter). It survives MarianMT verbatim
in every tested position, unlike `__INLINE_CODE_0__`, bare digits, or
letter-based prefixes.

What this module does
---------------------
* `protect_line(line) -> (protected_line, placeholders)` extracts inline
  spans in the required order — links -> inline code -> HTML — and replaces
  each protected span with a placeholder.
* `restore_line(line, placeholders)` puts the original spans back after
  translation and squashes the spurious `] (` that the model inserts between
  a Markdown link's text and URL (0b fact).
* `choose_delimiter(source_text)` picks a placeholder delimiter that does not
  collide with literal `§X…§`-shaped text already present in the source.
* `markers_intact(...)` verifies that every placeholder returned verbatim.

Documented limitations (deliberately not covered, see HANDOFF.md):
* Inline code with double backticks (`` a `b` c ``) — only single-backtick
  spans are matched.
* URLs containing parentheses (Wikipedia-style) — the URL pattern stops at the
  first `)`, so such links are not protected as a unit.
* Link destinations with a title (`[text](url "title")`) are not matched.
"""

from __future__ import annotations

import re

# Primary placeholder delimiter/label (0b: survives 5/5 syntactic positions).
PRIMARY_DELIM = "§"  # § section sign
PRIMARY_LABEL = "X"

# Alternate delimiter for the per-line marker-loss fallback (0b: also 5/5).
FALLBACK_DELIM = "◆"  # ◆ black diamond
FALLBACK_LABEL = "X"

# Delimiters tried in order when choosing a run-wide delimiter that does not
# collide with literal placeholder-shaped text in the source. Ordered by
# rarity: § is the primary; ◆ and # are documented alternates.
DELIM_CANDIDATES = [
    (PRIMARY_DELIM, PRIMARY_LABEL),
    (FALLBACK_DELIM, FALLBACK_LABEL),
    ("#", "X"),
]

# Markdown link: [text](url). The text is kept translatable; the URL is
# protected. URL = no whitespace, no parentheses (documented limitation).
_LINK_RE = re.compile(r"\[([^\]\n]*)\]\(([^()\s]*)\)")

# Inline code: single-backtick spans (double backticks not covered).
_CODE_RE = re.compile(r"`([^`\n]+)`")

# HTML tag: match by tag name (a letter-led name), not by arbitrary angle
# brackets, so comparisons like "a < b" are not mistaken for tags.
_HTML_RE = re.compile(r"</?[a-zA-Z][a-zA-Z0-9-]*(?:\s[^<>]*?)?/?>")


def placeholder_pattern(delim: str, label: str) -> re.Pattern[str]:
    """Regex matching a generated placeholder, e.g. §X12§."""
    return re.compile(re.escape(delim) + re.escape(label) + r"\d+" + re.escape(delim))


class Protector:
    """Extracts/restores inline spans for one delimiter/label and one counter.

    The counter advances across calls, so placeholders stay unique within a
    run; reset it per file by constructing a fresh Protector (or via
    ``reset()`` for the module-level default).
    """

    def __init__(self, delim: str = PRIMARY_DELIM, label: str = PRIMARY_LABEL) -> None:
        self.delim = delim
        self.label = label
        self.counter = 0

    def _ph(self, n: int) -> str:
        return f"{self.delim}{self.label}{n}{self.delim}"

    def _next(self) -> str:
        k = self._ph(self.counter)
        self.counter += 1
        return k

    def protect_line(self, line: str) -> tuple[str, dict[str, str]]:
        """Replace inline spans with placeholders. Returns (line, ph->original).

        Order is links -> inline code -> HTML (docs/handoff.md p.4), so nested
        patterns do not conflict. The link text `[...]` stays translatable;
        only the URL `(...)` is protected.
        """
        mapping: dict[str, str] = {}

        def link_repl(m: re.Match[str]) -> str:
            k = self._next()
            mapping[k] = m.group(2)  # URL only
            return f"[{m.group(1)}]({k})"

        def code_repl(m: re.Match[str]) -> str:
            k = self._next()
            mapping[k] = m.group(0)  # whole `code` span incl. backticks
            return k

        def html_repl(m: re.Match[str]) -> str:
            k = self._next()
            mapping[k] = m.group(0)  # whole tag
            return k

        line = _LINK_RE.sub(link_repl, line)
        line = _CODE_RE.sub(code_repl, line)
        line = _HTML_RE.sub(html_repl, line)
        return line, mapping

    def restore_line(self, line: str, placeholders: dict[str, str]) -> str:
        """Substitute placeholders back and fix the model's spurious ``] (``."""
        for k, v in placeholders.items():
            line = line.replace(k, v)
        line = line.replace("] (", "](")
        return line


# Module-level default protector (used by the top-level convenience API).
_protector = Protector()


def reset(delim: str = PRIMARY_DELIM, label: str = PRIMARY_LABEL) -> None:
    """Reset the module-level protector (called once per file)."""
    global _protector
    _protector = Protector(delim, label)


def protect_line(line: str) -> tuple[str, dict[str, str]]:
    return _protector.protect_line(line)


def restore_line(line: str, placeholders: dict[str, str]) -> str:
    return _protector.restore_line(line, placeholders)


def choose_delimiter(source_text: str) -> tuple[str, str]:
    """Pick a (delim, label) whose placeholder pattern is absent from the source.

    If the source already contains literal `§X…§`, switching the whole run to
    a different delimiter avoids shadowing real text (0b collision rule).
    """
    for delim, label in DELIM_CANDIDATES:
        if not placeholder_pattern(delim, label).search(source_text):
            return delim, label
    return DELIM_CANDIDATES[0]


def fallback_delimiter(primary: str) -> tuple[str, str]:
    """The alternate delimiter used when a line's markers are lost in transit."""
    if primary != FALLBACK_DELIM:
        return FALLBACK_DELIM, FALLBACK_LABEL
    return PRIMARY_DELIM, PRIMARY_LABEL


def markers_intact(protected: str, translated: str, placeholders: dict[str, str]) -> bool:
    """True when every placeholder returned verbatim (full set, exact counts)."""
    return all(translated.count(k) == protected.count(k) for k in placeholders)
