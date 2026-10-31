#!/usr/bin/env python3
"""Validation — Phase 3 stub.

Phase 3 will implement:
* Blind check: line-count match + full placeholder set (count and identity).
* Diff-error detection: anomalies such as a non-empty source line translating
  to empty; writes a diff log.
* The stdout contract: `[VALIDATION SUCCESS]` / `[VALIDATION WARNING]` /
  `[ERROR]` / `[SKIP]` / `[INFO]` / `[TIMING]`.

This module exists now with the interface Phase 3 will fill in, so main.py's
call shape is fixed in advance.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ValidationResult:
    """Outcome of validating one file's translation."""
    line_count_match: bool
    issues: list[str] = field(default_factory=list)  # each "line N: message"

    @property
    def ok(self) -> bool:
        return self.line_count_match and not self.issues


def validate(original_lines: list[str], translated_lines: list[str]) -> ValidationResult:
    """Stub. Phase 3: real blind + diff-error checks.

    The only check kept now is the cheap line-count comparison (no model call),
    which Phase 3 builds on. Full `[VALIDATION ...]` logging is not emitted yet.
    """
    return ValidationResult(line_count_match=(len(original_lines) == len(translated_lines)))
