#!/usr/bin/env python3
"""File I/O: traversal, _ru naming, encoding (Phase 1).

Phase 1 responsibilities
------------------------
* Walk a directory recursively, yielding `.md`/`.txt` files only.
* Skip hidden (dot-prefixed) directories, symlinks, and files whose stem
  already ends with `_ru` (the "already translated" guard).
* Derive the output path `README.md -> README_ru.md` (same directory).
* Read files as UTF-8 and signal non-UTF-8 content so the caller skips + warns
  instead of crashing the whole run.

NOT here yet:
* Phase 4 — atomic write (temp + `os.replace`), CRLF/LF preservation, long
  line handling, exit-code contract. `write_text` is a plain write for now.
"""

from __future__ import annotations

import os
from pathlib import Path

TARGET_SUFFIXES = {".md", ".txt"}


def is_target_file(path: Path) -> bool:
    return path.suffix.lower() in TARGET_SUFFIXES


def is_ru_output(path: Path) -> bool:
    """True when the file looks like an existing translated output.

    Uses `Path.stem.endswith("_ru")`, NOT a substring check: `bar_rus.md`
    contains the substring `_ru` but its stem ends with `rus`, so it is a
    normal input and must be translated. `bar_ru.md` ends with `_ru` and is
    skipped.
    """
    return path.stem.endswith("_ru")


def output_path_for(src: Path) -> Path:
    """`README.md -> README_ru.md`, `notes.txt -> notes_ru.txt` (same dir)."""
    return src.with_name(src.stem + "_ru" + src.suffix)


def iter_input_files(root: Path):
    """Yield `.md`/`.txt` input files under `root`, recursively.

    Skips hidden directories, symlinks, non-target suffixes and `_ru` outputs.
    Deterministic: directories and files are visited in sorted order.
    """
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
        for name in sorted(filenames):
            p = Path(dirpath) / name
            if p.is_symlink():
                continue
            if not is_target_file(p):
                continue
            if is_ru_output(p):
                continue
            yield p


def read_text_utf8(path: Path) -> str | None:
    """Read `path` as UTF-8. Return `None` when decoding fails (non-UTF-8).

    The caller logs a `[SKIP]` warning and keeps going; the whole run must not
    abort on one bad file. OSError (missing/permission) still propagates.
    """
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return None


def write_text(path: Path, text: str) -> None:
    """Write `text` to `path` (UTF-8).

    Phase 4 will make this atomic (temp file + `os.replace`) and preserve the
    source line-ending style (CRLF/LF). Phase 1 uses a plain write.
    """
    path.write_text(text, encoding="utf-8")
