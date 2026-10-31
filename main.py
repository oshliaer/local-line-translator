#!/usr/bin/env python3
"""CLI + orchestration (Phase 2 — string pipeline).

Phase 2 scope: the per-line pipeline that makes line-in == line-out safe for
Markdown: a code-fence FSM (fences and their contents are copied verbatim),
skip heuristics for symbol-only and CJK lines, and inline protection/restore
(links -> inline code -> HTML) around the model call.

Per-file pipeline:
  read (UTF-8) -> for each line: fence FSM -> skip heuristics -> protect inline
  spans -> translate (lazy single model load) -> verify markers -> restore ->
  write the `_ru` output next to the source.

Validation (Phase 3) and atomic write / CRLF handling (Phase 4) live in
`validator.py` / `fileio.py` and are not wired in yet.

Usage
-----
    .venv/bin/python main.py --file tests/fixtures/00-setup-README.md
    .venv/bin/python main.py --dir some/docs/dir
    .venv/bin/python main.py --dir docs --threads 8 --seed 0

Environment
-----------
    HF_HOME   must point at a writable directory (the sandbox HOME is read-only).
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import fileio
import inline_protect
import translator

EXIT_OK = 0
EXIT_FATAL = 1
EXIT_USAGE = 2

# Code-fence detection: the first non-whitespace characters are 3+ backticks
# (optionally followed by an info string) or 3+ tildes. A line that ALSO
# contains the fence character later (```code``` on one line) is
# NOT a clean fence — documented decision, see HANDOFF.md.
_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})(.*)$")
_VOWELS = frozenset("aeiouAEIOU")


def log(tag: str, msg: str) -> None:
    print(f"[{tag}] {msg}", flush=True)


def is_fence(line: str) -> bool:
    """True when `line` is a code-fence delimiter (opening or closing)."""
    m = _FENCE_RE.match(line)
    if not m:
        return False
    marker = m.group(1)[0]  # '`' or '~'
    return marker not in m.group(2)


def has_cjk(line: str) -> bool:
    """True when the line carries CJK/fullwidth characters (skip wholesale).

    MarianMT (EN->RU) garbles CJK text even when the URL around it survives
    (0b fact), so such lines are copied verbatim rather than inline-protected.
    """
    for ch in line:
        cp = ord(ch)
        if (0x3000 <= cp <= 0x303F or  # CJK symbols/punctuation
                0x3040 <= cp <= 0x30FF or  # Hiragana/Katakana
                0x3400 <= cp <= 0x4DBF or  # CJK Unified Ext. A
                0x4E00 <= cp <= 0x9FFF or  # CJK Unified Ideographs
                0xAC00 <= cp <= 0xD7AF or  # Hangul syllables
                0xFF00 <= cp <= 0xFFEF):   # fullwidth forms
            return True
    return False


def is_symbol_only(line: str) -> bool:
    """Heuristic for untranslatable lines (table separators, paths, commands).

    Skips empty/whitespace-only, lines with no ASCII letters, lines with a low
    letter density, and lines whose letters carry no vowels (shell commands,
    config keys, paths). See docs/handoff.md p.2.
    """
    stripped = line.strip()
    if not stripped:
        return True
    visible = [c for c in stripped if not c.isspace()]
    if not visible:
        return True
    letters = [c for c in visible if "a" <= c <= "z" or "A" <= c <= "Z"]
    if not letters:
        return True
    if len(letters) / len(visible) < 0.3:
        return True
    if not any(c in _VOWELS for c in letters):
        return True
    return False


def _has_ascii_letters(text: str) -> bool:
    return any("a" <= c <= "z" or "A" <= c <= "Z" for c in text)


def translate_line_protected(line: str, delim: str) -> str:
    """Protect inline spans, translate, verify markers, restore; fall back on loss.

    Marker-loss path: retry the line with the alternate delimiter (◆X{n}◆);
    if that also loses a marker, copy the original line and log a warning —
    never fail and never drop a marker silently. The full diff-log is Phase 3.
    """
    protected, mapping = inline_protect.protect_line(line)
    if not mapping:
        return translator.translate_line(line)
    if not _has_ascii_letters(protected):
        # The line is entirely inline spans (code/link/html only): nothing to
        # translate — keep it verbatim instead of feeding bare markers to the
        # model.
        return line

    translated = translator.translate_line(protected)
    if inline_protect.markers_intact(protected, translated, mapping):
        return inline_protect.restore_line(translated, mapping)

    # Fallback: re-protect and re-translate with the alternate delimiter.
    fdelim, flabel = inline_protect.fallback_delimiter(delim)
    alt = inline_protect.Protector(fdelim, flabel)
    protected2, mapping2 = alt.protect_line(line)
    translated2 = translator.translate_line(protected2)
    if inline_protect.markers_intact(protected2, translated2, mapping2):
        return alt.restore_line(translated2, mapping2)

    log("WARNING", f"inline markers lost; copied original verbatim: {line!r}")
    return line


def translate_lines(text: str) -> tuple[list[str], bool]:
    """Split text into lines (preserving a trailing newline) and translate each.

    Returns `(out_lines, had_trailing_newline)`. One input line -> one output
    line is guaranteed: fences/code/skipped lines are copied verbatim, and
    `translator.translate_line` collapses model newlines.
    """
    had_trailing_newline = text.endswith("\n")
    lines = text.split("\n")
    if had_trailing_newline:
        lines = lines[:-1]

    delim, label = inline_protect.choose_delimiter(text)
    inline_protect.reset(delim, label)
    if (delim, label) != (inline_protect.PRIMARY_DELIM, inline_protect.PRIMARY_LABEL):
        log("INFO", f"placeholder delimiter switched to {delim}{label}{{n}}{delim} "
                    f"(collision with source {inline_protect.PRIMARY_DELIM}X…{inline_protect.PRIMARY_DELIM})")

    out_lines: list[str] = []
    in_code_block = False
    for idx, line in enumerate(lines, start=1):
        try:
            if is_fence(line):
                out_lines.append(line)
                in_code_block = not in_code_block
                continue
            if in_code_block:
                out_lines.append(line)
                continue
            if has_cjk(line) or is_symbol_only(line):
                out_lines.append(line)
                continue
            out_lines.append(translate_line_protected(line, delim))
        except Exception as exc:  # a failing line must not abort the run
            out_lines.append(line)
            log("ERROR", f"line {idx}: {type(exc).__name__}: {exc}")

    if in_code_block:
        log("WARNING", "unbalanced code fence: file ends inside a code block "
                       "(tail copied verbatim)")
    return out_lines, had_trailing_newline


def process_file(src: Path) -> bool:
    """Translate one input file and write its `_ru` output. Returns success."""
    try:
        text = fileio.read_text_utf8(src)
    except OSError as exc:
        log("ERROR", f"{src}: {type(exc).__name__}: {exc}")
        return False

    if text is None:
        log("SKIP", f"{src}: not valid UTF-8, skipped (non-UTF-8 support is out of scope)")
        return True  # a skip is not a failure

    out_lines, had_trailing_newline = translate_lines(text)

    dst = fileio.output_path_for(src)
    content = "\n".join(out_lines) + ("\n" if had_trailing_newline else "")
    try:
        fileio.write_text(dst, content)
    except OSError as exc:
        log("ERROR", f"{dst}: {type(exc).__name__}: {exc}")
        return False

    log("INFO", f"{src}: lines={len(out_lines)} -> {dst.name}")
    return True


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    src_group = ap.add_mutually_exclusive_group(required=True)
    src_group.add_argument("--file", metavar="PATH", help="single file to translate")
    src_group.add_argument("--dir", metavar="PATH", help="directory to walk for .md/.txt files")
    ap.add_argument("--model", default=translator.MODEL_ID,
                    help=f"HF model id (default: {translator.MODEL_ID})")
    ap.add_argument("--threads", type=int, default=translator.DEFAULT_THREADS,
                    help=f"torch.set_num_threads (default: {translator.DEFAULT_THREADS})")
    ap.add_argument("--seed", type=int, default=translator.DEFAULT_SEED,
                    help=f"torch.manual_seed (default: {translator.DEFAULT_SEED})")
    args = ap.parse_args(argv)

    translator.configure(model_id=args.model, seed=args.seed, threads=args.threads)

    if args.file:
        src = Path(args.file)
        if not src.is_file():
            log("ERROR", f"file not found: {src}")
            return EXIT_USAGE
        return EXIT_OK if process_file(src) else EXIT_FATAL

    root = Path(args.dir)
    if not root.is_dir():
        log("ERROR", f"directory not found: {root}")
        return EXIT_USAGE

    files = list(fileio.iter_input_files(root))
    if not files:
        log("INFO", f"no .md/.txt input files under {root}")
        return EXIT_OK

    log("INFO", f"dir={root} found={len(files)} input file(s)")
    t_start = time.perf_counter()
    failures = 0
    for src in files:
        if not process_file(src):
            failures += 1
    total = time.perf_counter() - t_start
    log("INFO", f"dir done: ok={len(files) - failures}/{len(files)} total_s={total:.3f}s")
    return EXIT_FATAL if failures else EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
