#!/usr/bin/env python3
"""CLI + orchestration (Phase 1 — framework).

Phase 1 scope: module split, `--file`/`--dir` traversal, `_ru` naming,
UTF-8 encoding handling (non-UTF-8 -> skip + warning), and a lazy single model
load.

Per-file pipeline:
  read (UTF-8) -> translate every line (lazy model; one line in -> one line
  out) -> write the `_ru` output next to the source.

Inline protection (Phase 2) and validation (Phase 3) live in
`inline_protect.py` / `validator.py` as stubs and are not wired in yet.
The code-fence FSM and symbol-line heuristics are also Phase 2.

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
import sys
import time
from pathlib import Path

import fileio
import translator

EXIT_OK = 0
EXIT_FATAL = 1
EXIT_USAGE = 2


def log(tag: str, msg: str) -> None:
    print(f"[{tag}] {msg}", flush=True)


def translate_lines(text: str) -> tuple[list[str], bool]:
    """Split text into lines (preserving a trailing newline) and translate each.

    Returns `(out_lines, had_trailing_newline)`. One input line -> one output
    line is guaranteed by `translator.translate_line` (model newlines are
    collapsed), so the output line count cannot drift.
    """
    had_trailing_newline = text.endswith("\n")
    lines = text.split("\n")
    if had_trailing_newline:
        lines = lines[:-1]

    out_lines: list[str] = []
    for idx, line in enumerate(lines, start=1):
        try:
            out_lines.append(translator.translate_line(line))
        except Exception as exc:  # a failing line must not abort the run
            out_lines.append(line)
            log("ERROR", f"line {idx}: {type(exc).__name__}: {exc}")
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
