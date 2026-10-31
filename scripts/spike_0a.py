#!/usr/bin/env python3
"""Spike 0a: CPU-only EN->RU translation of the reference fixture.

What it does
------------
1. Loads Helsinki-NLP/opus-mt-en-ru (MarianMT) on CPU, measuring download
   (first run only) and load time.
2. Walks the reference file line by line with a *simple* fence FSM and skips:
     - empty lines,
     - lines that contain no letters at all (whitespace / punctuation only,
       e.g. table separators, "---", ASCII art rules),
     - lines inside fenced code blocks and the fence lines themselves.
   Inline protection is Phase 0b and is deliberately NOT implemented here.
3. Translates every remaining line, guaranteeing one output line per input
   line (model newlines are collapsed, so the line count cannot drift).
4. Reports timings and peak RSS to stdout with [INFO]/[TIMING] prefixes.

By default nothing is written to disk. Use --out DIR to dump the translation
for manual inspection (point it at a temporary directory, never at the repo).

Usage
-----
    .venv/bin/python scripts/spike_0a.py
    .venv/bin/python scripts/spike_0a.py --threads 8
    .venv/bin/python scripts/spike_0a.py --out /tmp/spike-out

Environment
-----------
    HF_HOME   must point at a writable directory (the sandbox HOME is read-only)
"""

from __future__ import annotations

import argparse
import os
import re
import resource
import sys
import time
from pathlib import Path

# Must be set before transformers/tokenizers import to silence the fork warning.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

MODEL_ID = "Helsinki-NLP/opus-mt-en-ru"
DEFAULT_SEED = 0
DEFAULT_THREADS = 8

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FILE = REPO_ROOT / "tests" / "fixtures" / "00-setup-README.md"

# Files actually needed to run the PyTorch MarianMT model. The repo also
# ships rust_model.ot (~560 MB) which we never load; snapshot_download() without
# this filter would fetch it too.
DOWNLOAD_ALLOW_PATTERNS = ["*.json", "*.spm", "*.txt", "*.model", "pytorch_model.bin", "model.safetensors"]

# A fence line: optional indent, then 3+ backticks or 3+ tildes.
FENCE_RE = re.compile(r"^[ \t]*(`{3,}|~{3,})(.*)$")


def log(tag: str, msg: str) -> None:
    print(f"[{tag}] {msg}", flush=True)


def peak_rss_mb() -> float:
    """Peak resident set size of this process in MiB (Linux: ru_maxrss is KiB)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def is_translatable(line: str) -> bool:
    """False for empty lines and lines without a single letter.

    Letters (not digits/punctuation) are the practical proxy for "there is
    something here for the model to translate". Table rules like
    '| --- | --- |' and hr lines like '---' therefore get skipped, while
    headings, prose, list items and table rows with text get translated.
    """
    stripped = line.strip()
    if not stripped:
        return False
    return any(ch.isalpha() for ch in stripped)


class FenceFSM:
    """Minimal in-fence tracker.

    Deliberately simple (Phase 0a): a line is a fence when its first
    non-whitespace run is 3+ backticks or 3+ tildes. A fence of the same
    character with at least the same length and nothing but whitespace after
    it closes the block. Unclosed blocks swallow the rest of the file, which
    is exactly the behaviour we want to be able to observe and log.
    """

    def __init__(self) -> None:
        self.in_block = False
        self._char = ""
        self._len = 0
        self.unclosed = False

    def feed(self, line: str) -> bool:
        """Return True when the line belongs to a code block (incl. fences)."""
        m = FENCE_RE.match(line)
        if not self.in_block:
            if m:
                fence = m.group(1)
                self.in_block = True
                self._char = fence[0]
                self._len = len(fence)
                self.unclosed = True
                return True
            return False

        # Inside a block: only a matching closing fence ends it.
        if m:
            fence, rest = m.group(1), m.group(2)
            if fence[0] == self._char and len(fence) >= self._len and not rest.strip():
                self.in_block = False
                self.unclosed = False
                self._char = ""
                self._len = 0
                return True
        return True


def normalize_model_output(text: str) -> str:
    """Collapse the model's output onto exactly one line."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--file", default=str(DEFAULT_FILE),
                    help="file to translate (default: reference fixture)")
    ap.add_argument("--model", default=MODEL_ID, help=f"HF model id (default: {MODEL_ID})")
    ap.add_argument("--threads", type=int, default=DEFAULT_THREADS,
                    help="torch.set_num_threads (default: 8)")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED, help="torch.manual_seed (default: 0)")
    ap.add_argument("--limit", type=int, default=0, help="only process the first N lines (0 = all)")
    ap.add_argument("--out", default="",
                    help="optional directory for the translated file (default: write nothing)")
    ap.add_argument("--no-download", action="store_true", help="never hit the network (cache only)")
    args = ap.parse_args()

    src = Path(args.file)
    if not src.is_file():
        log("ERROR", f"file not found: {src}")
        return 2

    # --- imports (heavy) ---------------------------------------------------
    t_import0 = time.perf_counter()
    import torch
    from transformers import MarianMTModel, MarianTokenizer
    import transformers
    t_import = time.perf_counter() - t_import0

    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)

    log("INFO", f"python={sys.version.split()[0]} torch={torch.__version__} "
                f"transformers={transformers.__version__} device=cpu")
    log("INFO", f"seed={args.seed} threads={args.threads} "
                f"cuda_available={torch.cuda.is_available()}")
    log("INFO", f"model={args.model}")
    log("TIMING", f"import_libs={t_import:.3f}s")

    # --- download (only actually downloads on a cold HF cache) -------------
    download_s = 0.0
    if args.no_download:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        log("INFO", "download skipped (--no-download)")
    else:
        from huggingface_hub import snapshot_download
        t_dl0 = time.perf_counter()
        snapshot_download(args.model, allow_patterns=DOWNLOAD_ALLOW_PATTERNS)
        download_s = time.perf_counter() - t_dl0
        log("TIMING", f"download_model={download_s:.3f}s (cached runs are ~0s)")
        log("INFO", f"hf_home={os.environ.get('HF_HOME', '<default>')}")

    # --- load --------------------------------------------------------------
    rss_before_load = peak_rss_mb()
    t_load0 = time.perf_counter()
    tokenizer = MarianTokenizer.from_pretrained(args.model)
    model = MarianMTModel.from_pretrained(args.model)
    model.eval()
    load_s = time.perf_counter() - t_load0
    peak_after_load = peak_rss_mb()
    log("TIMING", f"load_model={load_s:.3f}s")
    log("TIMING", f"peak_rss_after_load_mb={peak_after_load:.1f} before_load_mb={rss_before_load:.1f}")
    log("INFO", f"model_num_beams={getattr(model.config, 'num_beams', 'n/a')} "
                f"max_length={getattr(model.config, 'max_length', 'n/a')} "
                f"params={sum(p.numel() for p in model.parameters()):,}")

    # --- read --------------------------------------------------------------
    raw = src.read_text(encoding="utf-8")
    had_trailing_newline = raw.endswith("\n")
    in_lines = raw.split("\n")
    if had_trailing_newline:
        in_lines = in_lines[:-1]
    total_lines = len(in_lines)
    log("INFO", f"file={src} lines={total_lines} trailing_newline={had_trailing_newline}")

    def translate_one(line: str) -> str:
        batch = tokenizer([line], return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.inference_mode():
            generated = model.generate(**batch)
        return normalize_model_output(tokenizer.batch_decode(generated, skip_special_tokens=True)[0])

    # --- walk + translate --------------------------------------------------
    fsm = FenceFSM()
    out_lines = []
    translated = skipped_empty = skipped_symbols = skipped_code = 0
    model_seconds = 0.0

    t_file0 = time.perf_counter()
    for idx, line in enumerate(in_lines, start=1):
        if args.limit and idx > args.limit:
            out_lines.append(line)
            continue
        if fsm.feed(line):
            out_lines.append(line)
            skipped_code += 1
            continue
        if not is_translatable(line):
            out_lines.append(line)
            if line.strip():
                skipped_symbols += 1
            else:
                skipped_empty += 1
            continue
        t0 = time.perf_counter()
        try:
            out_lines.append(translate_one(line))
        except Exception as exc:  # never abort the whole run on one line
            out_lines.append(line)
            log("ERROR", f"line {idx}: {type(exc).__name__}: {exc}")
        model_seconds += time.perf_counter() - t0
        translated += 1
    total_s = time.perf_counter() - t_file0
    peak_rss = peak_rss_mb()

    skipped = skipped_empty + skipped_symbols + skipped_code
    per_line_ms = (model_seconds / translated * 1000.0) if translated else 0.0

    if fsm.unclosed:
        log("WARNING", "unclosed code fence: the rest of the file was treated as code")

    # --- line integrity ----------------------------------------------------
    out_count = len(out_lines)
    log("INFO", f"line_count in={total_lines} out={out_count} match={total_lines == out_count}")
    log("INFO", f"lines translated={translated} skipped={skipped} "
                f"(empty={skipped_empty} symbols={skipped_symbols} code={skipped_code})")
    log("TIMING", f"file={src.name} lines={total_lines} translated={translated} skipped={skipped} "
                  f"total_s={total_s:.3f} per_line_ms={per_line_ms:.1f} peak_rss_mb={peak_rss:.1f}")
    log("TIMING", f"summary load_model={load_s:.3f}s download_model={download_s:.3f}s "
                  f"file_total={total_s:.3f}s translate_total={model_seconds:.3f}s "
                  f"per_line_ms={per_line_ms:.1f} peak_rss_mb={peak_rss:.1f} "
                  f"threads={args.threads} beams={getattr(model.config, 'num_beams', 'n/a')}")

    if args.out:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        dst = out_dir / (src.stem + "_ru" + src.suffix)
        dst.write_text("\n".join(out_lines) + ("\n" if had_trailing_newline else ""), encoding="utf-8")
        log("INFO", f"wrote {dst}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
