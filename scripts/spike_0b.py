#!/usr/bin/env python3
"""Spike 0b: placeholder survival in MarianMT (SentencePiece) + edge cases.

What it does
------------
1. Loads Helsinki-NLP/opus-mt-en-ru (MarianMT) on CPU with the same
   parameters as 0a: torch.manual_seed(0), torch.set_num_threads(8),
   num_beams=4, max_length=512.
2. Part A -- tokenizer introspection: for every candidate placeholder family
   prints how MarianTokenizer (SentencePiece) splits the marker. A marker that
   is a *single* token is the strongest signal it will be copied verbatim.
3. Part B -- survival through generation: embeds three distinct placeholders
   of each family (indices 0,1,2) into one English sentence, translates it,
   and checks, for every marker, whether it came back verbatim (count + full
   set) and whether it was split/merged/dropped. Full input/output are printed.
4. Part C -- edge cases: fixed inputs (code line, fenced block line, HTML tag,
   Markdown link, empty string, string with an embedded newline) and their raw
   model output, to decide what the pipeline must protect/skip.
5. Part D -- scoring table and a recommended format, based on: verbatim
   survival (3/3), single-token atomicity, and delimiter rarity.

Nothing is written to disk. It is a read-only probe against the cached model.

Usage
-----
    export HF_HOME="$PWD/.tmp/hf-cache"
    .venv/bin/python scripts/spike_0b.py
    .venv/bin/python scripts/spike_0b.py --no-download

Environment
-----------
    HF_HOME   must point at a writable directory (the sandbox HOME is read-only).
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

MODEL_ID = "Helsinki-NLP/opus-mt-en-ru"
DEFAULT_SEED = 0
DEFAULT_THREADS = 8
NUM_BEAMS = 4
MAX_LENGTH = 512
REPO_ROOT = Path(__file__).resolve().parent.parent

DOWNLOAD_ALLOW_PATTERNS = ["*.json", "*.spm", "*.txt", "*.model", "pytorch_model.bin", "model.safetensors"]

# Candidate placeholder families. {i} is replaced with the slot index 0..2.
# Categories: word-like (control), ASCII delimiters, brackets, Unicode symbols,
# numeric, CJK/rare brackets. Symbols are written literally for readability.
FAMILIES = [
    # --- control / word-like (expected to fail or be mangled) ---
    ("double_underscore", "__INLINE_CODE_{i}__"),
    ("single_underscore", "__IC{i}__"),
    ("camel_word", "InlineCode{i}"),
    # --- ASCII delimiter + label ---
    ("dollar", "$IC{i}$"),
    ("hash", "#IC{i}#"),
    ("at", "@IC{i}@"),
    ("percent", "%IC{i}%"),
    ("caret", "^IC{i}^"),
    ("amp", "&IC{i}&"),
    ("plus", "+IC{i}+"),
    ("equal", "=IC{i}="),
    ("tilde", "~IC{i}~"),
    # --- brackets ---
    ("square_bracket", "[IC{i}]"),
    ("curly_brace", "{IC{i}}"),
    ("paren", "(IC{i})"),
    ("angle", "<IC{i}>"),
    # --- Unicode symbols (rare, likely atomic) ---
    ("section_X", "§X{i}§"),
    ("section", "§IC{i}§"),
    ("paragraph", "¶IC{i}¶"),
    ("dagger", "†IC{i}†"),
    ("double_dagger", "‡IC{i}‡"),
    ("lozenge", "◊IC{i}◊"),
    ("black_diamond_X", "◆X{i}◆"),
    ("black_diamond", "◆IC{i}◆"),
    ("white_diamond", "◇IC{i}◇"),
    ("bullet", "•IC{i}•"),
    ("black_circle", "●IC{i}●"),
    ("black_square", "▪IC{i}▪"),
    ("star", "★IC{i}★"),
    # --- numeric (digits are usually copied verbatim) ---
    ("numeric_plain", "{i}"),
    ("numeric_x", "0x{i}"),
    ("numeric_underscore", "_{i}_"),
    ("numeric_hash", "#{i}#"),
    ("numeric_section", "§{i}§"),
    # --- rare/CJK brackets ---
    ("cjk_double", "⟦{i}⟧"),
    ("cjk_angle", "《{i}》"),
    ("cjk_corner", "「{i}」"),
]


def log(tag: str, msg: str) -> None:
    print(f"[{tag}] {msg}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=MODEL_ID, help=f"HF model id (default: {MODEL_ID})")
    ap.add_argument("--threads", type=int, default=DEFAULT_THREADS, help="torch.set_num_threads (default: 8)")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED, help="torch.manual_seed (default: 0)")
    ap.add_argument("--no-download", action="store_true", help="never hit the network (cache only)")
    args = ap.parse_args()

    # --- imports (heavy) ---------------------------------------------------
    import torch
    from transformers import MarianMTModel, MarianTokenizer
    import transformers

    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)

    log("INFO", f"python={sys.version.split()[0]} torch={torch.__version__} "
                f"transformers={transformers.__version__} device=cpu")
    log("INFO", f"seed={args.seed} threads={args.threads} beams={NUM_BEAMS} max_length={MAX_LENGTH} "
                f"cuda_available={torch.cuda.is_available()}")
    log("INFO", f"model={args.model}")

    if args.no_download:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        log("INFO", "download skipped (--no-download)")
    else:
        from huggingface_hub import snapshot_download
        t0 = time.perf_counter()
        snapshot_download(args.model, allow_patterns=DOWNLOAD_ALLOW_PATTERNS)
        log("TIMING", f"download_model={time.perf_counter() - t0:.3f}s (cached runs are ~0s)")
        log("INFO", f"hf_home={os.environ.get('HF_HOME', '<default>')}")

    tokenizer = MarianTokenizer.from_pretrained(args.model)
    model = MarianMTModel.from_pretrained(args.model)
    model.to("cpu")
    model.eval()
    log("INFO", f"model_num_beams={getattr(model.config, 'num_beams', 'n/a')} "
                f"max_length={getattr(model.config, 'max_length', 'n/a')} "
                f"params={sum(p.numel() for p in model.parameters()):,}")

    def translate(texts):
        batch = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH)
        with torch.inference_mode():
            generated = model.generate(**batch, num_beams=NUM_BEAMS, max_length=MAX_LENGTH)
        return tokenizer.batch_decode(generated, skip_special_tokens=True)

    # --- Part A: tokenizer introspection ----------------------------------
    log("A", "=== Part A: how SentencePiece tokenizes each placeholder (index 0) ===")
    for name, tpl in FAMILIES:
        ph = tpl.replace("{i}", "0")
        pieces = tokenizer.tokenize(ph)
        ids = tokenizer.convert_tokens_to_ids(pieces)
        atomic = "ATOMIC" if len(pieces) == 1 else f"split-{len(pieces)}"
        print(f"[A] {name:20s} {atomic:8s} ph={ph!r} pieces={pieces} ids={ids}")

    # --- Part B: survival through generation ------------------------------
    log("B", "=== Part B: survival of 3 distinct markers (idx 0,1,2) through generation ===")
    template = "The marker {p0} denotes inline code and {p1} marks a link target; finally {p2} ends the line."
    rows = []
    all_sentences = []
    all_meta = []
    for name, tpl in FAMILIES:
        phs = [tpl.replace("{i}", str(i)) for i in (0, 1, 2)]
        s = template.replace("{p0}", phs[0]).replace("{p1}", phs[1]).replace("{p2}", phs[2])
        all_sentences.append(s)
        all_meta.append((name, phs))

    outputs = translate(all_sentences)
    for (name, phs), src, out in zip(all_meta, all_sentences, outputs):
        survived = 0
        detail = []
        for ph in phs:
            ok = out.count(ph) == src.count(ph)
            survived += int(ok)
            detail.append(f"{ph!r}:{'OK' if ok else 'LOST'}(in={src.count(ph)} out={out.count(ph)})")
        rows.append({"name": name, "survived": survived, "out": out, "detail": detail})
        print(f"[B] {name:20s} survived={survived}/3  {' | '.join(detail)}")
        print(f"    in : {src}")
        print(f"    out: {out}")

    # --- Part C: edge cases ----------------------------------------------
    log("C", "=== Part C: edge cases (input -> raw output) ===")
    fence_open = "```python"  # three backticks + python
    edge_cases = [
        ("code_line", 'import torch; model = torch.load("x.pt")'),
        ("code_fence_open", fence_open),
        ("html_tag", '<b>bold text</b> and <a href="https://x.dev">link</a>'),
        ("md_link", "See [this guide](https://example.com/path?q=1) for details."),
        ("empty_string", ""),
        ("newline_in_string", "First line.\nSecond line."),
    ]
    for name, text in edge_cases:
        try:
            out = translate([text])[0]
        except Exception as exc:  # empty input can make generation unhappy
            out = f"<ERROR {type(exc).__name__}: {exc}>"
        print(f"[C] {name:20s} in ={text!r}")
        print(f"    out={out!r}")

    # --- Part E: confirmation of finalists in varied syntactic positions ---
    log("E", "=== Part E: confirmation of finalist formats in varied contexts ===")
    FINALISTS = [
        ("section_X", "§X{i}§"),
        ("section_M", "§M{i}§"),
        ("paragraph_X", "¶X{i}¶"),
        ("black_diamond_X", "◆X{i}◆"),
        ("hash_X", "#X{i}#"),
        ("at_X", "@X{i}@"),
        ("hash_IC", "#IC{i}#"),
        ("section_IC", "§IC{i}§"),
    ]
    CONTEXTS = [
        ("start", "{p0} marks the beginning of this sentence."),
        ("end", "This sentence ends with the marker {p0}."),
        ("punct", 'Use the token ({p0}) and "{p0}" inside quotes.'),
        ("adjacent", "Two markers {p0} and {p1} sit next to each other."),
        ("mid_complex", "Run {p0} then read {p1} before you stop."),
    ]
    e_rows = []
    for name, tpl in FINALISTS:
        for ctx_name, ctx_tpl in CONTEXTS:
            phs = [tpl.replace("{i}", str(i)) for i in (0, 1)]
            s = ctx_tpl.replace("{p0}", phs[0]).replace("{p1}", phs[1])
            e_rows.append((name, ctx_name, s, phs))
    e_outputs = translate([r[2] for r in e_rows])
    e_summary = {}
    for (name, ctx_name, src, phs), out in zip(e_rows, e_outputs):
        present = [ph for ph in phs if ph in src]
        ok = all(out.count(ph) == src.count(ph) for ph in present)
        e_summary.setdefault(name, []).append(ok)
        if not ok:
            print(f"[E] {name:20s} ctx={ctx_name:12s} FAIL  in ={src}")
            print(f"    out={out}")
    print("[E] summary (contexts passing / total):")
    for name, _ in FINALISTS:
        flags = e_summary.get(name, [])
        print(f"[E] {name:20s} {sum(flags)}/{len(flags)}")
        for (n, c, src, phs), out in zip(e_rows, e_outputs):
            if n == name:
                print(f"    [{c}] {src}")
                print(f"          -> {out}")

    # --- Part D: scoring + recommendation ---------------------------------
    log("D", "=== Part D: scoring + recommendation ===")
    # Genuinely rare Unicode delimiters. ASCII punctuation ($ # @ % & + ~ =)
    # is common in markdown/code and is NOT "rare", so it does not qualify here.
    RARE = set("§¶†‡◊◆◇•●▪★⟦⟧《》「」")

    def is_rare_delimited(tpl):
        # Check the FRAME symbols (non-alphanumeric delimiters around the
        # label/digit) are rare. The label letters (X/IC/M) are intentional and
        # must NOT make the marker look "common".
        core = tpl.replace("{i}", "")
        delims = [c for c in core if not c.isalnum()]
        return bool(delims) and all(c in RARE for c in delims)

    def is_bare_digit(name):
        # Digit-only core (no opaque letter label) collides with real numbers.
        return name.startswith("numeric_")

    # A usable placeholder must (a) survive 3/3 verbatim in Part B, (b) use a
    # rare delimiter (low collision risk), and (c) carry an opaque letter label
    # (not a bare digit that collides with real numbers).
    print(f"[D] {'name':20s} {'surv':>4s} {'atomic':>6s} {'rare-delim':>11s} verdict")
    best = None
    for name, tpl in FAMILIES:
        ph0 = tpl.replace("{i}", "0")
        atomic = len(tokenizer.tokenize(ph0)) == 1
        rare = is_rare_delimited(tpl)
        row = next(r for r in rows if r["name"] == name)
        if is_bare_digit(name):
            verdict = "reject (digit core: collides with numbers)"
        elif row["survived"] != 3:
            verdict = "reject (lost)"
        elif not rare:
            verdict = "works but delimiter common"
        else:
            verdict = "candidate"
        if row["survived"] == 3 and rare and not is_bare_digit(name):
            if best is None:
                best = (name, tpl)
        print(f"[D] {name:20s} {row['survived']:>4d} {str(atomic):>6s} {str(rare):>11s} {verdict}")
    if best:
        log("D", f"RECOMMENDED family={best[0]!r} template={best[1]!r} (3/3 verbatim + rare delimiter + label)")
    else:
        log("D", "no rare-delimited labeled 3/3 survivor; review Part B manually")

    # --- Part F: placeholders on real inline positions of the fixture ---------
    log("F", "=== Part F: §X{n}§ on real inline positions of the fixture ===")
    LINK_RE = re.compile(r"\[([^\]]*)\]\(([^)]*)\)")
    CODE_RE = re.compile(r"`([^`]+)`")

    def protect_inline(line):
        mapping = {}
        counter = [0]

        def new_ph():
            k = f"§X{counter[0]}§"
            counter[0] += 1
            return k

        def link_repl(m):
            k = new_ph()
            mapping[k] = m.group(2)  # protect URL only, keep [text] translatable
            return f"[{m.group(1)}]({k})"

        def code_repl(m):
            k = new_ph()
            mapping[k] = m.group(0)  # protect the whole inline-code span
            return k

        line = LINK_RE.sub(link_repl, line)
        line = CODE_RE.sub(code_repl, line)
        return line, mapping

    def restore(line, mapping):
        for k, v in mapping.items():
            line = line.replace(k, v)
        return line

    fixture = REPO_ROOT / "tests" / "fixtures" / "00-setup-README.md"
    fixture_lines = fixture.read_text(encoding="utf-8").splitlines() if fixture.is_file() else []
    for n in (3, 30, 57, 149):
        if n > len(fixture_lines):
            continue
        line = fixture_lines[n - 1]
        prot, mapping = protect_inline(line)
        if not mapping:
            continue
        out = translate([prot])[0]
        lost = [k for k in mapping if out.count(k) != prot.count(k)]
        restored = restore(out, mapping)
        print(f"[F] fixture line {n}:")
        print(f"    in  : {line}")
        print(f"    prot: {prot}")
        print(f"    out : {out}")
        print(f"    lost: {lost if lost else 'none'}  restored: {restored}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
