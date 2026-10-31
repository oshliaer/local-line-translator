#!/usr/bin/env python3
"""Model access: lazy single load + line translation (Phase 1).

Phase 1 responsibilities
------------------------
* Load `Helsinki-NLP/opus-mt-en-ru` (MarianMT) exactly once, lazily, on the
  first call to `translate_line` — not at import time and not once per file.
* Pin determinism and limits: `device='cpu'`, `torch.manual_seed(0)`,
  `torch.set_num_threads(8)`.
* `translate_line(line) -> str` always returns exactly one line (model
  newlines are collapsed; see `normalize_model_output`).

NOT here yet (Phase 2): code-fence FSM, inline placeholder protection, and the
symbol-only line heuristic. `translate_line` only refuses empty/whitespace-only
input so MarianMT never hallucinates on empty strings (documented in 0a/0b).
"""

from __future__ import annotations

import os
import re
import time

# Silence the tokenizers fork warning before transformers is ever imported.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

MODEL_ID = "Helsinki-NLP/opus-mt-en-ru"
DEVICE = "cpu"
DEFAULT_SEED = 0
DEFAULT_THREADS = 8

# Lazy singleton state; populated by load_model() on first use.
_CONFIG = {"model_id": MODEL_ID, "seed": DEFAULT_SEED, "threads": DEFAULT_THREADS}
_tokenizer = None
_model = None
_torch = None
_loaded = False


def normalize_model_output(text: str) -> str:
    """Collapse the model's output onto exactly one line (from spike 0a)."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def configure(model_id: str | None = None, seed: int | None = None, threads: int | None = None) -> None:
    """Override defaults for the (single) load. No-op after the model is loaded.

    Kept separate from `load_model` so callers can pass CLI options without
    triggering the heavy torch/transformers import eagerly.
    """
    if model_id is not None:
        _CONFIG["model_id"] = model_id
    if seed is not None:
        _CONFIG["seed"] = seed
    if threads is not None:
        _CONFIG["threads"] = threads


def is_loaded() -> bool:
    return _loaded


def load_model() -> None:
    """Load tokenizer + model once. Idempotent: no-op after the first success.

    The heavy imports (torch/transformers) happen here, not at module import,
    so importing this module stays cheap.
    """
    global _tokenizer, _model, _torch, _loaded
    if _loaded:
        return

    import torch
    from transformers import MarianMTModel, MarianTokenizer

    model_id = _CONFIG["model_id"]
    seed = _CONFIG["seed"]
    threads = _CONFIG["threads"]

    torch.manual_seed(seed)
    torch.set_num_threads(threads)

    t0 = time.perf_counter()
    tokenizer = MarianTokenizer.from_pretrained(model_id)
    model = MarianMTModel.from_pretrained(model_id).to(DEVICE)
    model.eval()
    load_s = time.perf_counter() - t0

    _torch = torch
    _tokenizer = tokenizer
    _model = model
    _loaded = True

    print(
        f"[INFO] model_loaded model={model_id} device={DEVICE} seed={seed} "
        f"threads={threads} load_s={load_s:.3f} "
        f"num_beams={getattr(model.config, 'num_beams', 'n/a')} "
        f"max_length={getattr(model.config, 'max_length', 'n/a')}",
        flush=True,
    )


def translate_line(line: str) -> str:
    """Translate one line. Loads the model on first use. Returns exactly one line.

    Empty/whitespace-only lines are returned unchanged and never sent to the
    model (empty input makes MarianMT hallucinate — see HANDOFF.md, 0a/0b).
    """
    if not line.strip():
        return line
    load_model()
    batch = _tokenizer([line], return_tensors="pt", padding=True, truncation=True, max_length=512)
    with _torch.inference_mode():
        generated = _model.generate(**batch)
    return normalize_model_output(_tokenizer.batch_decode(generated, skip_special_tokens=True)[0])
