#!/usr/bin/env python3
"""
logit_lens_analysis.py — Phase 4a: Logit Lens Diagnostic Analysis
==================================================================
Honours Thesis: "Enhancing Reasoning Abilities of LLMs"
Gabriel D'Costa, Macquarie University. Supervisor: Usman Naseem.

PURPOSE
-------
Secondary diagnostic to complement the primary activation patching results
(Phases 3a–3c).  For each contrast example, projects the residual stream
at every layer through the unembedding matrix and records:
  - the logit assigned to the gold answer's first token
  - its rank in the vocabulary distribution
  - whether it is the top-1 prediction

IMPORTANT METHODOLOGICAL NOTE
------------------------------
Clean contrast and noisy contrast are SEPARATE example sets with different
selection criteria:
  - Clean contrast: Cell A (direct clean) WRONG  and  Cell C (structured clean) CORRECT
  - Noisy contrast: Cell B (direct noisy) WRONG  and  Cell D (structured noisy) CORRECT

The noisy contrast set is loaded from its own file (--noisy-contrast), NOT
derived from the clean contrast IDs.  These are typically different examples.

Logit lens is a SECONDARY diagnostic — correlational, not causal.  Activation
patching (Phases 3a–3c) provides the primary causal evidence.

USAGE
-----
  # Clean contrast only (default)
  python scripts/phase_4a_logit_lens/logit_lens_analysis.py

  # Noisy contrast only
  python scripts/phase_4a_logit_lens/logit_lens_analysis.py --noisy

  # Both in one run
  python scripts/phase_4a_logit_lens/logit_lens_analysis.py --include-noisy

  # Custom paths (all args have sensible defaults — override only as needed)
  python scripts/phase_4a_logit_lens/logit_lens_analysis.py \
      --include-noisy --max-examples 30

INPUT FILES
-----------
  dataset/processed/contrast_examples.json       — clean contrast (A wrong, C correct)
      Expected keys: example_id, gold_answer, cell_A.prompt, cell_C.prompt
  dataset/processed/noisy_contrast_examples.json — noisy contrast (B wrong, D correct)
      Expected keys: example_id, gold_answer, cell_B.prompt, cell_D.prompt
      OR: example_id, gold_answer (prompts fetched from dataset.json)
  dataset/processed/dataset.json                 — full 200-example dataset (fallback
      for noisy prompts if not embedded in contrast file)

OUTPUT FILES
------------
  results/phase_4a_logit_lens/<model>/logit_lens_per_example.csv        — default A/C
  results/phase_4a_logit_lens/<model>/logit_lens_summary.csv            — default A/C
  results/phase_4a_logit_lens/<model>/noisy_logit_lens_per_example.csv  — B/D noisy
  results/phase_4a_logit_lens/<model>/noisy_logit_lens_summary.csv      — B/D noisy
  figures/phase_4a_logit_lens/<model>/logit_lens_top1.png               — default A/C
  figures/phase_4a_logit_lens/<model>/logit_lens_logit.png              — default A/C
  figures/phase_4a_logit_lens/<model>/noisy_logit_lens_top1.png         — B/D noisy
  figures/phase_4a_logit_lens/<model>/noisy_logit_lens_logit.png        — B/D noisy

Thesis terminology:
  direct prompt, structured prompt, filler control, contrast examples,
  bridge entity, causal mediation effect
"""

import argparse
import csv
import gc
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless figure saving
import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    from scripts.utils.contrast_config import contrast_path_for, model_file_prefix, output_prefix_for
except ModuleNotFoundError:
    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))
    from scripts.utils.contrast_config import contrast_path_for, model_file_prefix, output_prefix_for

# ============================================================================
# Prompt materialisation (shared utility — identical to build_dataset.py)
# ============================================================================

def materialise_prompt(cell_dict, tokenizer) -> str:
    """
    Reconstruct the exact runnable model input from the stored cell schema.

    Supports both the new schema (dict with 'prompt', 'prefix_eos_pad',
    optional 'inline_eos_filler') and legacy format (plain string).

    For Cell E, inline filler is inserted before the final '\\nAnswer:'
    suffix in the clean prompt text.
    """
    if isinstance(cell_dict, str):
        return cell_dict

    eos = tokenizer.eos_token
    prompt = cell_dict["prompt"]
    prefix_pad = cell_dict.get("prefix_eos_pad", 0)
    inline_filler = cell_dict.get("inline_eos_filler", 0)

    if inline_filler > 0:
        marker = "\nAnswer:"
        idx = prompt.rfind(marker)
        if idx != -1:
            prompt = prompt[:idx] + (eos * inline_filler) + prompt[idx:]
        else:
            prompt = prompt + (eos * inline_filler)

    if prefix_pad > 0:
        prompt = (eos * prefix_pad) + prompt

    return prompt


# ============================================================================
# Logging
# ============================================================================

VERBOSE = False


def log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def vlog(msg: str) -> None:
    if VERBOSE:
        log(msg)


def fmt_s(s: float) -> str:
    return f"{s:.1f}s" if s < 60 else f"{int(s // 60)}m {s % 60:.1f}s"


def cuda_mem(device: str) -> str:
    if not device.startswith("cuda") or not torch.cuda.is_available():
        return ""
    try:
        a = torch.cuda.memory_allocated() / (1024 ** 3)
        r = torch.cuda.memory_reserved() / (1024 ** 3)
        return f"  [VRAM alloc={a:.2f}GB res={r:.2f}GB]"
    except Exception:
        return ""


# ============================================================================
# Data loading
# ============================================================================

def load_contrast_file(path: str, label: str) -> list:
    """
    Load a contrast-examples JSON file.  Validates that each entry has
    example_id and gold_answer.  The cell prompt keys depend on the
    contrast type and are validated later during prompt resolution.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{label} contrast file not found: {p}")

    log(f"[load] Reading {label} contrast examples from {p}")
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list) or len(data) == 0:
        raise ValueError(f"{label} contrast file must be a non-empty JSON list.")

    valid = []
    for i, ex in enumerate(data):
        if not isinstance(ex, dict):
            log(f"  WARN: {label} index {i}: not a dict, skipping")
            continue
        if "example_id" not in ex or "gold_answer" not in ex:
            log(f"  WARN: {label} index {i}: missing example_id or gold_answer, skipping")
            continue
        if not ex["gold_answer"]:
            log(f"  WARN: {label} index {i}: empty gold_answer, skipping")
            continue
        valid.append(ex)

    log(f"[load] {len(valid)}/{len(data)} {label} contrast examples validated")
    return valid


def load_dataset_index(path: str) -> dict:
    """Load full dataset.json, return dict keyed by example ID."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset file not found: {p}")

    log(f"[load] Reading dataset from {p}")
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)

    by_id = {}
    for ex in data:
        eid = ex.get("id") or ex.get("example_id")
        if eid:
            by_id[eid] = ex
    log(f"[load] {len(by_id)} examples indexed from dataset")
    return by_id


def resolve_prompts(
    examples: list,
    cell_baseline: str,
    cell_structured: str,
    dataset_index: dict | None,
    tokenizer=None,
) -> list:
    """
    For each contrast example, extract the two prompts needed for logit
    lens analysis.  First tries embedded cell dicts (e.g. cell_A.prompt),
    then falls back to dataset.json cells if available.

    If tokenizer is provided, cell dicts are materialised via
    materialise_prompt() to reconstruct the full runnable prompt
    (including EOS alignment padding).  If tokenizer is None, falls back
    to extracting the 'prompt' key directly (legacy behaviour).

    Returns list of dicts: {example_id, gold_answer, prompt_baseline, prompt_structured}
    Skips examples where prompts can't be resolved.
    """
    cell_bl_key = f"cell_{cell_baseline}"   # e.g. "cell_A" or "cell_B"
    cell_st_key = f"cell_{cell_structured}"  # e.g. "cell_C" or "cell_D"

    def _extract(cell_value):
        """Extract a runnable prompt string from a cell value."""
        if cell_value is None:
            return None
        if tokenizer is not None:
            return materialise_prompt(cell_value, tokenizer)
        # Legacy fallback: extract 'prompt' key from dict, or use string directly
        if isinstance(cell_value, str):
            return cell_value if cell_value else None
        if isinstance(cell_value, dict):
            p = cell_value.get("prompt")
            return p if (isinstance(p, str) and p) else None
        return None

    resolved = []
    for ex in examples:
        eid = ex["example_id"]
        gold = ex["gold_answer"]

        prompt_bl = None
        prompt_st = None

        # Try embedded cell dicts first
        if cell_bl_key in ex:
            prompt_bl = _extract(ex[cell_bl_key])
        if cell_st_key in ex:
            prompt_st = _extract(ex[cell_st_key])

        # Fallback to dataset.json
        if (prompt_bl is None or prompt_st is None) and dataset_index is not None:
            ds = dataset_index.get(eid)
            if ds and "cells" in ds:
                if prompt_bl is None:
                    prompt_bl = _extract(ds["cells"].get(cell_baseline))
                if prompt_st is None:
                    prompt_st = _extract(ds["cells"].get(cell_structured))

        if not prompt_bl:
            log(f"  WARN: {eid} — no prompt for Cell {cell_baseline}, skipping")
            continue
        if not prompt_st:
            log(f"  WARN: {eid} — no prompt for Cell {cell_structured}, skipping")
            continue

        resolved.append({
            "example_id": eid,
            "gold_answer": gold,
            "prompt_baseline": prompt_bl,
            "prompt_structured": prompt_st,
        })

    return resolved


# ============================================================================
# Model loading
# ============================================================================

def load_model(model_name: str, device: str):
    """Load Pythia model via TransformerLens HookedTransformer."""
    from transformer_lens import HookedTransformer

    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    log(f"[model] Loading {model_name} on {device} ({dtype}) ...")
    t0 = time.time()

    model = HookedTransformer.from_pretrained(
        model_name,
        device=device,
        dtype=dtype,
    )
    model.eval()

    log(f"[model] Loaded in {fmt_s(time.time() - t0)} | "
        f"n_layers={model.cfg.n_layers} d_model={model.cfg.d_model}"
        f"{cuda_mem(device)}")
    return model


# ============================================================================
# Token utilities
# ============================================================================

def get_gold_first_token_id(model, gold_answer: str) -> tuple:
    """
    Return (token_id, token_string) for the first token of the gold answer.
    Prepends a space to match how the model sees the answer after "A:" or
    "Answer:" in the prompt.  Consistent with scoring in patching scripts.
    """
    spaced = " " + gold_answer.strip()
    ids = model.tokenizer.encode(spaced, add_special_tokens=False)
    if len(ids) > 0:
        tid = ids[0]
        return tid, model.tokenizer.decode([tid]).strip()

    # Fallback without leading space
    ids = model.tokenizer.encode(gold_answer.strip(), add_special_tokens=False)
    if len(ids) > 0:
        tid = ids[0]
        return tid, model.tokenizer.decode([tid]).strip()

    return None, None


def check_token_alignment(model, prompt_a: str, prompt_b: str,
                          label_a: str, label_b: str, eid: str) -> bool:
    """
    Check that two prompts have identical token counts.  This is a hard
    prerequisite for valid comparison — if they differ, residual stream
    positions correspond to different input tokens, making layer-by-layer
    comparison meaningless.

    Returns True if aligned, False (with logged warning) if mismatched.
    """
    toks_a = model.to_tokens(prompt_a)
    toks_b = model.to_tokens(prompt_b)
    len_a = toks_a.shape[1]
    len_b = toks_b.shape[1]

    if len_a != len_b:
        log(f"  ALIGNMENT FAIL: {eid} — Cell {label_a}={len_a} tokens vs "
            f"Cell {label_b}={len_b} tokens (diff={abs(len_a - len_b)}) — SKIPPING")
        return False
    return True


# ============================================================================
# Core logit lens computation
# ============================================================================

def extract_layerwise_gold_scores(
    model, prompt: str, gold_token_id: int
) -> list:
    """
    Run model with activation caching, then for each layer project the
    residual stream at the FINAL token position through ln_final and the
    unembedding matrix to get vocabulary logits.

    Logit lens procedure at layer l:
        logits_l = Unembed( LayerNorm_final( resid_post_l[-1] ) )

    TransformerLens hook: "blocks.{l}.hook_resid_post"
    Applying ln_final before unembed is critical for meaningful results.

    Returns list of dicts: [{layer, gold_logit, gold_rank, is_top1,
                             top1_token, top1_logit}, ...]
    """
    tokens = model.to_tokens(prompt)  # [1, seq_len]

    with torch.no_grad():
        _, cache = model.run_with_cache(
            tokens,
            names_filter=lambda name: name.endswith("hook_resid_post"),
        )

    n_layers = model.cfg.n_layers
    W_U = model.unembed.W_U   # [d_model, d_vocab]
    b_U = model.unembed.b_U   # [d_vocab] or None

    results = []
    for layer_idx in range(n_layers):
        resid = cache[f"blocks.{layer_idx}.hook_resid_post"][0, -1, :]  # [d_model]

        # Apply final LayerNorm then project through unembedding
        resid_normed = model.ln_final(resid.unsqueeze(0)).squeeze(0)
        logits = resid_normed @ W_U  # [d_vocab]
        if b_U is not None:
            logits = logits + b_U

        # Gold token scoring
        gold_logit = logits[gold_token_id].item()
        sorted_indices = torch.argsort(logits, descending=True)
        rank_match = (sorted_indices == gold_token_id).nonzero(as_tuple=True)[0]
        gold_rank = rank_match.item() if rank_match.numel() > 0 else -1
        is_top1 = (gold_rank == 0)

        # Top-1 token for diagnostic context
        top1_id = sorted_indices[0].item()
        top1_token = model.tokenizer.decode([top1_id]).strip()
        top1_logit = logits[top1_id].item()

        results.append({
            "layer": layer_idx,
            "gold_logit": round(gold_logit, 4),
            "gold_rank": gold_rank,
            "is_top1": is_top1,
            "top1_token": top1_token,
            "top1_logit": round(top1_logit, 4),
        })

    del cache, tokens
    torch.cuda.empty_cache()
    gc.collect()

    return results


# ============================================================================
# Per-example analysis
# ============================================================================

def analyse_example(
    model,
    example_id: str,
    gold_answer: str,
    prompt_baseline: str,
    prompt_structured: str,
    label_baseline: str,
    label_structured: str,
) -> list:
    """
    Run logit lens on both conditions for one contrast example.
    Returns a flat list of row dicts for the per-example CSV.
    """
    gold_token_id, gold_token_str = get_gold_first_token_id(model, gold_answer)
    if gold_token_id is None:
        log(f"  WARN: {example_id} — could not tokenise "
            f"gold answer '{gold_answer}', skipping")
        return []

    rows = []
    for label, prompt in [
        (label_baseline, prompt_baseline),
        (label_structured, prompt_structured),
    ]:
        vlog(f"    {example_id} | {label}")
        layer_scores = extract_layerwise_gold_scores(model, prompt, gold_token_id)

        for ls in layer_scores:
            rows.append({
                "example_id": example_id,
                "condition": label,
                "layer": ls["layer"],
                "gold_answer": gold_answer,
                "gold_first_token": gold_token_str,
                "gold_token_id": gold_token_id,
                "gold_logit": ls["gold_logit"],
                "gold_rank": ls["gold_rank"],
                "is_top1": int(ls["is_top1"]),
                "top1_token": ls["top1_token"],
                "top1_logit": ls["top1_logit"],
            })

    return rows


# ============================================================================
# Summary aggregation
# ============================================================================

def build_summary(per_example_rows: list) -> list:
    """Aggregate per-example rows by (condition, layer)."""
    groups = defaultdict(list)
    for row in per_example_rows:
        key = (row["condition"], row["layer"])
        groups[key].append(row)

    summary = []
    for (condition, layer), rows in sorted(groups.items()):
        n = len(rows)
        logits = [r["gold_logit"] for r in rows]
        ranks = [r["gold_rank"] for r in rows]
        top1_count = sum(r["is_top1"] for r in rows)
        summary.append({
            "condition": condition,
            "layer": layer,
            "n_examples": n,
            "mean_gold_logit": round(float(np.mean(logits)), 4),
            "std_gold_logit": round(float(np.std(logits)), 4),
            "mean_gold_rank": round(float(np.mean(ranks)), 2),
            "median_gold_rank": int(np.median(ranks)),
            "top1_rate": round(top1_count / n, 4) if n > 0 else 0.0,
            "top1_count": top1_count,
        })
    return summary


def print_emergence(summary: list, cell_bl: str, cell_st: str) -> None:
    """Print emergence layers and a readable summary table."""
    label_bl = f"cell_{cell_bl}"
    label_st = f"cell_{cell_st}"

    by_cond = defaultdict(list)
    for row in summary:
        by_cond[row["condition"]].append(row)

    log(f"\n{'=' * 75}")
    log(f"LOGIT LENS SUMMARY — Cell {cell_bl} (direct) vs "
        f"Cell {cell_st} (structured)")
    log(f"{'=' * 75}")
    log(f"{'Layer':>6}  {'Cond':>8}  {'MeanLogit':>10}  {'StdLogit':>9}  "
        f"{'MeanRank':>9}  {'MedRank':>8}  {'Top1%':>7}")
    log(f"{'-' * 75}")

    # Print a selection of layers to keep output readable
    for row in summary:
        layer = row["layer"]
        n_layers = max(r["layer"] for r in summary) + 1
        if layer % 4 == 0 or layer <= 1 or layer >= n_layers - 3:
            log(f"{layer:>6}  {row['condition']:>8}  "
                f"{row['mean_gold_logit']:>10.3f}  "
                f"{row['std_gold_logit']:>9.3f}  "
                f"{row['mean_gold_rank']:>9.1f}  "
                f"{row['median_gold_rank']:>8}  "
                f"{row['top1_rate'] * 100:>6.1f}%")

    log(f"{'=' * 75}")

    # Emergence layer: first layer where top1_rate > 0.5
    for label in [label_bl, label_st]:
        rows = sorted(by_cond.get(label, []), key=lambda r: r["layer"])
        emergence = None
        for r in rows:
            if r["top1_rate"] > 0.5:
                emergence = r["layer"]
                break
        tag = "(direct)" if label == label_bl else "(structured)"
        if emergence is not None:
            log(f"  Emergence layer {tag}: layer {emergence} "
                f"(first with top1_rate > 50%)")
        else:
            log(f"  Emergence layer {tag}: never reaches 50% top-1 rate")
    log("")


# ============================================================================
# CSV I/O
# ============================================================================

PER_EXAMPLE_FIELDS = [
    "example_id", "condition", "layer",
    "gold_answer", "gold_first_token", "gold_token_id",
    "gold_logit", "gold_rank", "is_top1",
    "top1_token", "top1_logit",
]

SUMMARY_FIELDS = [
    "condition", "layer", "n_examples",
    "mean_gold_logit", "std_gold_logit",
    "mean_gold_rank", "median_gold_rank",
    "top1_rate", "top1_count",
]


def write_csv(rows: list, path: str, fieldnames: list) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    log(f"[output] Wrote {len(rows)} rows -> {p}")


# ============================================================================
# Plotting
# ============================================================================

# Display labels for conditions (used in legends and titles)
CONDITION_LABELS = {
    "cell_A": "Cell A (direct clean)",
    "cell_B": "Cell B (direct noisy)",
    "cell_C": "Cell C (structured clean)",
    "cell_D": "Cell D (structured noisy)",
}

# Line colours: direct = muted, structured = saturated
CONDITION_COLOURS = {
    "cell_A": "#7f8c8d",   # grey
    "cell_B": "#7f8c8d",   # grey
    "cell_C": "#2980b9",   # blue
    "cell_D": "#2980b9",   # blue
}

CONDITION_LINESTYLES = {
    "cell_A": "--",
    "cell_B": "--",
    "cell_C": "-",
    "cell_D": "-",
}


def _extract_series(summary_rows: list, condition: str, y_key: str):
    """Extract (layers, values) arrays for one condition from summary rows."""
    rows = sorted(
        [r for r in summary_rows if r["condition"] == condition],
        key=lambda r: r["layer"],
    )
    layers = [r["layer"] for r in rows]
    values = [r[y_key] for r in rows]
    return layers, values


def plot_top1_curve(summary_rows: list, output_path: str, title: str) -> None:
    """
    Plot top-1 rate vs layer for two conditions.
    X-axis: layer index.  Y-axis: proportion of contrast examples where the
    gold answer's first token is the top-1 prediction at that layer.
    """
    conditions = sorted(set(r["condition"] for r in summary_rows))
    if len(conditions) != 2:
        log(f"[plot] WARN: expected 2 conditions, got {conditions} — skipping plot")
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    for cond in conditions:
        layers, vals = _extract_series(summary_rows, cond, "top1_rate")
        ax.plot(
            layers, vals,
            label=CONDITION_LABELS.get(cond, cond),
            color=CONDITION_COLOURS.get(cond, None),
            linestyle=CONDITION_LINESTYLES.get(cond, "-"),
            linewidth=1.8,
        )

    ax.set_xlabel("Layer", fontsize=11)
    ax.set_ylabel("Top-1 Rate", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=9)
    ax.set_xlim(0, max(r["layer"] for r in summary_rows))
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log(f"[figure] Saved -> {output_path}")


def plot_logit_curve(summary_rows: list, output_path: str, title: str) -> None:
    """
    Plot mean gold-token logit vs layer for two conditions.
    X-axis: layer index.  Y-axis: mean logit assigned to the gold answer's
    first token across contrast examples.
    """
    conditions = sorted(set(r["condition"] for r in summary_rows))
    if len(conditions) != 2:
        log(f"[plot] WARN: expected 2 conditions, got {conditions} — skipping plot")
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    for cond in conditions:
        layers, vals = _extract_series(summary_rows, cond, "mean_gold_logit")
        ax.plot(
            layers, vals,
            label=CONDITION_LABELS.get(cond, cond),
            color=CONDITION_COLOURS.get(cond, None),
            linestyle=CONDITION_LINESTYLES.get(cond, "-"),
            linewidth=1.8,
        )

    ax.set_xlabel("Layer", fontsize=11)
    ax.set_ylabel("Mean Gold-Token Logit", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=9)
    ax.set_xlim(0, max(r["layer"] for r in summary_rows))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log(f"[figure] Saved -> {output_path}")


# ============================================================================
# Run one analysis pass (clean or noisy)
# ============================================================================

def run_pass(
    model,
    examples: list,
    cell_baseline: str,
    cell_structured: str,
    dataset_index: dict | None,
    outdir: str,
    figdir: str,
    file_prefix: str,
    max_examples: int | None,
    device: str,
) -> None:
    """
    Execute logit lens analysis for one contrast type (clean or noisy).

    Parameters
    ----------
    examples       — contrast examples loaded from the appropriate file
    cell_baseline  — "A" for clean, "B" for noisy
    cell_structured — "C" for clean, "D" for noisy
    file_prefix    — "" for default A/C, "noisy_" for B/D outputs
    figdir         — directory for figure output (e.g. results/figures)
    """
    tag = "clean" if cell_baseline == "A" else "noisy"

    # Resolve prompts for each example
    resolved = resolve_prompts(examples, cell_baseline, cell_structured, dataset_index,
                               tokenizer=model.tokenizer)
    if max_examples is not None:
        resolved = resolved[:max_examples]

    if not resolved:
        log(f"[{tag}] No examples with resolved prompts — skipping {tag} pass")
        return

    log(f"\n{'—' * 70}")
    log(f"  {tag.upper()} CONTRAST: Cell {cell_baseline} (direct) vs "
        f"Cell {cell_structured} (structured)")
    log(f"  {len(resolved)} examples with resolved prompts")
    log(f"{'—' * 70}\n")

    all_rows = []
    skipped_alignment = 0
    skipped_error = 0

    for i, ex in enumerate(resolved):
        eid = ex["example_id"]
        gold = ex["gold_answer"]
        t0 = time.time()

        # ---- Token alignment assertion (HARD prerequisite) ----
        aligned = check_token_alignment(
            model,
            ex["prompt_baseline"], ex["prompt_structured"],
            cell_baseline, cell_structured, eid,
        )
        if not aligned:
            skipped_alignment += 1
            continue

        try:
            rows = analyse_example(
                model=model,
                example_id=eid,
                gold_answer=gold,
                prompt_baseline=ex["prompt_baseline"],
                prompt_structured=ex["prompt_structured"],
                label_baseline=f"cell_{cell_baseline}",
                label_structured=f"cell_{cell_structured}",
            )
            all_rows.extend(rows)
        except Exception as e:
            log(f"  ERROR: {eid}: {e}")
            skipped_error += 1
            gc.collect()
            torch.cuda.empty_cache()
            continue

        elapsed = time.time() - t0
        n_done = i + 1 - skipped_alignment - skipped_error
        if n_done % 5 == 0 or i == 0 or i == len(resolved) - 1:
            log(f"  [{n_done}/{len(resolved)}] {eid}  "
                f"gold=\"{gold}\"  ({fmt_s(elapsed)}){cuda_mem(device)}")

        torch.cuda.empty_cache()
        gc.collect()

    n_ok = len(resolved) - skipped_alignment - skipped_error
    log(f"\n[{tag}] Processed {n_ok} examples "
        f"(skipped: {skipped_alignment} alignment, {skipped_error} errors)")

    if not all_rows:
        log(f"[{tag}] No results — nothing to write")
        return

    # Write per-example CSV
    per_ex_path = str(Path(outdir) / f"{file_prefix}logit_lens_per_example.csv")
    write_csv(all_rows, per_ex_path, PER_EXAMPLE_FIELDS)

    # Write summary CSV
    summary = build_summary(all_rows)
    sum_path = str(Path(outdir) / f"{file_prefix}logit_lens_summary.csv")
    write_csv(summary, sum_path, SUMMARY_FIELDS)

    # Print summary to console
    print_emergence(summary, cell_baseline, cell_structured)

    # ---- Generate figures ----
    tag_title = "Clean" if cell_baseline == "A" else "Noisy"

    plot_top1_curve(
        summary,
        str(Path(figdir) / f"{file_prefix}logit_lens_top1.png"),
        f"Logit Lens: Top-1 Rate by Layer ({tag_title} Contrast)",
    )
    plot_logit_curve(
        summary,
        str(Path(figdir) / f"{file_prefix}logit_lens_logit.png"),
        f"Logit Lens: Mean Gold-Token Logit by Layer ({tag_title} Contrast)",
    )


# ============================================================================
# Main
# ============================================================================

def _model_slug(model_name: str) -> str:
    """'EleutherAI/pythia-2.8b' -> 'pythia-2.8b', 'Qwen/Qwen2.5-3B' -> 'qwen2.5-3b'"""
    return model_name.split("/")[-1].lower()


def main():
    global VERBOSE

    parser = argparse.ArgumentParser(
        description="Phase 4a: Logit Lens Analysis (secondary diagnostic)"
    )
    parser.add_argument(
        "--clean-contrast", type=str, default=None,
        help="Path to clean contrast examples JSON (A wrong, C correct) "
             "(default: dataset/processed/<model-slug>/contrast_examples.json)"
    )
    parser.add_argument(
        "--noisy-contrast", type=str, default=None,
        help="Path to noisy contrast examples JSON (B wrong, D correct) "
             "(default: dataset/processed/<model-slug>/noisy_contrast_examples.json)"
    )
    parser.add_argument(
        "--contrast-file", type=str, default=None,
        help="Path to the source/donor contrast examples JSON. Overrides contrast routing."
    )
    parser.add_argument(
        "--dataset", type=str, default=None,
        help="Path to full dataset JSON (fallback for prompt lookup) "
             "(default: dataset/processed/<model-slug>/dataset.json)"
    )
    parser.add_argument(
        "--outdir", type=str, default=None,
        help="Output directory for CSV files "
             "(default: results/phase_4a_logit_lens/<model-slug>/)"
    )
    parser.add_argument(
        "--figdir", type=str, default=None,
        help="Output directory for figure PNG files "
             "(default: figures/phase_4a_logit_lens/<model-slug>/)"
    )
    parser.add_argument(
        "--model", type=str, default="EleutherAI/pythia-2.8b",
        help="HuggingFace model name for TransformerLens"
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device: 'cuda' or 'cpu'"
    )
    parser.add_argument(
        "--source-cell", type=str, default="A",
        choices=["A", "B", "C", "D", "E"],
        help="Baseline/source cell to analyse (default: A)"
    )
    parser.add_argument(
        "--donor-cell", type=str, default="C",
        choices=["A", "B", "C", "D", "E"],
        help="Structured/donor cell to analyse (default: C)"
    )
    parser.add_argument(
        "--output-prefix", type=str, default=None,
        help="Optional filename prefix override. Defaults come from source/donor contrast routing."
    )

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--noisy", action="store_true",
        help="Run ONLY noisy contrast (Cell B vs D). Requires --noisy-contrast."
    )
    mode.add_argument(
        "--include-noisy", action="store_true",
        help="Run BOTH clean and noisy contrast in one invocation."
    )

    parser.add_argument(
        "--max-examples", type=int, default=None,
        help="Cap number of examples per pass (for quick debugging)"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable verbose per-token logging"
    )
    args = parser.parse_args()
    VERBOSE = args.verbose

    # ---- Resolve model-namespaced defaults ----
    slug = _model_slug(args.model)
    source_cell = args.source_cell.upper()
    donor_cell = args.donor_cell.upper()

    clean_contrast = args.clean_contrast or f"dataset/processed/{slug}/contrast_examples.json"
    noisy_contrast = args.noisy_contrast or f"dataset/processed/{slug}/noisy_contrast_examples.json"
    dataset_path   = args.dataset        or f"dataset/processed/{slug}/dataset.json"
    outdir = args.outdir or f"results/phase_4a_logit_lens/{slug}"
    figdir = args.figdir or f"figures/phase_4a_logit_lens/{slug}"

    # Determine which passes to run
    custom_cells = (source_cell, donor_cell) != ("A", "C")
    run_custom = custom_cells or args.contrast_file is not None or args.output_prefix is not None
    run_clean = (not args.noisy) and not run_custom
    run_noisy = (args.noisy or args.include_noisy) and not run_custom
    custom_contrast = args.contrast_file or contrast_path_for(slug, source_cell, donor_cell)
    custom_label = f"{source_cell.lower()}_{donor_cell.lower()}"
    custom_file_prefix = model_file_prefix(slug, output_prefix_for(source_cell, donor_cell, args.output_prefix))

    # ---- Ensure output directories exist ----
    Path(outdir).mkdir(parents=True, exist_ok=True)
    Path(figdir).mkdir(parents=True, exist_ok=True)

    # ---- Banner ----
    log("=" * 70)
    log("Phase 4a: Logit Lens Analysis (Secondary Diagnostic)")
    log("  NOTE: Logit lens is correlational, NOT primary causal evidence.")
    log("  Activation patching (Phases 3a-3c) is the primary method.")
    log("=" * 70)
    log(f"  Model:           {args.model}")
    log(f"  Clean contrast:  {clean_contrast}")
    log(f"  Noisy contrast:  {noisy_contrast}")
    log(f"  Dataset:         {dataset_path}")
    log(f"  Source cell:     {source_cell}")
    log(f"  Donor cell:      {donor_cell}")
    log(f"  Run clean:       {run_clean}")
    log(f"  Run noisy:       {run_noisy}")
    log(f"  Run custom:      {run_custom}")
    log(f"  Output dir:      {outdir}")
    log(f"  Figure dir:      {figdir}")
    log(f"  Max examples:    {args.max_examples or 'all'}")
    log("")

    # ---- Load dataset (optional fallback for prompt lookup) ----
    dataset_index = None
    ds_path = Path(dataset_path)
    if ds_path.exists():
        dataset_index = load_dataset_index(str(ds_path))
    elif run_noisy:
        log(f"[WARN] Dataset not found at {dataset_path} — noisy prompt "
            f"fallback will not be available")

    # ---- Load model ----
    model = load_model(args.model, args.device)

    # ---- Explicit source/donor pass ----
    if run_custom:
        try:
            custom_examples = load_contrast_file(custom_contrast, custom_label)
        except FileNotFoundError as e:
            log(f"[ERROR] {e}")
            sys.exit(1)

        run_pass(
            model=model,
            examples=custom_examples,
            cell_baseline=source_cell,
            cell_structured=donor_cell,
            dataset_index=dataset_index,
            outdir=outdir,
            figdir=figdir,
            file_prefix=custom_file_prefix,
            max_examples=args.max_examples,
            device=args.device,
        )

    # ---- Clean pass ----
    if run_clean:
        try:
            clean_examples = load_contrast_file(clean_contrast, "clean")
        except FileNotFoundError as e:
            log(f"[ERROR] {e}")
            if not run_noisy:
                sys.exit(1)
            clean_examples = []

        if clean_examples:
            run_pass(
                model=model,
                examples=clean_examples,
                cell_baseline="A",
                cell_structured="C",
                dataset_index=dataset_index,
                outdir=outdir,
                figdir=figdir,
                file_prefix=model_file_prefix(slug),
                max_examples=args.max_examples,
                device=args.device,
            )

    # ---- Noisy pass ----
    if run_noisy:
        try:
            noisy_examples = load_contrast_file(noisy_contrast, "noisy")
        except FileNotFoundError as e:
            log(f"[ERROR] {e}")
            log(f"")
            log(f"  To run noisy logit lens, you first need a noisy contrast file.")
            log(f"  Noisy contrast = examples where Cell B is WRONG and Cell D is CORRECT.")
            log(f"  This is a DIFFERENT example set from clean contrast (A wrong, C correct).")
            log(f"")
            log(f"  Generate it from your Phase 2 behavioural results, e.g.:")
            log(f"    noisy_contrast = [ex for ex in results")
            log(f"                      if not ex['cell_B']['correct']")
            log(f"                      and ex['cell_D']['correct']]")
            log(f"")
            log(f"  Save as: {noisy_contrast}")
            if not run_clean:
                sys.exit(1)
            noisy_examples = []

        if noisy_examples:
            run_pass(
                model=model,
                examples=noisy_examples,
                cell_baseline="B",
                cell_structured="D",
                dataset_index=dataset_index,
                outdir=outdir,
                figdir=figdir,
                file_prefix=model_file_prefix(slug, "noisy_"),
                max_examples=args.max_examples,
                device=args.device,
            )

    # ---- Done ----
    log(f"\nPhase 4a COMPLETE.")
    log(f"  CSVs:    {outdir}/")
    log(f"  Figures: {figdir}/")
    log(f"  Next: review figures, then proceed to Phase 4b (attention heatmaps).")
    log(f"  Remember: logit lens is secondary — interpret alongside")
    log(f"  activation patching results from Phases 3a-3c.")


if __name__ == "__main__":
    main()
