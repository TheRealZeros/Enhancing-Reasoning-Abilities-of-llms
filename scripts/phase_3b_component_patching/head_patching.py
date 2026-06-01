#!/usr/bin/env python3
"""
Phase 3b (Step 2 of 2): Targeted Head-Level Activation Patching
================================================================
Drills into the attention output at selected layers by patching individual
attention heads from the structured run (Cell C) into the direct run
(Cell A). This follows the broad component decomposition (Step 1), which
identified layers where the combined attention output carries a positive
causal mediation effect.

For each contrast example and each head h at layer ℓ, the script patches
the per-head attention result from the structured cache into the direct
run and measures:

    Δℓ,h = score(patched head h at layer ℓ) − score(baseline)

where "score" is the logit (default) or probability for the gold answer's
first token at the final sequence position.

TransformerLens hook used:
    blocks.{layer}.attn.hook_z  — shape (batch, seq, n_heads, d_head)
    This is the per-head attention output BEFORE the output projection W_O.
    Patching at this level is the standard approach for head-level causal
    attribution (cf. Wang et al. 2022, IOI circuit).

The hook replaces only head h at the FINAL sequence position, matching
the patch scope used in Phase 3a (layer-level) and Phase 3b Step 1
(component-level).

Usage:
    python scripts/phase_3b_component_patching/head_patching.py --layers 30 31
    python scripts/phase_3b_component_patching/head_patching.py --layers 30 31 --verbose

Outputs:
    results/phase_3b_component_patching/head_patch_results.csv   – one row per (example, layer, head)
    results/phase_3b_component_patching/head_patch_summary.csv   – per-(layer, head) aggregated Δℓ,h
    figures/phase_3b_component_patching/head_patch_heatmap.png    – thesis figure: head-level heatmap

Methodological precedents:
    Wang et al. 2022 (IOI circuit), Meng et al. 2022 (ROME/causal tracing),
    Elhage et al. 2021 (transformer circuits framework).
"""

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless servers
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


# ---------------------------------------------------------------------------
# Prompt materialisation (shared utility — identical to build_dataset.py)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Hook template for per-head attention result
# ---------------------------------------------------------------------------

# TransformerLens stores the per-head attention output (before the output
# projection W_O) at this hook point. Shape: (batch, seq, n_heads, d_head).
# This is "hook_z" in TransformerLens, NOT "hook_result" (which, if it
# exists, is the post-W_O combined output and has shape (batch, seq, d_model)).
# Patching hook_z is the standard approach for head-level attribution
# (cf. Wang et al. 2022, IOI circuit).
HEAD_HOOK_TEMPLATE = "blocks.{layer}.attn.hook_z"


# ---------------------------------------------------------------------------
# Logging and VRAM utilities (shared with Phase 3a / 3b Step 1)
# ---------------------------------------------------------------------------

def log(msg: str) -> None:
    """Print a message immediately."""
    print(msg, flush=True)


def format_seconds(seconds: float) -> str:
    """Format elapsed seconds in a compact human-readable way."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes}m {secs:.1f}s"


def get_cuda_mem_string(device: str) -> str:
    """Return current CUDA memory stats if available."""
    if not device.startswith("cuda") or not torch.cuda.is_available():
        return "cuda_mem=n/a"

    try:
        alloc = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        max_alloc = torch.cuda.max_memory_allocated() / (1024 ** 3)
        return (
            f"cuda_mem alloc={alloc:.2f}GB "
            f"reserved={reserved:.2f}GB "
            f"max_alloc={max_alloc:.2f}GB"
        )
    except Exception as e:
        return f"cuda_mem=unavailable ({e})"


def reset_cuda_peak_memory_stats(device: str) -> None:
    """Reset CUDA peak memory stats if on GPU."""
    if device.startswith("cuda") and torch.cuda.is_available():
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Contrast example loading and validation
# ---------------------------------------------------------------------------

REQUIRED_KEYS = {"example_id", "domain", "gold_answer", "cell_A", "cell_C"}
REQUIRED_CELL_KEYS = {"prompt"}


def load_contrast_examples(path: str) -> list[dict]:
    """Load the contrast examples JSON produced by Phase 2."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Contrast file not found: {path}")

    t0 = time.time()
    log(f"[load] Reading contrast examples from {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list) or len(data) == 0:
        raise ValueError("Contrast file must be a non-empty JSON list.")

    valid = []
    for i, ex in enumerate(data):
        ok, reason = validate_contrast_example(ex, i)
        if ok:
            valid.append(ex)
        else:
            log(f"[load] WARNING: skipping index {i}: {reason}")

    elapsed = time.time() - t0
    log(f"[load] {len(valid)}/{len(data)} contrast examples validated from {path} in {format_seconds(elapsed)}")
    return valid


def validate_contrast_example(ex: dict, idx: int) -> tuple[bool, str]:
    """Check that a contrast example has all required fields."""
    if not isinstance(ex, dict):
        return False, f"index {idx} is not a dict"

    missing = REQUIRED_KEYS - set(ex.keys())
    if missing:
        return False, f"missing top-level keys: {missing}"

    for cell_name in ("cell_A", "cell_C"):
        cell = ex[cell_name]
        if not isinstance(cell, dict):
            return False, f"{cell_name} is not a dict"
        missing_cell = REQUIRED_CELL_KEYS - set(cell.keys())
        if missing_cell:
            return False, f"{cell_name} missing keys: {missing_cell}"
        if not isinstance(cell["prompt"], str) or len(cell["prompt"]) == 0:
            return False, f"{cell_name}.prompt is empty or not a string"

    if not isinstance(ex["gold_answer"], str) or len(ex["gold_answer"]) == 0:
        return False, "gold_answer is empty or not a string"

    return True, ""


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_name: str, device: str):
    """Load model via TransformerLens HookedTransformer in float16."""
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

    elapsed = time.time() - t0
    log(
        f"[model] Loaded in {format_seconds(elapsed)} | "
        f"n_layers={model.cfg.n_layers} "
        f"n_heads={model.cfg.n_heads} "
        f"d_model={model.cfg.d_model} | "
        f"{get_cuda_mem_string(device)}"
    )
    return model


# ---------------------------------------------------------------------------
# Token-level utilities (identical to Phase 3a / 3b Step 1)
# ---------------------------------------------------------------------------

def get_target_token_id(model, gold_answer: str) -> tuple[int, str]:
    """
    Return the FIRST token ID of the gold answer (with leading space).
    """
    spaced = " " + gold_answer
    token_ids = model.tokenizer.encode(spaced, add_special_tokens=False)

    if len(token_ids) == 0:
        raise ValueError(f"Gold answer '{gold_answer}' tokenises to zero tokens")

    first_id = token_ids[0]
    first_str = model.tokenizer.decode([first_id])
    return first_id, first_str


def get_score_for_token(
    logits: torch.Tensor,
    token_id: int,
    metric: str,
) -> float:
    """
    Extract the score for a specific token at the LAST sequence position.
    """
    last_logits = logits[0, -1, :]

    if metric == "logit":
        return last_logits[token_id].item()
    elif metric == "prob":
        probs = torch.softmax(last_logits.float(), dim=-1)
        return probs[token_id].item()
    else:
        raise ValueError(f"Unknown metric: {metric}")


# ---------------------------------------------------------------------------
# Structured run (Cell C) — cache activations
# ---------------------------------------------------------------------------

def run_structured_with_cache(
    model,
    tokens: torch.Tensor,
    verbose: bool = False,
    device: str = "cpu",
) -> tuple[torch.Tensor, dict]:
    """
    Run the structured prompt (Cell C) through the model and cache all activations.
    """
    if verbose:
        log(f"    [structured] Starting cached forward pass | seq_len={tokens.shape[1]} | {get_cuda_mem_string(device)}")
    t0 = time.time()

    with torch.no_grad():
        logits, cache = model.run_with_cache(tokens)

    elapsed = time.time() - t0
    if verbose:
        log(f"    [structured] Done in {format_seconds(elapsed)} | {get_cuda_mem_string(device)}")

    return logits, cache


# ---------------------------------------------------------------------------
# Direct baseline run (Cell A) — no hooks, just forward pass
# ---------------------------------------------------------------------------

def run_direct_baseline(
    model,
    tokens: torch.Tensor,
    verbose: bool = False,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Run the direct prompt (Cell A) as a clean forward pass with no hooks.
    """
    if verbose:
        log(f"    [baseline] Starting direct forward pass | seq_len={tokens.shape[1]} | {get_cuda_mem_string(device)}")
    t0 = time.time()

    with torch.no_grad():
        logits = model(tokens)

    elapsed = time.time() - t0
    if verbose:
        log(f"    [baseline] Done in {format_seconds(elapsed)} | {get_cuda_mem_string(device)}")

    return logits


# ---------------------------------------------------------------------------
# Hook construction — single-head patching
# ---------------------------------------------------------------------------

def make_head_patch_hook(cached_activation: torch.Tensor, head_idx: int):
    """
    Create a TransformerLens hook function that REPLACES a single attention
    head's output at the FINAL sequence position only.

    cached_activation: shape (batch, seq, n_heads, d_head) from the
        structured run's blocks.{layer}.attn.hook_result cache entry.
    head_idx: which head to patch (0-indexed).

    The hook receives the full hook_result tensor during the direct run
    and overwrites only [batch, final_pos, head_idx, :] with the
    structured run's value.
    """
    def hook_fn(activation, hook):
        # activation shape: (batch, seq, n_heads, d_head)
        patched = activation.clone()
        patched[:, -1, head_idx, :] = cached_activation[:, -1, head_idx, :]
        return patched
    return hook_fn


# ---------------------------------------------------------------------------
# Head sweep for one contrast example
# ---------------------------------------------------------------------------

def run_head_sweep_for_example(
    model,
    example: dict,
    layers: list[int],
    metric: str,
    device: str,
    verbose: bool,
    head_log_interval: int = 4,
) -> list[dict]:
    """
    For one contrast example, patch each attention head at each selected
    layer and record the causal mediation effect Δℓ,h.
    """
    example_t0 = time.time()

    ex_id = example["example_id"]
    domain = example["domain"]
    gold = example["gold_answer"]
    prompt_a = materialise_prompt(example["cell_A"], model.tokenizer)
    prompt_c = materialise_prompt(example["cell_C"], model.tokenizer)

    n_heads = model.cfg.n_heads
    n_layers = len(layers)
    total_patches = n_layers * n_heads

    log(f"  [example:{ex_id}] Stage 1/5: tokenising prompts")

    # ---- Stage 1: Tokenise and verify alignment ----
    token_t0 = time.time()
    tokens_a = model.tokenizer.encode(prompt_a, return_tensors="pt").to(device)
    tokens_c = model.tokenizer.encode(prompt_c, return_tensors="pt").to(device)
    len_a = tokens_a.shape[1]
    len_c = tokens_c.shape[1]
    token_elapsed = time.time() - token_t0
    log(f"  [example:{ex_id}] Tokenisation done in {format_seconds(token_elapsed)} | len_A={len_a} len_C={len_c}")

    if len_a != len_c:
        reason = f"token misalignment: Cell A={len_a}, Cell C={len_c}"
        log(f"  [example:{ex_id}] SKIP | {reason}")
        return [_make_skip_row(ex_id, domain, gold, metric, len_a, len_c, reason)]

    # ---- Stage 2: Resolve gold token ----
    log(f"  [example:{ex_id}] Stage 2/5: resolving gold token")
    try:
        gold_token_id, gold_token_str = get_target_token_id(model, gold)
    except ValueError as e:
        reason = f"tokenisation error: {e}"
        log(f"  [example:{ex_id}] SKIP | {reason}")
        return [_make_skip_row(ex_id, domain, gold, metric, len_a, len_c, reason)]

    gold_token_count = len(model.tokenizer.encode(" " + gold, add_special_tokens=False))
    log(
        f"  [example:{ex_id}] Gold token resolved | "
        f"gold='{gold}' first_token='{gold_token_str}' id={gold_token_id} token_count={gold_token_count}"
    )

    # ---- Stage 3: Structured cached run ----
    log(f"  [example:{ex_id}] Stage 3/5: structured cached run")
    logits_c, cache = run_structured_with_cache(
        model=model,
        tokens=tokens_c,
        verbose=verbose,
        device=device,
    )

    # ---- Stage 4: Direct baseline run ----
    log(f"  [example:{ex_id}] Stage 4/5: direct baseline run")
    logits_a = run_direct_baseline(
        model=model,
        tokens=tokens_a,
        verbose=verbose,
        device=device,
    )
    baseline_score = get_score_for_token(logits_a, gold_token_id, metric)

    # ---- Stage 5: Head sweep ----
    log(f"  [example:{ex_id}] Stage 5/5: head sweep starting ({total_patches} patches: {n_layers} layers × {n_heads} heads)")

    # Validate that the expected hook keys exist in the cache before sweeping.
    # This catches TransformerLens version mismatches early with a clear message.
    first_hook = HEAD_HOOK_TEMPLATE.format(layer=layers[0])
    if first_hook not in cache.cache_dict:
        # List available attention-related keys to help diagnose
        attn_keys = sorted(k for k in cache.cache_dict.keys() if "attn" in k)
        log(f"  [example:{ex_id}] ERROR: cache key '{first_hook}' not found.")
        log(f"  Available attention-related cache keys (first 15):")
        for k in attn_keys[:15]:
            log(f"    {k}  shape={cache.cache_dict[k].shape}")
        reason = f"cache key '{first_hook}' not found — check TransformerLens hook naming"
        log(f"  [example:{ex_id}] SKIP | {reason}")
        del logits_c, cache, logits_a
        return [_make_skip_row(ex_id, domain, gold, metric, len_a, len_c, reason)]

    rows = []
    patch_times = []
    patch_count = 0

    sweep_t0 = time.time()

    for layer in layers:
        hook_name = HEAD_HOOK_TEMPLATE.format(layer=layer)
        cached_act = cache[hook_name]

        for head in range(n_heads):
            patch_t0 = time.time()
            patch_count += 1

            should_log = (
                verbose
                or patch_count == 1
                or patch_count % head_log_interval == 0
                or patch_count == total_patches
            )

            if should_log:
                log(
                    f"    [{patch_count:03d}/{total_patches:03d}] "
                    f"layer={layer} head={head:02d} | "
                    f"starting | {get_cuda_mem_string(device)}"
                )

            hook_fn = make_head_patch_hook(cached_act, head)

            with torch.no_grad():
                patched_logits = model.run_with_hooks(
                    tokens_a,
                    fwd_hooks=[(hook_name, hook_fn)],
                )

            patched_score = get_score_for_token(patched_logits, gold_token_id, metric)
            delta = patched_score - baseline_score

            patch_elapsed = time.time() - patch_t0
            patch_times.append(patch_elapsed)

            rows.append({
                "example_id": ex_id,
                "domain": domain,
                "layer": layer,
                "head": head,
                "hook_name": hook_name,
                "metric": metric,
                "gold_answer": gold,
                "gold_token_id": gold_token_id,
                "gold_token_str": gold_token_str,
                "gold_token_count": gold_token_count,
                "direct_token_count": len_a,
                "structured_token_count": len_c,
                "baseline_score": baseline_score,
                "patched_score": patched_score,
                "delta": delta,
                "valid_example": True,
                "skip_reason": "",
            })

            if should_log:
                log(
                    f"    [{patch_count:03d}/{total_patches:03d}] "
                    f"layer={layer} head={head:02d} | "
                    f"done in {format_seconds(patch_elapsed)} | "
                    f"patched={patched_score:.6f} delta={delta:+.6f} | "
                    f"{get_cuda_mem_string(device)}"
                )

    sweep_elapsed = time.time() - sweep_t0
    avg_patch_time = sum(patch_times) / len(patch_times) if patch_times else 0.0

    log(
        f"  [example:{ex_id}] Head sweep complete in {format_seconds(sweep_elapsed)} | "
        f"avg_per_patch={format_seconds(avg_patch_time)}"
    )

    # Free memory
    del logits_c, cache, logits_a, patched_logits

    total_elapsed = time.time() - example_t0
    log(f"  [example:{ex_id}] COMPLETE in {format_seconds(total_elapsed)}")
    return rows


def _make_skip_row(
    ex_id: str,
    domain: str,
    gold: str,
    metric: str,
    len_a: int,
    len_c: int,
    reason: str,
) -> dict:
    """Create a single skip-row for an invalid example."""
    return {
        "example_id": ex_id,
        "domain": domain,
        "layer": -1,
        "head": -1,
        "hook_name": "",
        "metric": metric,
        "gold_answer": gold,
        "gold_token_id": -1,
        "gold_token_str": "",
        "gold_token_count": -1,
        "direct_token_count": len_a,
        "structured_token_count": len_c,
        "baseline_score": float("nan"),
        "patched_score": float("nan"),
        "delta": float("nan"),
        "valid_example": False,
        "skip_reason": reason,
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_head_results(
    df: pd.DataFrame,
    metric: str,
) -> pd.DataFrame:
    """
    Compute per-(layer, head) mean and std of Δℓ,h across all valid
    contrast examples.
    """
    valid = df[df["valid_example"]].copy()
    if len(valid) == 0:
        log("[aggregate] WARNING: no valid examples to aggregate")
        return pd.DataFrame()

    t0 = time.time()
    log(f"[aggregate] Aggregating {len(valid)} valid rows")

    summary = (
        valid
        .groupby(["layer", "head"])
        .agg(
            mean_delta=("delta", "mean"),
            std_delta=("delta", "std"),
            n_examples=("delta", "size"),
        )
        .reset_index()
    )

    summary["hook_name"] = summary["layer"].apply(
        lambda l: HEAD_HOOK_TEMPLATE.format(layer=int(l))
    )
    summary["metric"] = metric

    summary = summary.sort_values(["layer", "head"]).reset_index(drop=True)
    summary = summary[[
        "layer", "head", "hook_name", "metric",
        "mean_delta", "std_delta", "n_examples",
    ]]

    elapsed = time.time() - t0
    log(f"[aggregate] Done in {format_seconds(elapsed)}")
    return summary


# ---------------------------------------------------------------------------
# Plotting — head-level heatmap
# ---------------------------------------------------------------------------

def plot_head_heatmap(
    summary_df: pd.DataFrame,
    output_path: str,
    metric: str,
    n_examples: int,
):
    """
    Plot a heatmap of mean Δℓ,h with heads on the x-axis and layers on
    the y-axis.
    """
    if summary_df.empty:
        log("[plot] WARNING: no data to plot")
        return

    t0 = time.time()
    log(f"[plot] Generating head-level heatmap at {output_path}")

    # Pivot to (layer × head) matrix
    pivot = summary_df.pivot(index="layer", columns="head", values="mean_delta")

    # Sort rows (layers) in ascending order
    pivot = pivot.sort_index(ascending=True)
    # Sort columns (heads) numerically
    pivot = pivot[sorted(pivot.columns)]

    data = pivot.values
    layers = list(pivot.index)
    heads = list(pivot.columns)

    n_rows = len(layers)
    n_cols = len(heads)

    fig, ax = plt.subplots(figsize=(max(8, n_cols * 0.55), max(3, n_rows * 1.2)))

    # Diverging colour scale centred at 0
    vmax = max(abs(np.nanmin(data)), abs(np.nanmax(data)))
    if vmax == 0:
        vmax = 1.0
    vmin = -vmax

    im = ax.imshow(
        data,
        aspect="auto",
        cmap="RdBu_r",
        vmin=vmin,
        vmax=vmax,
    )

    # Annotate cells with numeric values (only if reasonably few cells)
    if n_rows * n_cols <= 128:
        for i in range(n_rows):
            for j in range(n_cols):
                val = data[i, j]
                if not np.isnan(val):
                    text_colour = "white" if abs(val) > 0.6 * vmax else "black"
                    fontsize = 7 if n_cols > 20 else 8
                    ax.text(
                        j, i, f"{val:+.2f}",
                        ha="center", va="center",
                        fontsize=fontsize,
                        color=text_colour,
                    )

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels([str(h) for h in heads], fontsize=8)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels([str(l) for l in layers], fontsize=10)
    ax.set_xlabel("Attention Head", fontsize=11)
    ax.set_ylabel("Layer", fontsize=11)

    metric_label = "logit" if metric == "logit" else "probability"
    ax.set_title(
        f"Head-Level Causal Mediation Effect ({metric_label}, n={n_examples})",
        fontsize=12,
    )

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(f"Mean Δℓ,h ({metric_label})", fontsize=10)

    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    elapsed = time.time() - t0
    log(f"[plot] Saved {output_path} in {format_seconds(elapsed)}")


# ---------------------------------------------------------------------------
# CLI and main
# ---------------------------------------------------------------------------

def _model_slug(model_name: str) -> str:
    return model_name.split("/")[-1].lower()


def main():
    parser = argparse.ArgumentParser(
        description="Phase 3b Step 2: Targeted head-level activation patching at selected layers"
    )
    parser.add_argument("--contrast-file", type=str, default=None,
                        help="Path to contrast_examples.json from Phase 2 "
                             "(default: dataset/processed/<model-slug>/contrast_examples.json)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory for output CSV files "
                             "(default: results/phase_3b_component_patching/<model-slug>/)")
    parser.add_argument("--figure-dir", type=str, default=None,
                        help="Directory for output figure files "
                             "(default: figures/phase_3b_component_patching/<model-slug>/)")
    parser.add_argument("--model", type=str, default="EleutherAI/pythia-2.8b",
                        help="HuggingFace model name for HookedTransformer")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda or cpu)")
    parser.add_argument("--max-examples", type=int, default=None,
                        help="Limit to first N contrast examples (for debugging)")
    parser.add_argument("--layers", type=int, nargs="+",
                        default=[30, 31],
                        help="Layers to sweep heads at (from Phase 3b Step 1 results)")
    parser.add_argument("--metric", type=str, default="logit",
                        choices=["logit", "prob"],
                        help="Score metric: 'logit' (default) or 'prob'")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-patch details and stage timings")
    parser.add_argument("--head-log-interval", type=int, default=8,
                        help="Print head progress every N patches (default: 8)")
    args = parser.parse_args()

    slug = _model_slug(args.model)
    contrast_file = args.contrast_file or f"dataset/processed/{slug}/contrast_examples.json"
    out_dir_path  = args.output_dir   or f"results/phase_3b_component_patching/{slug}"
    fig_dir_path  = args.figure_dir   or f"figures/phase_3b_component_patching/{slug}"

    overall_t0 = time.time()

    out_dir = Path(out_dir_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = Path(fig_dir_path)
    fig_dir.mkdir(parents=True, exist_ok=True)

    log(f"[main] Output directory: {out_dir.resolve()}")
    log(f"[main] Figure directory: {fig_dir.resolve()}")
    log(f"[main] Starting Phase 3b Step 2: targeted head-level patching")

    # ---- Validate selected layers ----
    layers = sorted(set(args.layers))
    log(f"[main] Selected layers: {layers}")

    # ---- Load contrast examples ----
    examples = load_contrast_examples(contrast_file)
    if args.max_examples is not None:
        examples = examples[:args.max_examples]
        log(f"[main] Limiting to first {args.max_examples} contrast examples")

    if len(examples) == 0:
        log("[main] ERROR: no valid contrast examples. Exiting.")
        sys.exit(1)

    # ---- Load model ----
    model = load_model(args.model, args.device)

    # ---- Validate layers against model ----
    n_model_layers = model.cfg.n_layers
    invalid_layers = [l for l in layers if l < 0 or l >= n_model_layers]
    if invalid_layers:
        log(f"[main] ERROR: layers {invalid_layers} are out of range for model with {n_model_layers} layers. Exiting.")
        sys.exit(1)

    n_examples = len(examples)
    n_layers = len(layers)
    n_heads = model.cfg.n_heads
    n_total = n_examples * n_layers * n_heads

    log("\n" + "=" * 70)
    log("Phase 3b (Step 2): Targeted Head-Level Activation Patching")
    log(f"  examples:             {n_examples}")
    log(f"  layers:               {layers}")
    log(f"  heads per layer:      {n_heads}")
    log(f"  patches per example:  {n_layers * n_heads}")
    log(f"  total patch runs:     {n_total}")
    log(f"  hook template:        {HEAD_HOOK_TEMPLATE}")
    log(f"  metric:               {args.metric}")
    log(f"  device:               {args.device}")
    log(f"  head log interval:    {args.head_log_interval}")
    log("=" * 70 + "\n")

    # ---- Main loop ----
    all_rows: list[dict] = []
    run_t0 = time.time()
    n_valid = 0
    n_skipped = 0

    for i, example in enumerate(examples):
        example_outer_t0 = time.time()
        reset_cuda_peak_memory_stats(args.device)

        log(
            f"[{i + 1}/{n_examples}] START example_id={example['example_id']} "
            f"domain={example['domain']} | {get_cuda_mem_string(args.device)}"
        )

        rows = run_head_sweep_for_example(
            model=model,
            example=example,
            layers=layers,
            metric=args.metric,
            device=args.device,
            verbose=args.verbose,
            head_log_interval=max(1, args.head_log_interval),
        )
        all_rows.extend(rows)

        if rows and rows[0]["valid_example"]:
            n_valid += 1
            example_status = "valid"
        else:
            n_skipped += 1
            example_status = "skipped"

        # VRAM hygiene
        if args.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        example_outer_elapsed = time.time() - example_outer_t0
        log(
            f"[{i + 1}/{n_examples}] END example_id={example['example_id']} "
            f"status={example_status} "
            f"time={format_seconds(example_outer_elapsed)} | "
            f"{get_cuda_mem_string(args.device)}"
        )

    elapsed = time.time() - run_t0
    log(f"\n[main] Patching loop finished in {format_seconds(elapsed)} ({n_valid} valid, {n_skipped} skipped)")

    # ---- Save detailed results ----
    results_df = pd.DataFrame(all_rows)

    file_prefix = f"{slug}_"
    detail_path = out_dir / f"{file_prefix}head_patch_results.csv"
    t0 = time.time()
    results_df.to_csv(detail_path, index=False, encoding="utf-8")
    log(f"[save] {detail_path} ({len(results_df)} rows) in {format_seconds(time.time() - t0)}")

    # ---- Aggregate and save summary ----
    summary_df = aggregate_head_results(results_df, args.metric)
    summary_path = out_dir / f"{file_prefix}head_patch_summary.csv"
    t0 = time.time()
    summary_df.to_csv(summary_path, index=False, encoding="utf-8")
    log(f"[save] {summary_path} in {format_seconds(time.time() - t0)}")

    # ---- Plot heatmap ----
    fig_path = fig_dir / f"{file_prefix}head_patch_heatmap.png"
    plot_head_heatmap(summary_df, str(fig_path), args.metric, n_valid)

    # ---- Console summary ----
    log("\n" + "=" * 70)
    log("HEAD-LEVEL PATCHING SUMMARY")
    log("=" * 70)
    log(f"  Valid contrast examples: {n_valid}")
    log(f"  Skipped examples:        {n_skipped}")

    if not summary_df.empty:
        # Show top-10 heads by mean delta
        top_k = min(10, len(summary_df))
        top_heads = summary_df.nlargest(top_k, "mean_delta")
        log(f"\n  Top {top_k} heads by mean Δℓ,h ({args.metric}):")
        for _, row in top_heads.iterrows():
            log(
                f"    Layer {int(row['layer']):2d} Head {int(row['head']):2d}: "
                f"mean_Δ={row['mean_delta']:+.4f} "
                f"std={row['std_delta']:.4f} "
                f"n={int(row['n_examples'])}"
            )

        # Also show bottom-3 for completeness
        bottom_k = min(3, len(summary_df))
        bottom_heads = summary_df.nsmallest(bottom_k, "mean_delta")
        log(f"\n  Bottom {bottom_k} heads by mean Δℓ,h:")
        for _, row in bottom_heads.iterrows():
            log(
                f"    Layer {int(row['layer']):2d} Head {int(row['head']):2d}: "
                f"mean_Δ={row['mean_delta']:+.4f} "
                f"std={row['std_delta']:.4f}"
            )

    total_elapsed = time.time() - overall_t0
    log("\n" + "=" * 70)
    if n_valid >= 20:
        log("Phase 3b Step 2 (head-level patching) COMPLETE.")
        log("  Phase 3b is now fully done. Proceed to Phase 3c (cross-condition: clean vs noisy).")
    elif n_valid > 0:
        log(f"Phase 3b Step 2 produced results but only {n_valid} valid examples (target: 20+).")
    else:
        log("Phase 3b Step 2 FAILED: no valid examples. Check token alignment / tokenisation.")
    log(f"Total wall time: {format_seconds(total_elapsed)}")
    log("=" * 70)


if __name__ == "__main__":
    main()
