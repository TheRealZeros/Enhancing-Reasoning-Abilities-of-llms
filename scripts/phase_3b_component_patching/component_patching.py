#!/usr/bin/env python3
"""
Phase 3b (Step 1 of 2): Broad Component Decomposition
=======================================================
Decomposes the layer-level causal mediation effects found in Phase 3a into
attention-output vs MLP-output contributions at selected high-effect layers.

For each contrast example (Cell A wrong, Cell C correct), patches individual
component outputs (attention out, MLP out) from the structured run (Cell C)
into the direct run (Cell A). Measures the component-level causal mediation
effect Δℓ,c:

    Δℓ,c = score(patched component c at layer ℓ) − score(baseline)

where "score" is the logit (default) or probability for the gold answer's
first token at the final sequence position.

A large positive Δℓ,c means that component c at layer ℓ carries causally
relevant reasoning information present in the structured run but absent
(or weaker) in the direct run.

This script performs a broad component decomposition (attention output vs
MLP output) but does NOT resolve individual attention heads. If late-layer
attention effects are present, a targeted head-level follow-up (Step 2)
should be run next using head_patching.py.

Noisy-condition comparison is deferred to Phase 3c.

Usage:
    python scripts/phase_3b_component_patching/component_patching.py --layers 24 25 29 30 31
    python scripts/phase_3b_component_patching/component_patching.py --layers 24 25 29 30 31 --verbose

Outputs:
    results/phase_3b_component_patching/component_patch_results.csv  – one row per (example, layer, component)
    results/phase_3b_component_patching/component_patch_summary.csv  – per-(layer, component) aggregated Δℓ,c
    figures/phase_3b_component_patching/component_patch_heatmap.png   – thesis figure: component heatmap

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

try:
    from scripts.utils.contrast_config import contrast_path_for, model_file_prefix, output_prefix_for
except ModuleNotFoundError:
    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))
    from scripts.utils.contrast_config import contrast_path_for, model_file_prefix, output_prefix_for


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
# Component definitions
# ---------------------------------------------------------------------------

# Each component is a (name, hook_template) pair.
# hook_template uses {layer} as placeholder, resolved at runtime.
COMPONENTS = [
    ("attn_out", "blocks.{layer}.hook_attn_out"),
    ("mlp_out",  "blocks.{layer}.hook_mlp_out"),
]


# ---------------------------------------------------------------------------
# Logging and VRAM utilities (same as Phase 3a)
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

REQUIRED_KEYS = {"example_id", "domain", "gold_answer"}
REQUIRED_CELL_KEYS = {"prompt"}


def load_contrast_examples(
    path: str,
    source_cell: str = "A",
    donor_cell: str = "C",
) -> list[dict]:
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
        ok, reason = validate_contrast_example(ex, i, source_cell, donor_cell)
        if ok:
            valid.append(ex)
        else:
            log(f"[load] WARNING: skipping index {i}: {reason}")

    elapsed = time.time() - t0
    log(f"[load] {len(valid)}/{len(data)} contrast examples validated from {path} in {format_seconds(elapsed)}")
    return valid


def validate_contrast_example(
    ex: dict,
    idx: int,
    source_cell: str = "A",
    donor_cell: str = "C",
) -> tuple[bool, str]:
    """Check that a contrast example has all required fields."""
    if not isinstance(ex, dict):
        return False, f"index {idx} is not a dict"

    missing = REQUIRED_KEYS - set(ex.keys())
    if missing:
        return False, f"missing top-level keys: {missing}"

    for cell_name in (f"cell_{source_cell}", f"cell_{donor_cell}"):
        if cell_name not in ex:
            return False, f"missing cell key: {cell_name}"
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
# Token-level utilities (identical to Phase 3a)
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
# Structured run (Cell C) — cache component-level activations
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
# Hook construction — component-level patching
# ---------------------------------------------------------------------------

def make_component_patch_hook(cached_activation: torch.Tensor):
    """
    Create a TransformerLens hook function that REPLACES the component
    output at the FINAL sequence position only.

    This patches either the attention output or MLP output at one layer,
    injecting structured-run activations into the direct-run forward pass.
    The hook replaces only the final position to match Phase 3a methodology.
    """
    def hook_fn(activation, hook):
        patched = activation.clone()
        patched[:, -1, :] = cached_activation[:, -1, :]
        return patched
    return hook_fn


# ---------------------------------------------------------------------------
# Component sweep for one contrast example
# ---------------------------------------------------------------------------

def run_component_sweep_for_example(
    model,
    example: dict,
    layers: list[int],
    metric: str,
    device: str,
    verbose: bool,
    component_log_interval: int = 1,
    source_cell: str = "A",
    donor_cell: str = "C",
) -> list[dict]:
    """
    For one contrast example, patch each component (attn_out, mlp_out)
    at each selected layer and record the causal mediation effect.
    """
    example_t0 = time.time()

    ex_id = example["example_id"]
    domain = example["domain"]
    gold = example["gold_answer"]
    prompt_src = materialise_prompt(example[f"cell_{source_cell}"], model.tokenizer)
    prompt_don = materialise_prompt(example[f"cell_{donor_cell}"], model.tokenizer)

    n_components = len(COMPONENTS)
    n_layers = len(layers)
    total_patches = n_layers * n_components

    log(f"  [example:{ex_id}] Stage 1/5: tokenising prompts")

    # ---- Stage 1: Tokenise and verify alignment ----
    token_t0 = time.time()
    tokens_src = model.tokenizer.encode(prompt_src, return_tensors="pt").to(device)
    tokens_don = model.tokenizer.encode(prompt_don, return_tensors="pt").to(device)
    len_src = tokens_src.shape[1]
    len_don = tokens_don.shape[1]
    token_elapsed = time.time() - token_t0
    log(
        f"  [example:{ex_id}] Tokenisation done in {format_seconds(token_elapsed)} | "
        f"len_{source_cell}={len_src} len_{donor_cell}={len_don}"
    )

    if len_src != len_don:
        reason = f"token misalignment: Cell {source_cell}={len_src}, Cell {donor_cell}={len_don}"
        log(f"  [example:{ex_id}] SKIP | {reason}")
        return [_make_skip_row(ex_id, domain, gold, metric, len_src, len_don, reason)]

    # ---- Stage 2: Resolve gold token ----
    log(f"  [example:{ex_id}] Stage 2/5: resolving gold token")
    try:
        gold_token_id, gold_token_str = get_target_token_id(model, gold)
    except ValueError as e:
        reason = f"tokenisation error: {e}"
        log(f"  [example:{ex_id}] SKIP | {reason}")
        return [_make_skip_row(ex_id, domain, gold, metric, len_src, len_don, reason)]

    gold_token_count = len(model.tokenizer.encode(" " + gold, add_special_tokens=False))
    log(
        f"  [example:{ex_id}] Gold token resolved | "
        f"gold='{gold}' first_token='{gold_token_str}' id={gold_token_id} token_count={gold_token_count}"
    )

    # ---- Stage 3: Donor cached run ----
    log(f"  [example:{ex_id}] Stage 3/5: donor cached run (cell_{donor_cell})")
    logits_don, cache = run_structured_with_cache(
        model=model,
        tokens=tokens_don,
        verbose=verbose,
        device=device,
    )

    # ---- Stage 4: Source baseline run ----
    log(f"  [example:{ex_id}] Stage 4/5: source baseline run (cell_{source_cell})")
    logits_src = run_direct_baseline(
        model=model,
        tokens=tokens_src,
        verbose=verbose,
        device=device,
    )
    baseline_score = get_score_for_token(logits_src, gold_token_id, metric)
    donor_score = get_score_for_token(logits_don, gold_token_id, metric)

    log(
        f"  [example:{ex_id}] Baseline vs donor | "
        f"baseline={baseline_score:.6f} donor={donor_score:.6f} "
        f"delta_donor_baseline={donor_score - baseline_score:+.6f}"
    )

    # ---- Stage 5: Component sweep ----
    log(f"  [example:{ex_id}] Stage 5/5: component sweep starting ({total_patches} patches across {n_layers} layers)")
    rows = []
    patch_times = []
    patch_count = 0

    sweep_t0 = time.time()

    for layer in layers:
        for comp_name, hook_template in COMPONENTS:
            patch_t0 = time.time()
            hook_name = hook_template.format(layer=layer)
            patch_count += 1

            should_log = (
                verbose
                or patch_count == 1
                or patch_count % component_log_interval == 0
                or patch_count == total_patches
            )

            if should_log:
                log(
                    f"    [{patch_count:02d}/{total_patches:02d}] "
                    f"layer={layer} component={comp_name} hook={hook_name} | "
                    f"starting | {get_cuda_mem_string(device)}"
                )

            # Retrieve the cached activation for this component hook
            cached_act = cache[hook_name]
            hook_fn = make_component_patch_hook(cached_act)

            with torch.no_grad():
                patched_logits = model.run_with_hooks(
                    tokens_src,
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
                "component": comp_name,
                "hook_name": hook_name,
                "metric": metric,
                "gold_answer": gold,
                "gold_token_id": gold_token_id,
                "gold_token_str": gold_token_str,
                "gold_token_count": gold_token_count,
                "direct_token_count": len_src,
                "structured_token_count": len_don,
                "baseline_score": baseline_score,
                "patched_score": patched_score,
                "delta": delta,
                "valid_example": True,
                "skip_reason": "",
            })

            if should_log:
                log(
                    f"    [{patch_count:02d}/{total_patches:02d}] "
                    f"layer={layer} component={comp_name} | "
                    f"done in {format_seconds(patch_elapsed)} | "
                    f"patched={patched_score:.6f} delta={delta:+.6f} | "
                    f"{get_cuda_mem_string(device)}"
                )

    sweep_elapsed = time.time() - sweep_t0
    avg_patch_time = sum(patch_times) / len(patch_times) if patch_times else 0.0

    log(
        f"  [example:{ex_id}] Component sweep complete in {format_seconds(sweep_elapsed)} | "
        f"avg_per_patch={format_seconds(avg_patch_time)}"
    )

    # Free memory
    del logits_don, cache, logits_src, patched_logits

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
        "component": "",
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

def aggregate_component_results(
    df: pd.DataFrame,
    metric: str,
) -> pd.DataFrame:
    """
    Compute per-(layer, component) mean and std of Δℓ,c across all
    valid contrast examples.
    """
    valid = df[df["valid_example"]].copy()
    if len(valid) == 0:
        log("[aggregate] WARNING: no valid examples to aggregate")
        return pd.DataFrame()

    t0 = time.time()
    log(f"[aggregate] Aggregating {len(valid)} valid rows")

    summary = (
        valid
        .groupby(["layer", "component"])
        .agg(
            mean_delta=("delta", "mean"),
            std_delta=("delta", "std"),
            n_examples=("delta", "size"),
        )
        .reset_index()
    )

    # Add the hook_name for reference
    hook_lookup = {name: tpl for name, tpl in COMPONENTS}
    summary["hook_name"] = summary.apply(
        lambda row: hook_lookup[row["component"]].format(layer=int(row["layer"])),
        axis=1,
    )
    summary["metric"] = metric

    # Sort by layer then component for readability
    summary = summary.sort_values(["layer", "component"]).reset_index(drop=True)
    summary = summary[[
        "layer", "component", "hook_name", "metric",
        "mean_delta", "std_delta", "n_examples",
    ]]

    elapsed = time.time() - t0
    log(f"[aggregate] Done in {format_seconds(elapsed)}")
    return summary


# ---------------------------------------------------------------------------
# Plotting — component heatmap
# ---------------------------------------------------------------------------

def plot_component_heatmap(
    summary_df: pd.DataFrame,
    output_path: str,
    metric: str,
    n_examples: int,
):
    """
    Plot a heatmap of mean Δℓ,c with layers on the x-axis and component
    types on the y-axis.
    """
    if summary_df.empty:
        log("[plot] WARNING: no data to plot")
        return

    t0 = time.time()
    log(f"[plot] Generating component heatmap at {output_path}")

    # Pivot to (component × layer) matrix
    pivot = summary_df.pivot(index="component", columns="layer", values="mean_delta")

    # Ensure consistent component ordering: attn_out first, then mlp_out
    comp_order = [name for name, _ in COMPONENTS if name in pivot.index]
    pivot = pivot.reindex(comp_order)

    # Sort columns (layers) numerically
    pivot = pivot[sorted(pivot.columns)]

    fig, ax = plt.subplots(figsize=(max(6, len(pivot.columns) * 1.2), 3.5))

    data = pivot.values
    layers = list(pivot.columns)
    components = list(pivot.index)

    # Colour scale: diverging, centred at 0
    vmax = max(abs(np.nanmin(data)), abs(np.nanmax(data)))
    vmin = -vmax if vmax > 0 else -1.0

    im = ax.imshow(
        data,
        aspect="auto",
        cmap="RdBu_r",
        vmin=vmin,
        vmax=vmax,
    )

    # Annotate cells with numeric values
    for i in range(len(components)):
        for j in range(len(layers)):
            val = data[i, j]
            if not np.isnan(val):
                text_colour = "white" if abs(val) > 0.6 * vmax else "black"
                ax.text(
                    j, i, f"{val:+.2f}",
                    ha="center", va="center",
                    fontsize=9, fontweight="bold",
                    color=text_colour,
                )

    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels([str(l) for l in layers], fontsize=10)
    ax.set_yticks(range(len(components)))
    ax.set_yticklabels(components, fontsize=10)
    ax.set_xlabel("Layer", fontsize=11)
    ax.set_ylabel("Component", fontsize=11)

    metric_label = "logit" if metric == "logit" else "probability"
    ax.set_title(
        f"Component-Level Causal Mediation Effect ({metric_label}, n={n_examples})",
        fontsize=12,
    )

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(f"Mean Δℓ,c ({metric_label})", fontsize=10)

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
        description="Phase 3b Step 1: Broad component decomposition (attn_out vs mlp_out) at selected layers"
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
                        default=[24, 25, 29, 30, 31],
                        help="Layers to patch (from Phase 3a top-k selection)")
    parser.add_argument("--metric", type=str, default="logit",
                        choices=["logit", "prob"],
                        help="Score metric: 'logit' (default) or 'prob'")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-patch details and stage timings")
    parser.add_argument("--component-log-interval", type=int, default=2,
                        help="Print patch progress every N patches (default: 2)")
    parser.add_argument("--source-cell", type=str, default="A",
                        help="Source (baseline/corrupted) cell letter, e.g. 'A' or 'B' (default: A)")
    parser.add_argument("--donor-cell", type=str, default="C",
                        help="Donor (structured/clean) cell letter, e.g. 'C' or 'D' (default: C)")
    parser.add_argument("--output-prefix", type=str, default=None,
                        help="Optional filename prefix override. Defaults come from source/donor contrast routing.")
    args = parser.parse_args()

    slug = _model_slug(args.model)
    source_cell = args.source_cell.upper()
    donor_cell = args.donor_cell.upper()
    file_prefix = model_file_prefix(slug, output_prefix_for(source_cell, donor_cell, args.output_prefix))
    default_contrast = contrast_path_for(slug, source_cell, donor_cell)
    base_out = f"results/phase_3b_component_patching/{slug}"
    base_fig = f"figures/phase_3b_component_patching/{slug}"
    contrast_file = args.contrast_file or default_contrast
    out_dir_path  = args.output_dir   or base_out
    fig_dir_path  = args.figure_dir   or base_fig

    overall_t0 = time.time()

    out_dir = Path(out_dir_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = Path(fig_dir_path)
    fig_dir.mkdir(parents=True, exist_ok=True)

    log(f"[main] Output directory: {out_dir.resolve()}")
    log(f"[main] Figure directory: {fig_dir.resolve()}")
    log(f"[main] Starting Phase 3b component-level activation patching")

    # ---- Validate selected layers ----
    layers = sorted(set(args.layers))
    log(f"[main] Selected layers: {layers}")

    # ---- Load contrast examples ----
    examples = load_contrast_examples(contrast_file, source_cell, donor_cell)
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
    n_components = len(COMPONENTS)
    n_total = n_examples * n_layers * n_components

    log("\n" + "=" * 70)
    log("Phase 3b (Step 1): Broad Component Decomposition (attn_out vs mlp_out)")
    log(f"  examples:                {n_examples}")
    log(f"  layers:                  {layers}")
    log(f"  components per layer:    {n_components} ({', '.join(n for n, _ in COMPONENTS)})")
    log(f"  patches per example:     {n_layers * n_components}")
    log(f"  total patch runs:        {n_total}")
    log(f"  metric:                  {args.metric}")
    log(f"  device:                  {args.device}")
    log(f"  component log interval:  {args.component_log_interval}")
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

        rows = run_component_sweep_for_example(
            model=model,
            example=example,
            layers=layers,
            metric=args.metric,
            device=args.device,
            verbose=args.verbose,
            component_log_interval=max(1, args.component_log_interval),
            source_cell=source_cell,
            donor_cell=donor_cell,
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

    detail_path = out_dir / f"{file_prefix}component_patch_results.csv"
    t0 = time.time()
    results_df.to_csv(detail_path, index=False, encoding="utf-8")
    log(f"[save] {detail_path} ({len(results_df)} rows) in {format_seconds(time.time() - t0)}")

    # ---- Aggregate and save summary ----
    summary_df = aggregate_component_results(results_df, args.metric)
    summary_path = out_dir / f"{file_prefix}component_patch_summary.csv"
    t0 = time.time()
    summary_df.to_csv(summary_path, index=False, encoding="utf-8")
    log(f"[save] {summary_path} in {format_seconds(time.time() - t0)}")

    # ---- Plot heatmap ----
    fig_path = fig_dir / f"{file_prefix}component_patch_heatmap.png"
    plot_component_heatmap(summary_df, str(fig_path), args.metric, n_valid)

    # ---- Console summary ----
    log("\n" + "=" * 70)
    log("COMPONENT-LEVEL PATCHING SUMMARY")
    log("=" * 70)
    log(f"  Valid contrast examples: {n_valid}")
    log(f"  Skipped examples:        {n_skipped}")

    if not summary_df.empty:
        # Show all (layer, component) pairs sorted by mean delta descending
        sorted_summary = summary_df.sort_values("mean_delta", ascending=False)
        log(f"\n  All (layer, component) pairs by mean delta ({args.metric}):")
        for _, row in sorted_summary.iterrows():
            log(
                f"    Layer {int(row['layer']):2d} {row['component']:8s}: "
                f"mean_delta={row['mean_delta']:+.4f} "
                f"std={row['std_delta']:.4f} "
                f"n={int(row['n_examples'])}"
            )

        # Highlight the top component
        top = sorted_summary.iloc[0]
        log(
            f"\n  Strongest component: Layer {int(top['layer'])} {top['component']} "
            f"(mean_delta={top['mean_delta']:+.4f})"
        )

    total_elapsed = time.time() - overall_t0
    log("\n" + "=" * 70)
    if n_valid >= 20:
        log("Phase 3b Step 1 (broad component decomposition) COMPLETE.")
        log("  If late-layer attention effects are present, run head_patching.py")
        log("  on the relevant layers (Step 2) before proceeding to Phase 3c.")
    elif n_valid > 0:
        log(f"Phase 3b Step 1 produced results but only {n_valid} valid examples (target: 20+).")
    else:
        log("Phase 3b Step 1 FAILED: no valid examples. Check token alignment / tokenisation.")
    log(f"Total wall time: {format_seconds(total_elapsed)}")
    log("=" * 70)


if __name__ == "__main__":
    main()
