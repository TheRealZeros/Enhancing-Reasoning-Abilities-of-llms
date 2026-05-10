#!/usr/bin/env python3
"""
Phase 3a: Layer-Level Activation Patching
==========================================
For each contrast example (Cell A wrong, Cell C correct), patches the
residual stream from the structured run (Cell C) into the direct run
(Cell A) one layer at a time. Measures the causal mediation effect Δℓ:

    Δℓ = score(patched at ℓ) − score(baseline)

where "score" is the logit (default) or probability for the gold answer's
first token at the final sequence position.

A large positive Δℓ means layer ℓ carries causally relevant information
that the structured prompt provides and the direct prompt lacks.

Usage:
    python scripts/phase_3a_layer_patching/activation_patching.py
    python scripts/phase_3a_layer_patching/activation_patching.py --verbose --device cuda

Outputs:
    results/phase_3a_layer_patching/layer_patch_results.csv   – one row per (example, layer)
    results/phase_3a_layer_patching/layer_patch_summary.csv   – per-layer aggregated Δℓ
    figures/phase_3a_layer_patching/layer_patch_curve.png      – primary thesis figure

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
# Contrast example loading and validation
# ---------------------------------------------------------------------------

REQUIRED_KEYS = {"example_id", "domain", "gold_answer", "cell_A", "cell_C"}
REQUIRED_CELL_KEYS = {"prompt"}


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
# Token-level utilities
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
# Structured run (Cell C) — cache all residual stream activations
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
# Hook construction — residual stream patching
# ---------------------------------------------------------------------------

def make_resid_patch_hook(cached_activation: torch.Tensor):
    """
    Create a TransformerLens hook function that REPLACES the residual
    stream at the FINAL sequence position only.
    """
    def hook_fn(activation, hook):
        patched = activation.clone()
        patched[:, -1, :] = cached_activation[:, -1, :]
        return patched
    return hook_fn


# ---------------------------------------------------------------------------
# Layer sweep for one contrast example
# ---------------------------------------------------------------------------

def run_layer_sweep_for_example(
    model,
    example: dict,
    metric: str,
    hook_template: str,
    device: str,
    verbose: bool,
    layer_log_interval: int = 1,
) -> list[dict]:
    """
    For one contrast example, run the full layer-level patching sweep.
    """
    example_t0 = time.time()

    ex_id = example["example_id"]
    domain = example["domain"]
    gold = example["gold_answer"]

    # Materialise prompts from the stored cell schema
    prompt_a = materialise_prompt(example["cell_A"], model.tokenizer)
    prompt_c = materialise_prompt(example["cell_C"], model.tokenizer)

    log(f"  [example:{ex_id}] Stage 1/5: tokenising prompts")

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
        return [{
            "example_id": ex_id,
            "domain": domain,
            "layer": -1,
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
        }]

    log(f"  [example:{ex_id}] Stage 2/5: resolving gold token")
    try:
        gold_token_id, gold_token_str = get_target_token_id(model, gold)
    except ValueError as e:
        reason = f"tokenisation error: {e}"
        log(f"  [example:{ex_id}] SKIP | {reason}")
        return [{
            "example_id": ex_id,
            "domain": domain,
            "layer": -1,
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
        }]

    gold_token_count = len(model.tokenizer.encode(" " + gold, add_special_tokens=False))
    log(
        f"  [example:{ex_id}] Gold token resolved | "
        f"gold='{gold}' first_token='{gold_token_str}' id={gold_token_id} token_count={gold_token_count}"
    )

    log(f"  [example:{ex_id}] Stage 3/5: structured cached run")
    logits_c, cache = run_structured_with_cache(
        model=model,
        tokens=tokens_c,
        verbose=verbose,
        device=device,
    )

    log(f"  [example:{ex_id}] Stage 4/5: direct baseline run")
    logits_a = run_direct_baseline(
        model=model,
        tokens=tokens_a,
        verbose=verbose,
        device=device,
    )
    baseline_score = get_score_for_token(logits_a, gold_token_id, metric)
    structured_score = get_score_for_token(logits_c, gold_token_id, metric)

    log(
        f"  [example:{ex_id}] Baseline vs structured | "
        f"baseline={baseline_score:.6f} structured={structured_score:.6f} "
        f"delta_structured_baseline={structured_score - baseline_score:+.6f}"
    )

    log(f"  [example:{ex_id}] Stage 5/5: layer sweep starting")
    n_layers = model.cfg.n_layers
    rows = []
    layer_times = []

    sweep_t0 = time.time()

    for layer in range(n_layers):
        layer_t0 = time.time()
        hook_name = hook_template.format(layer=layer)

        if verbose or layer == 0 or (layer + 1) % layer_log_interval == 0 or layer == n_layers - 1:
            log(
                f"    [layer {layer:02d}/{n_layers - 1:02d}] "
                f"hook={hook_name} | starting | {get_cuda_mem_string(device)}"
            )

        cached_act = cache[hook_name]
        hook_fn = make_resid_patch_hook(cached_act)

        with torch.no_grad():
            patched_logits = model.run_with_hooks(
                tokens_a,
                fwd_hooks=[(hook_name, hook_fn)],
            )

        patched_score = get_score_for_token(patched_logits, gold_token_id, metric)
        delta = patched_score - baseline_score

        layer_elapsed = time.time() - layer_t0
        layer_times.append(layer_elapsed)

        rows.append({
            "example_id": ex_id,
            "domain": domain,
            "layer": layer,
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

        if verbose or layer == 0 or (layer + 1) % layer_log_interval == 0 or layer == n_layers - 1:
            log(
                f"    [layer {layer:02d}/{n_layers - 1:02d}] "
                f"done in {format_seconds(layer_elapsed)} | "
                f"patched={patched_score:.6f} delta={delta:+.6f} | "
                f"{get_cuda_mem_string(device)}"
            )

    sweep_elapsed = time.time() - sweep_t0
    avg_layer_time = sum(layer_times) / len(layer_times) if layer_times else 0.0

    log(
        f"  [example:{ex_id}] Layer sweep complete in {format_seconds(sweep_elapsed)} | "
        f"avg_per_layer={format_seconds(avg_layer_time)}"
    )

    del logits_c, cache, logits_a, patched_logits

    total_elapsed = time.time() - example_t0
    log(f"  [example:{ex_id}] COMPLETE in {format_seconds(total_elapsed)}")
    return rows


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_layer_results(
    df: pd.DataFrame,
    hook_template: str,
    metric: str,
) -> pd.DataFrame:
    """
    Compute per-layer mean and std of Δℓ across all valid contrast examples.
    """
    valid = df[df["valid_example"]].copy()
    if len(valid) == 0:
        log("[aggregate] WARNING: no valid examples to aggregate")
        return pd.DataFrame()

    t0 = time.time()
    log(f"[aggregate] Aggregating {len(valid)} valid rows")

    summary = (
        valid
        .groupby("layer")
        .agg(
            mean_delta=("delta", "mean"),
            std_delta=("delta", "std"),
            n_examples=("delta", "size"),
        )
        .reset_index()
    )
    summary["hook_name"] = summary["layer"].apply(
        lambda l: hook_template.format(layer=l)
    )
    summary["metric"] = metric
    summary = summary[["layer", "hook_name", "metric", "mean_delta", "std_delta", "n_examples"]]

    elapsed = time.time() - t0
    log(f"[aggregate] Done in {format_seconds(elapsed)}")
    return summary


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_layer_curve(
    summary_df: pd.DataFrame,
    output_path: str,
    metric: str,
    n_examples: int,
):
    """
    Plot the mean Δℓ curve across layers with ±1 standard deviation band.
    """
    if summary_df.empty:
        log("[plot] WARNING: no data to plot")
        return

    t0 = time.time()
    log(f"[plot] Generating figure at {output_path}")

    layers = summary_df["layer"].values
    means = summary_df["mean_delta"].values
    stds = summary_df["std_delta"].fillna(0).values

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(layers, means, marker="o", markersize=4, linewidth=1.5, color="#2c7bb6", label="Mean Δℓ")
    ax.fill_between(layers, means - stds, means + stds, alpha=0.2, color="#2c7bb6", label="±1 std")
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)

    ax.set_xlabel("Layer", fontsize=12)
    ylabel = "Δℓ (logit)" if metric == "logit" else "Δℓ (probability)"
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(
        f"Layer-Level Causal Mediation Effect (n={n_examples} contrast examples)",
        fontsize=13,
    )
    ax.legend(fontsize=10)
    ax.set_xticks(layers[::2] if len(layers) > 1 else layers)

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
    """'EleutherAI/pythia-2.8b' -> 'pythia-2.8b', 'gpt2-large' -> 'gpt2-large'"""
    return model_name.split("/")[-1]


def main():
    parser = argparse.ArgumentParser(
        description="Phase 3a: Layer-level residual stream activation patching"
    )
    parser.add_argument("--contrast-file", type=str,
                        default=None,
                        help="Path to contrast_examples.json from Phase 2 "
                             "(default: dataset/processed/<model-slug>/contrast_examples.json)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory for output CSV files "
                             "(default: results/phase_3a_layer_patching/<model-slug>/)")
    parser.add_argument("--figure-dir", type=str, default=None,
                        help="Directory for output figure files "
                             "(default: figures/phase_3a_layer_patching/<model-slug>/)")
    parser.add_argument("--model", type=str, default="EleutherAI/pythia-2.8b",
                        help="HuggingFace model name for HookedTransformer")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda or cpu)")
    parser.add_argument("--max-examples", type=int, default=None,
                        help="Limit to first N contrast examples (for debugging)")
    parser.add_argument("--hook-name", type=str,
                        default="blocks.{layer}.hook_resid_post",
                        help="Hook name template with {layer} placeholder")
    parser.add_argument("--metric", type=str, default="logit",
                        choices=["logit", "prob"],
                        help="Score metric: 'logit' (default) or 'prob'")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-example details and stage timings")
    parser.add_argument("--layer-log-interval", type=int, default=4,
                        help="Print layer progress every N layers (default: 4)")
    args = parser.parse_args()

    # ---- Resolve model-namespaced defaults ----
    slug = _model_slug(args.model)
    contrast_file = args.contrast_file or f"dataset/processed/{slug}/contrast_examples.json"
    out_dir_path  = args.output_dir   or f"results/phase_3a_layer_patching/{slug}"
    fig_dir_path  = args.figure_dir   or f"figures/phase_3a_layer_patching/{slug}"

    overall_t0 = time.time()

    out_dir = Path(out_dir_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = Path(fig_dir_path)
    fig_dir.mkdir(parents=True, exist_ok=True)

    log(f"[main] Output directory: {out_dir.resolve()}")
    log(f"[main] Figure directory: {fig_dir.resolve()}")
    log(f"[main] Starting Phase 3a activation patching pipeline")

    examples = load_contrast_examples(contrast_file)
    if args.max_examples is not None:
        examples = examples[:args.max_examples]
        log(f"[main] Limiting to first {args.max_examples} contrast examples")

    if len(examples) == 0:
        log("[main] ERROR: no valid contrast examples. Exiting.")
        sys.exit(1)

    model = load_model(args.model, args.device)

    n_layers = model.cfg.n_layers
    n_examples = len(examples)
    n_total = n_examples * n_layers

    log("\n" + "=" * 70)
    log("Phase 3a: Layer-Level Activation Patching")
    log(f"  examples:           {n_examples}")
    log(f"  layers per example: {n_layers}")
    log(f"  total patch runs:   {n_total}")
    log(f"  hook:               {args.hook_name}")
    log(f"  metric:             {args.metric}")
    log(f"  device:             {args.device}")
    log(f"  layer log interval: {args.layer_log_interval}")
    log("=" * 70 + "\n")

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

        rows = run_layer_sweep_for_example(
            model=model,
            example=example,
            metric=args.metric,
            hook_template=args.hook_name,
            device=args.device,
            verbose=args.verbose,
            layer_log_interval=max(1, args.layer_log_interval),
        )
        all_rows.extend(rows)

        if rows and rows[0]["valid_example"]:
            n_valid += 1
            example_status = "valid"
        else:
            n_skipped += 1
            example_status = "skipped"

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

    results_df = pd.DataFrame(all_rows)

    detail_path = out_dir / "layer_patch_results.csv"
    t0 = time.time()
    results_df.to_csv(detail_path, index=False, encoding="utf-8")
    log(f"[save] {detail_path} ({len(results_df)} rows) in {format_seconds(time.time() - t0)}")

    summary_df = aggregate_layer_results(results_df, args.hook_name, args.metric)
    summary_path = out_dir / "layer_patch_summary.csv"
    t0 = time.time()
    summary_df.to_csv(summary_path, index=False, encoding="utf-8")
    log(f"[save] {summary_path} in {format_seconds(time.time() - t0)}")

    fig_path = fig_dir / "layer_patch_curve.png"
    plot_layer_curve(summary_df, str(fig_path), args.metric, n_valid)

    log("\n" + "=" * 70)
    log("LAYER-LEVEL PATCHING SUMMARY")
    log("=" * 70)
    log(f"  Valid contrast examples: {n_valid}")
    log(f"  Skipped examples:        {n_skipped}")

    if not summary_df.empty:
        top_k = 5
        top_layers = summary_df.nlargest(top_k, "mean_delta")
        log(f"\n  Top {top_k} layers by mean Δℓ ({args.metric}):")
        for _, row in top_layers.iterrows():
            log(
                f"    Layer {int(row['layer']):2d}: "
                f"mean_Δ={row['mean_delta']:+.4f} "
                f"std={row['std_delta']:.4f} "
                f"n={int(row['n_examples'])}"
            )

        bottom_layers = summary_df.nsmallest(3, "mean_delta")
        log(f"\n  Bottom 3 layers by mean Δℓ:")
        for _, row in bottom_layers.iterrows():
            log(
                f"    Layer {int(row['layer']):2d}: "
                f"mean_Δ={row['mean_delta']:+.4f} "
                f"std={row['std_delta']:.4f}"
            )

    total_elapsed = time.time() - overall_t0
    log("\n" + "=" * 70)
    if n_valid >= 20:
        log("Phase 3a COMPLETE. Proceed to Phase 3b (component-level patching).")
    elif n_valid > 0:
        log(f"Phase 3a produced results but only {n_valid} valid examples (target: 20+).")
    else:
        log("Phase 3a FAILED: no valid examples. Check token alignment / tokenisation.")
    log(f"Total wall time: {format_seconds(total_elapsed)}")
    log("=" * 70)


if __name__ == "__main__":
    main()