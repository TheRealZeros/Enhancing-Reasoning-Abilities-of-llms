#!/usr/bin/env python3
"""
Phase 3c: Cross-Condition Layer-Level Causal Mediation Comparison
==================================================================
Answers RQ2: Does the presence of contextual distractors alter which
internal components are causally implicated in correct multi-hop
reasoning, and does structured prompting modulate this effect?

This is a LAYER-LEVEL comparison phase.  It does NOT perform component-
or head-level drill-down; that decision should be made after reviewing
the overlay figure produced here.

Pipeline:
  1. Identifies noisy contrast examples (Cell B wrong ∧ Cell D correct)
     from evaluation_results.csv and dataset.json.
  2. For each noisy contrast, patches Cell D (structured noisy) residual
     stream activations into the Cell B (direct noisy) forward pass,
     one layer at a time, measuring the causal mediation effect Δℓ.
  3. Aggregates the noisy Δℓ curve across layers.
  4. Loads the clean Δℓ curve from Phase 3a (layer_patch_summary.csv)
     and verifies metric consistency.
  5. Produces an overlay figure comparing clean (A→C) and noisy (B→D)
     causal mediation curves — the primary RQ2 figure.

The metric, hook, and scoring logic are identical to Phase 3a
(activation_patching.py).  The --patch-scope flag MUST match the
setting used in Phase 3a; the default (final_token) matches the
Phase 3a implementation shipped with this project.

Usage:
    python scripts/phase_3c_cross_condition/cross_condition_patching.py
    python scripts/phase_3c_cross_condition/cross_condition_patching.py \
        --dataset dataset/processed/dataset.json \
        --eval-results results/phase_2_behaviour/evaluation_results.csv \
        --clean-summary results/phase_3a_layer_patching/layer_patch_summary.csv \
        --device cuda --verbose

Outputs:
    dataset/processed/noisy_contrast_examples.json                        – noisy contrast pairs
    results/phase_3a_layer_patching/noisy_layer_patch_results.csv         – one row per (example, layer)
    results/phase_3a_layer_patching/noisy_layer_patch_summary.csv         – per-layer aggregated Δℓ (noisy)
    results/phase_3c_cross_condition/cross_condition_layer_comparison.csv  – merged clean + noisy by layer
    figures/phase_3a_layer_patching/clean_vs_noisy_layer_patch_overlay.png – primary RQ2 figure

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
matplotlib.use("Agg")
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
# Logging utilities (identical to Phase 3a)
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
# Robust boolean parsing
# ---------------------------------------------------------------------------

def robust_bool(value) -> bool:
    """
    Normalise a value to bool.  Handles:
      - Python bool / numpy bool
      - int/float 0 or 1
      - strings: 'true', 'True', 'TRUE', 'false', 'False', 'FALSE',
        '1', '0', 'yes', 'no'
    Raises ValueError on anything else.
    """
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        if value == 1:
            return True
        if value == 0:
            return False
        raise ValueError(f"Cannot interpret numeric value {value!r} as bool")
    if isinstance(value, str):
        low = value.strip().lower()
        if low in ("true", "1", "yes"):
            return True
        if low in ("false", "0", "no"):
            return False
        raise ValueError(f"Cannot interpret string {value!r} as bool")
    raise ValueError(f"Cannot interpret {type(value).__name__} value {value!r} as bool")


# ---------------------------------------------------------------------------
# Evaluation results validation
# ---------------------------------------------------------------------------

def validate_eval_dataframe(eval_df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and clean the evaluation results DataFrame:
      - Normalise example_id to str
      - Normalise 'correct' column to bool via robust_bool
      - Assert no duplicate (example_id, cell) pairs
    Returns a cleaned copy.
    """
    df = eval_df.copy()

    # Normalise IDs to string
    df["example_id"] = df["example_id"].astype(str).str.strip()

    # Normalise 'correct' column
    try:
        df["correct"] = df["correct"].apply(robust_bool)
    except ValueError as e:
        raise ValueError(
            f"Failed to parse 'correct' column in evaluation_results.csv: {e}\n"
            f"  Unique values found: {eval_df['correct'].unique().tolist()}"
        ) from e

    # Check for duplicates
    dupes = df.duplicated(subset=["example_id", "cell"], keep=False)
    if dupes.any():
        dupe_rows = df[dupes][["example_id", "cell"]].drop_duplicates()
        n_dupes = len(dupe_rows)
        sample = dupe_rows.head(5).to_string(index=False)
        raise ValueError(
            f"evaluation_results.csv contains {n_dupes} duplicate (example_id, cell) "
            f"pairs.  Each (example_id, cell) must appear exactly once.\n"
            f"  Sample duplicates:\n{sample}"
        )

    return df


# ---------------------------------------------------------------------------
# Dataset validation
# ---------------------------------------------------------------------------

REQUIRED_DATASET_KEYS = {"id", "domain", "answer", "cells"}


def validate_and_extract_prompt(cells_value, cell_name: str, example_id: str) -> str:
    """
    Extract prompt string from a cell value.  Supports two formats:
      1. Direct string:  cells["B"] = "<prompt text>"
      2. Dict with key:  cells["B"] = {"prompt": "<prompt text>", ...}
    """
    if isinstance(cells_value, str):
        if len(cells_value) == 0:
            raise ValueError(
                f"Example {example_id}: Cell {cell_name} prompt string is empty"
            )
        return cells_value
    if isinstance(cells_value, dict):
        prompt = cells_value.get("prompt")
        if not isinstance(prompt, str) or len(prompt) == 0:
            raise ValueError(
                f"Example {example_id}: Cell {cell_name} dict missing non-empty 'prompt' key"
            )
        return prompt
    raise ValueError(
        f"Example {example_id}: Cell {cell_name} has unsupported type "
        f"{type(cells_value).__name__}; expected str or dict"
    )


def validate_dataset_example(ex: dict, idx: int) -> tuple[bool, str]:
    """Check that a dataset example has all required fields and cells B/D."""
    if not isinstance(ex, dict):
        return False, f"index {idx} is not a dict"

    missing = REQUIRED_DATASET_KEYS - set(ex.keys())
    if missing:
        return False, f"missing top-level keys: {missing}"

    cells = ex.get("cells")
    if not isinstance(cells, dict):
        return False, f"'cells' is not a dict"

    for cell_name in ("B", "D"):
        if cell_name not in cells:
            return False, f"missing cell {cell_name}"
        try:
            validate_and_extract_prompt(cells[cell_name], cell_name, ex["id"])
        except ValueError as e:
            return False, str(e)

    if not isinstance(ex.get("answer"), str) or len(ex["answer"]) == 0:
        return False, "answer is empty or not a string"

    return True, ""


# ---------------------------------------------------------------------------
# Noisy contrast identification
# ---------------------------------------------------------------------------

def identify_noisy_contrasts(
    dataset_path: str,
    eval_results_path: str,
) -> list[dict]:
    """
    Identify noisy contrast examples: Cell B incorrect AND Cell D correct.
    Returns a list of dicts with Cell B and Cell D prompts plus metadata.
    """
    # --- Load and validate dataset ---
    log(f"[contrast] Loading dataset from {dataset_path}")
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    if not isinstance(dataset, list) or len(dataset) == 0:
        raise ValueError("Dataset must be a non-empty JSON list.")

    valid_dataset = []
    for i, ex in enumerate(dataset):
        ok, reason = validate_dataset_example(ex, i)
        if ok:
            valid_dataset.append(ex)
        else:
            log(f"[contrast] WARNING: skipping dataset index {i}: {reason}")

    # Normalise dataset IDs to string
    dataset_by_id = {str(ex["id"]).strip(): ex for ex in valid_dataset}
    log(f"[contrast] Loaded {len(valid_dataset)}/{len(dataset)} valid dataset examples")

    # --- Load and validate evaluation results ---
    log(f"[contrast] Loading evaluation results from {eval_results_path}")
    eval_df = pd.read_csv(eval_results_path)
    log(f"[contrast] Loaded {len(eval_df)} evaluation rows")

    eval_df = validate_eval_dataframe(eval_df)

    cell_b = eval_df[eval_df["cell"] == "B"].set_index("example_id")
    cell_d = eval_df[eval_df["cell"] == "D"].set_index("example_id")
    common_ids = cell_b.index.intersection(cell_d.index)

    contrasts = []
    for eid in common_ids:
        b_row = cell_b.loc[eid]
        d_row = cell_d.loc[eid]
        if (not b_row["correct"]) and d_row["correct"]:
            ex = dataset_by_id.get(str(eid).strip())
            if ex is None:
                log(f"[contrast] WARNING: example {eid} in eval but not in dataset, skipping")
                continue
            # Extract prompts robustly (supports str or dict-with-prompt)
            try:
                validate_and_extract_prompt(ex["cells"]["B"], "B", eid)
                validate_and_extract_prompt(ex["cells"]["D"], "D", eid)
            except ValueError as e:
                log(f"[contrast] WARNING: skipping {eid}: {e}")
                continue

            # Store the clean cell schema (dict with prompt + metadata),
            # NOT the materialised prompt with EOS padding.
            cell_b_data = ex["cells"]["B"]
            cell_d_data = ex["cells"]["D"]
            if isinstance(cell_b_data, str):
                cell_b_data = {"prompt": cell_b_data, "prefix_eos_pad": 0}
            if isinstance(cell_d_data, str):
                cell_d_data = {"prompt": cell_d_data, "prefix_eos_pad": 0}

            contrasts.append({
                "example_id": str(eid),
                "domain": ex["domain"],
                "gold_answer": ex["answer"],
                "cell_B": {
                    **cell_b_data,
                    "generated_answer_raw": str(b_row["generated_answer_raw"]),
                    "generated_answer_normalised": str(b_row["generated_answer_normalised"]),
                    "correct": False,
                },
                "cell_D": {
                    **cell_d_data,
                    "generated_answer_raw": str(d_row["generated_answer_raw"]),
                    "generated_answer_normalised": str(d_row["generated_answer_normalised"]),
                    "correct": True,
                },
            })

    log(f"[contrast] Found {len(contrasts)} noisy contrast examples (B wrong ∧ D correct)")
    return contrasts


# ---------------------------------------------------------------------------
# Metric consistency check for clean summary
# ---------------------------------------------------------------------------

def verify_clean_summary_metric(clean_summary_path: str, expected_metric: str) -> None:
    """
    If the clean summary CSV contains a 'metric' column, verify it matches
    the --metric used for this run.  A mismatch means the clean and noisy
    Δℓ values are not comparable.
    """
    df = pd.read_csv(clean_summary_path, nrows=1)
    if "metric" not in df.columns:
        log(
            f"[metric-check] WARNING: clean summary {clean_summary_path} has no "
            f"'metric' column; cannot verify consistency.  Proceeding with "
            f"--metric={expected_metric} and assuming the clean run used the same."
        )
        return

    clean_metric = str(df["metric"].iloc[0]).strip()
    if clean_metric != expected_metric:
        raise ValueError(
            f"Metric mismatch: clean summary uses '{clean_metric}' but this run "
            f"uses --metric='{expected_metric}'.  The clean and noisy Δℓ curves "
            f"would not be comparable.  Re-run Phase 3a with --metric={expected_metric} "
            f"or change --metric here to '{clean_metric}'."
        )
    log(f"[metric-check] Clean summary metric='{clean_metric}' matches --metric='{expected_metric}' ✓")


# ---------------------------------------------------------------------------
# Model loading (identical to Phase 3a)
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
    """Return the FIRST token ID of the gold answer (with leading space)."""
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
    """Extract the score for a specific token at the LAST sequence position."""
    last_logits = logits[0, -1, :]
    if metric == "logit":
        return last_logits[token_id].item()
    elif metric == "prob":
        probs = torch.softmax(last_logits.float(), dim=-1)
        return probs[token_id].item()
    else:
        raise ValueError(f"Unknown metric: {metric}")


# ---------------------------------------------------------------------------
# Structured noisy run (Cell D) — cache all residual stream activations
# ---------------------------------------------------------------------------

def run_structured_with_cache(
    model,
    tokens: torch.Tensor,
    verbose: bool = False,
    device: str = "cpu",
) -> tuple[torch.Tensor, dict]:
    """Run the structured noisy prompt (Cell D) and cache all activations."""
    if verbose:
        log(f"    [structured-noisy] Starting cached forward pass | seq_len={tokens.shape[1]} | {get_cuda_mem_string(device)}")
    t0 = time.time()
    with torch.no_grad():
        logits, cache = model.run_with_cache(tokens)
    elapsed = time.time() - t0
    if verbose:
        log(f"    [structured-noisy] Done in {format_seconds(elapsed)} | {get_cuda_mem_string(device)}")
    return logits, cache


# ---------------------------------------------------------------------------
# Direct noisy baseline run (Cell B) — no hooks
# ---------------------------------------------------------------------------

def run_direct_baseline(
    model,
    tokens: torch.Tensor,
    verbose: bool = False,
    device: str = "cpu",
) -> torch.Tensor:
    """Run the direct noisy prompt (Cell B) as a clean forward pass."""
    if verbose:
        log(f"    [baseline-noisy] Starting direct forward pass | seq_len={tokens.shape[1]} | {get_cuda_mem_string(device)}")
    t0 = time.time()
    with torch.no_grad():
        logits = model(tokens)
    elapsed = time.time() - t0
    if verbose:
        log(f"    [baseline-noisy] Done in {format_seconds(elapsed)} | {get_cuda_mem_string(device)}")
    return logits


# ---------------------------------------------------------------------------
# Hook construction — configurable patch scope
# ---------------------------------------------------------------------------

def make_resid_patch_hook(cached_activation: torch.Tensor, patch_scope: str):
    """
    Create a TransformerLens hook that replaces the residual stream.

    patch_scope controls WHICH positions are patched:
      - 'final_token': replace only the final sequence position (matches
        Phase 3a default).  This is the standard choice when the causal
        question is about the model's next-token prediction at the last
        position.
      - 'full': replace ALL sequence positions.  Use only if Phase 3a
        was also run with full-sequence patching.

    IMPORTANT: this setting MUST match the scope used in Phase 3a so that
    the clean and noisy Δℓ curves are methodologically comparable.
    """
    if patch_scope == "final_token":
        def hook_fn(activation, hook):
            patched = activation.clone()
            patched[:, -1, :] = cached_activation[:, -1, :]
            return patched
    elif patch_scope == "full":
        def hook_fn(activation, hook):
            return cached_activation
    else:
        raise ValueError(f"Unknown patch_scope: {patch_scope!r}")
    return hook_fn


# ---------------------------------------------------------------------------
# Layer sweep for one noisy contrast example
# ---------------------------------------------------------------------------

def run_layer_sweep_for_example(
    model,
    example: dict,
    metric: str,
    hook_template: str,
    patch_scope: str,
    device: str,
    verbose: bool,
    layer_log_interval: int = 4,
) -> list[dict]:
    """
    For one noisy contrast example, run the full layer-level patching sweep.
    Patches Cell D (structured noisy) activations into the Cell B (direct
    noisy) forward pass at each layer, measuring Δℓ per layer.

    This is a layer-level causal mediation measurement; component-level
    drill-down is NOT performed here.
    """
    example_t0 = time.time()

    ex_id = example["example_id"]
    domain = example["domain"]
    gold = example["gold_answer"]
    prompt_b = materialise_prompt(example["cell_B"], model.tokenizer)
    prompt_d = materialise_prompt(example["cell_D"], model.tokenizer)

    # --- Stage 1: Tokenise ---
    log(f"  [example:{ex_id}] Stage 1/5: tokenising prompts (B and D)")
    token_t0 = time.time()
    tokens_b = model.tokenizer.encode(prompt_b, return_tensors="pt").to(device)
    tokens_d = model.tokenizer.encode(prompt_d, return_tensors="pt").to(device)
    len_b = tokens_b.shape[1]
    len_d = tokens_d.shape[1]
    token_elapsed = time.time() - token_t0
    log(f"  [example:{ex_id}] Tokenisation done in {format_seconds(token_elapsed)} | len_B={len_b} len_D={len_d}")

    # --- Token alignment check (CRITICAL) ---
    if len_b != len_d:
        reason = f"token misalignment: Cell B={len_b}, Cell D={len_d}"
        log(f"  [example:{ex_id}] SKIP | {reason}")
        return [{
            "example_id": ex_id,
            "domain": domain,
            "layer": -1,
            "hook_name": "",
            "patch_scope": patch_scope,
            "metric": metric,
            "gold_answer": gold,
            "gold_token_id": -1,
            "gold_token_str": "",
            "gold_token_count": -1,
            "direct_noisy_token_count": len_b,
            "structured_noisy_token_count": len_d,
            "baseline_score": float("nan"),
            "patched_score": float("nan"),
            "delta": float("nan"),
            "valid_example": False,
            "skip_reason": reason,
        }]

    # --- Stage 2: Gold token ---
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
            "patch_scope": patch_scope,
            "metric": metric,
            "gold_answer": gold,
            "gold_token_id": -1,
            "gold_token_str": "",
            "gold_token_count": -1,
            "direct_noisy_token_count": len_b,
            "structured_noisy_token_count": len_d,
            "baseline_score": float("nan"),
            "patched_score": float("nan"),
            "delta": float("nan"),
            "valid_example": False,
            "skip_reason": reason,
        }]

    gold_token_count = len(model.tokenizer.encode(" " + gold, add_special_tokens=False))
    log(
        f"  [example:{ex_id}] Gold token resolved | "
        f"gold='{gold}' first_token='{gold_token_str}' id={gold_token_id} "
        f"token_count={gold_token_count}"
    )

    # --- Stage 3: Structured noisy cached run (Cell D) ---
    log(f"  [example:{ex_id}] Stage 3/5: structured noisy cached run (Cell D)")
    logits_d, cache = run_structured_with_cache(
        model=model,
        tokens=tokens_d,
        verbose=verbose,
        device=device,
    )

    # --- Stage 4: Direct noisy baseline run (Cell B) ---
    log(f"  [example:{ex_id}] Stage 4/5: direct noisy baseline run (Cell B)")
    logits_b = run_direct_baseline(
        model=model,
        tokens=tokens_b,
        verbose=verbose,
        device=device,
    )
    baseline_score = get_score_for_token(logits_b, gold_token_id, metric)
    structured_score = get_score_for_token(logits_d, gold_token_id, metric)

    log(
        f"  [example:{ex_id}] Baseline vs structured (noisy) | "
        f"baseline_B={baseline_score:.6f} structured_D={structured_score:.6f} "
        f"delta_D_minus_B={structured_score - baseline_score:+.6f}"
    )

    # --- Stage 5: Layer sweep ---
    log(f"  [example:{ex_id}] Stage 5/5: layer sweep starting (patch_scope={patch_scope})")
    n_layers = model.cfg.n_layers
    rows = []
    layer_times = []
    sweep_t0 = time.time()

    for layer in range(n_layers):
        layer_t0 = time.time()
        hook_name = hook_template.format(layer=layer)

        show_log = (
            verbose or layer == 0
            or (layer + 1) % layer_log_interval == 0
            or layer == n_layers - 1
        )

        if show_log:
            log(
                f"    [layer {layer:02d}/{n_layers - 1:02d}] "
                f"hook={hook_name} | starting | {get_cuda_mem_string(device)}"
            )

        cached_act = cache[hook_name]
        hook_fn = make_resid_patch_hook(cached_act, patch_scope)

        with torch.no_grad():
            patched_logits = model.run_with_hooks(
                tokens_b,
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
            "patch_scope": patch_scope,
            "metric": metric,
            "gold_answer": gold,
            "gold_token_id": gold_token_id,
            "gold_token_str": gold_token_str,
            "gold_token_count": gold_token_count,
            "direct_noisy_token_count": len_b,
            "structured_noisy_token_count": len_d,
            "baseline_score": baseline_score,
            "patched_score": patched_score,
            "delta": delta,
            "valid_example": True,
            "skip_reason": "",
        })

        if show_log:
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

    # Free tensors
    del logits_d, cache, logits_b, patched_logits

    total_elapsed = time.time() - example_t0
    log(f"  [example:{ex_id}] COMPLETE in {format_seconds(total_elapsed)}")
    return rows


# ---------------------------------------------------------------------------
# Aggregation (identical logic to Phase 3a)
# ---------------------------------------------------------------------------

def aggregate_layer_results(
    df: pd.DataFrame,
    hook_template: str,
    metric: str,
) -> pd.DataFrame:
    """Compute per-layer mean and std of Δℓ across all valid noisy contrast examples."""
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
# Cross-condition merge and comparison
# ---------------------------------------------------------------------------

def build_cross_condition_comparison(
    clean_summary_path: str,
    noisy_summary: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge clean (A→C) and noisy (B→D) layer-level summaries, computing the
    delta_gap = noisy_mean_delta − clean_mean_delta per layer.  This is the
    primary comparison for RQ2 at the layer level.
    """
    log(f"[compare] Loading clean summary from {clean_summary_path}")
    clean = pd.read_csv(clean_summary_path)
    log(f"[compare] Clean summary has {len(clean)} rows")

    if noisy_summary.empty:
        log("[compare] WARNING: noisy summary is empty; returning empty comparison")
        return pd.DataFrame()

    # Rename columns for the merge
    clean_renamed = clean[["layer", "mean_delta", "std_delta", "n_examples"]].rename(
        columns={
            "mean_delta": "clean_mean_delta",
            "std_delta": "clean_std_delta",
            "n_examples": "clean_n_examples",
        }
    )
    noisy_renamed = noisy_summary[["layer", "mean_delta", "std_delta", "n_examples"]].rename(
        columns={
            "mean_delta": "noisy_mean_delta",
            "std_delta": "noisy_std_delta",
            "n_examples": "noisy_n_examples",
        }
    )

    merged = pd.merge(clean_renamed, noisy_renamed, on="layer", how="outer").sort_values("layer")
    merged["delta_gap"] = merged["noisy_mean_delta"] - merged["clean_mean_delta"]

    # Rank orderings
    merged["clean_rank"] = merged["clean_mean_delta"].rank(ascending=False, method="min").astype(int)
    merged["noisy_rank"] = merged["noisy_mean_delta"].rank(ascending=False, method="min").astype(int)

    log(f"[compare] Merged comparison table has {len(merged)} rows")
    return merged


# ---------------------------------------------------------------------------
# Plotting — overlay figure
# ---------------------------------------------------------------------------

def plot_overlay(
    comparison_df: pd.DataFrame,
    output_path: str,
    metric: str,
    n_clean: int,
    n_noisy: int,
):
    """
    Plot clean (A→C) and noisy (B→D) layer-level Δℓ curves as an overlay
    with ±1 std bands.  This is the primary RQ2 figure.
    """
    if comparison_df.empty:
        log("[plot] WARNING: no data to plot")
        return

    t0 = time.time()
    log(f"[plot] Generating overlay figure at {output_path}")

    layers = comparison_df["layer"].values

    fig, ax = plt.subplots(figsize=(12, 5.5))

    # --- Clean curve ---
    clean_mean = comparison_df["clean_mean_delta"].values
    clean_std = comparison_df["clean_std_delta"].fillna(0).values
    ax.plot(
        layers, clean_mean,
        marker="o", markersize=4, linewidth=1.5,
        color="#2c7bb6", label=f"Clean (A\u2192C, n={n_clean})",
    )
    ax.fill_between(
        layers, clean_mean - clean_std, clean_mean + clean_std,
        alpha=0.15, color="#2c7bb6",
    )

    # --- Noisy curve ---
    noisy_mean = comparison_df["noisy_mean_delta"].values
    noisy_std = comparison_df["noisy_std_delta"].fillna(0).values
    ax.plot(
        layers, noisy_mean,
        marker="s", markersize=4, linewidth=1.5,
        color="#d7191c", label=f"Noisy (B\u2192D, n={n_noisy})",
    )
    ax.fill_between(
        layers, noisy_mean - noisy_std, noisy_mean + noisy_std,
        alpha=0.15, color="#d7191c",
    )

    # --- Zero line ---
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)

    ax.set_xlabel("Layer", fontsize=12)
    ylabel = "Mean Causal Mediation Effect (\u0394\u2113, logit)" if metric == "logit" else "Mean Causal Mediation Effect (\u0394\u2113, prob)"
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(
        "Clean vs. Noisy Layer-Level Causal Mediation Effect\n"
        "Clean: Direct (A) \u2192 Structured (C) | Noisy: Direct (B) \u2192 Structured (D)",
        fontsize=13,
    )
    ax.legend(fontsize=10, loc="upper left")

    # Tick every 2 layers for readability
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
    return model_name.split("/")[-1].lower()


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Phase 3c: Cross-condition layer-level causal mediation comparison "
            "(clean vs. noisy) for RQ2"
        )
    )
    parser.add_argument("--dataset", type=str, default=None,
                        help="Path to dataset.json "
                             "(default: dataset/processed/<model-slug>/dataset.json)")
    parser.add_argument("--eval-results", type=str, default=None,
                        help="Path to evaluation_results.csv from Phase 2 "
                             "(default: results/phase_2_behaviour/<model-slug>/evaluation_results.csv)")
    parser.add_argument("--clean-summary", type=str, default=None,
                        help="Path to layer_patch_summary.csv from Phase 3a "
                             "(default: results/phase_3a_layer_patching/<model-slug>/layer_patch_summary.csv)")
    parser.add_argument("--layer-output-dir", type=str, default=None,
                        help="Directory for noisy layer patch CSV files "
                             "(default: results/phase_3a_layer_patching/<model-slug>/)")
    parser.add_argument("--cross-output-dir", type=str, default=None,
                        help="Directory for cross-condition comparison CSV "
                             "(default: results/phase_3c_cross_condition/<model-slug>/)")
    parser.add_argument("--contrast-output-dir", type=str, default=None,
                        help="Directory for noisy_contrast_examples.json "
                             "(default: dataset/processed/<model-slug>/)")
    parser.add_argument("--figure-dir", type=str, default=None,
                        help="Directory for overlay figure output "
                             "(default: figures/phase_3a_layer_patching/<model-slug>/)")
    parser.add_argument("--model", type=str, default="EleutherAI/pythia-2.8b",
                        help="HuggingFace model name for HookedTransformer")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda or cpu)")
    parser.add_argument("--max-examples", type=int, default=None,
                        help="Limit to first N noisy contrast examples (for debugging)")
    parser.add_argument("--hook-name", type=str,
                        default="blocks.{layer}.hook_resid_post",
                        help="Hook name template with {layer} placeholder")
    parser.add_argument("--patch-scope", type=str, default="final_token",
                        choices=["final_token", "full"],
                        help=(
                            "Which positions to patch: 'final_token' replaces only "
                            "the last position (Phase 3a default); 'full' replaces "
                            "all positions.  MUST match the Phase 3a clean run."
                        ))
    parser.add_argument("--metric", type=str, default="logit",
                        choices=["logit", "prob"],
                        help="Score metric: 'logit' (default) or 'prob'")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-example details and stage timings")
    parser.add_argument("--layer-log-interval", type=int, default=4,
                        help="Print layer progress every N layers (default: 4)")
    args = parser.parse_args()

    slug = _model_slug(args.model)
    dataset_path        = args.dataset           or f"dataset/processed/{slug}/dataset.json"
    eval_results_path   = args.eval_results      or f"results/phase_2_behaviour/{slug}/evaluation_results.csv"
    clean_summary_path  = args.clean_summary     or f"results/phase_3a_layer_patching/{slug}/layer_patch_summary.csv"
    layer_out_dir_path  = args.layer_output_dir  or f"results/phase_3a_layer_patching/{slug}"
    cross_out_dir_path  = args.cross_output_dir  or f"results/phase_3c_cross_condition/{slug}"
    contrast_out_path   = args.contrast_output_dir or f"dataset/processed/{slug}"
    fig_dir_path        = args.figure_dir        or f"figures/phase_3a_layer_patching/{slug}"

    overall_t0 = time.time()

    layer_out_dir = Path(layer_out_dir_path)
    layer_out_dir.mkdir(parents=True, exist_ok=True)
    cross_out_dir = Path(cross_out_dir_path)
    cross_out_dir.mkdir(parents=True, exist_ok=True)
    contrast_out_dir = Path(contrast_out_path)
    contrast_out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = Path(fig_dir_path)
    fig_dir.mkdir(parents=True, exist_ok=True)

    log(f"[main] Layer output directory:    {layer_out_dir.resolve()}")
    log(f"[main] Cross output directory:    {cross_out_dir.resolve()}")
    log(f"[main] Contrast output directory: {contrast_out_dir.resolve()}")
    log(f"[main] Figure directory:          {fig_dir.resolve()}")
    log(f"[main] Starting Phase 3c cross-condition layer-level causal mediation comparison")
    log(f"[main] patch_scope={args.patch_scope}  metric={args.metric}")

    # ---- Pre-flight: verify metric consistency with clean summary ----
    verify_clean_summary_metric(clean_summary_path, args.metric)

    # ---- Step 1: Identify noisy contrast examples ----
    log("\n" + "=" * 70)
    log("Step 1: Identifying noisy contrast examples (B wrong \u2227 D correct)")
    log("=" * 70)

    contrasts = identify_noisy_contrasts(dataset_path, eval_results_path)

    # Save noisy contrasts for reproducibility
    contrast_path = contrast_out_dir / "noisy_contrast_examples.json"
    with open(contrast_path, "w", encoding="utf-8") as f:
        json.dump(contrasts, f, indent=2, ensure_ascii=False)
    log(f"[save] {contrast_path} ({len(contrasts)} noisy contrasts)")

    if args.max_examples is not None:
        contrasts = contrasts[:args.max_examples]
        log(f"[main] Limiting to first {args.max_examples} noisy contrast examples")

    if len(contrasts) == 0:
        log("[main] ERROR: no noisy contrast examples found. Cannot proceed.")
        log("  \u2192 Check that evaluation_results.csv contains Cell B and Cell D results.")
        sys.exit(1)

    # ---- Step 2: Load model ----
    model = load_model(args.model, args.device)

    n_layers = model.cfg.n_layers
    n_examples = len(contrasts)
    n_total = n_examples * n_layers

    log("\n" + "=" * 70)
    log("Phase 3c: Noisy Layer-Level Causal Mediation (B \u2192 D)")
    log(f"  noisy contrast examples:  {n_examples}")
    log(f"  layers per example:       {n_layers}")
    log(f"  total patch runs:         {n_total}")
    log(f"  hook:                     {args.hook_name}")
    log(f"  patch_scope:              {args.patch_scope}")
    log(f"  metric:                   {args.metric}")
    log(f"  device:                   {args.device}")
    log(f"  layer log interval:       {args.layer_log_interval}")
    log("=" * 70 + "\n")

    # ---- Step 3: Run noisy layer sweeps ----
    all_rows: list[dict] = []
    run_t0 = time.time()
    n_valid = 0
    n_skipped = 0

    for i, example in enumerate(contrasts):
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
            patch_scope=args.patch_scope,
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
    log(f"\n[main] Noisy patching loop finished in {format_seconds(elapsed)} "
        f"({n_valid} valid, {n_skipped} skipped)")

    # ---- Step 4: Save noisy results ----
    results_df = pd.DataFrame(all_rows)

    detail_path = layer_out_dir / "noisy_layer_patch_results.csv"
    t0 = time.time()
    results_df.to_csv(detail_path, index=False, encoding="utf-8")
    log(f"[save] {detail_path} ({len(results_df)} rows) in {format_seconds(time.time() - t0)}")

    noisy_summary = aggregate_layer_results(results_df, args.hook_name, args.metric)
    noisy_summary_path = layer_out_dir / "noisy_layer_patch_summary.csv"
    t0 = time.time()
    noisy_summary.to_csv(noisy_summary_path, index=False, encoding="utf-8")
    log(f"[save] {noisy_summary_path} in {format_seconds(time.time() - t0)}")

    # ---- Step 5: Cross-condition comparison ----
    log("\n" + "=" * 70)
    log("Step 5: Cross-condition layer-level comparison (clean vs. noisy)")
    log("=" * 70)

    comparison_df = build_cross_condition_comparison(clean_summary_path, noisy_summary)

    if not comparison_df.empty:
        comp_path = cross_out_dir / "cross_condition_layer_comparison.csv"
        t0 = time.time()
        comparison_df.to_csv(comp_path, index=False, encoding="utf-8")
        log(f"[save] {comp_path} in {format_seconds(time.time() - t0)}")

        # Load clean n_examples for the figure label
        clean_summary = pd.read_csv(clean_summary_path)
        n_clean = int(clean_summary["n_examples"].iloc[0]) if not clean_summary.empty else 0

        fig_path = fig_dir / "clean_vs_noisy_layer_patch_overlay.png"
        plot_overlay(comparison_df, str(fig_path), args.metric, n_clean, n_valid)

    # ---- Console summary ----
    log("\n" + "=" * 70)
    log("PHASE 3c: CROSS-CONDITION LAYER-LEVEL COMPARISON SUMMARY")
    log("=" * 70)
    log(f"  Noisy contrast examples (B wrong \u2227 D correct): {len(contrasts)}")
    log(f"  Valid for patching:   {n_valid}")
    log(f"  Skipped:              {n_skipped}")
    log(f"  Patch scope:          {args.patch_scope}")

    if not noisy_summary.empty:
        top_k = 5
        top_noisy = noisy_summary.nlargest(top_k, "mean_delta")
        log(f"\n  Top {top_k} noisy layers by mean \u0394\u2113 ({args.metric}):")
        for _, row in top_noisy.iterrows():
            log(
                f"    Layer {int(row['layer']):2d}: "
                f"mean_\u0394={row['mean_delta']:+.4f} "
                f"std={row['std_delta']:.4f} "
                f"n={int(row['n_examples'])}"
            )

    if not comparison_df.empty:
        peak_clean_layer = int(comparison_df.loc[comparison_df["clean_mean_delta"].idxmax(), "layer"])
        peak_noisy_layer = int(comparison_df.loc[comparison_df["noisy_mean_delta"].idxmax(), "layer"])
        log(f"\n  Peak clean layer: {peak_clean_layer}")
        log(f"  Peak noisy layer: {peak_noisy_layer}")

        # Largest positive delta_gap (noisy > clean)
        top_pos_gap = comparison_df.nlargest(3, "delta_gap")
        log(f"\n  Largest positive delta_gap (noisy > clean):")
        for _, row in top_pos_gap.iterrows():
            log(
                f"    Layer {int(row['layer']):2d}: "
                f"gap={row['delta_gap']:+.4f} "
                f"(clean={row['clean_mean_delta']:+.4f}, noisy={row['noisy_mean_delta']:+.4f})"
            )

        # Largest negative delta_gap (clean > noisy)
        top_neg_gap = comparison_df.nsmallest(3, "delta_gap")
        log(f"\n  Largest negative delta_gap (clean > noisy):")
        for _, row in top_neg_gap.iterrows():
            log(
                f"    Layer {int(row['layer']):2d}: "
                f"gap={row['delta_gap']:+.4f} "
                f"(clean={row['clean_mean_delta']:+.4f}, noisy={row['noisy_mean_delta']:+.4f})"
            )

    total_elapsed = time.time() - overall_t0
    log("\n" + "=" * 70)
    if n_valid > 0:
        log(f"Phase 3c COMPLETE. {n_valid} valid noisy contrast examples analysed.")
        log(
            "Review the overlay figure (clean_vs_noisy_layer_patch_overlay.png) "
            "to determine whether a targeted noisy component-level follow-up is "
            "warranted before proceeding to Phase 4a."
        )
    else:
        log("Phase 3c FAILED: no valid examples. Check token alignment.")
    log(f"Total wall time: {format_seconds(total_elapsed)}")
    log("=" * 70)


if __name__ == "__main__":
    main()