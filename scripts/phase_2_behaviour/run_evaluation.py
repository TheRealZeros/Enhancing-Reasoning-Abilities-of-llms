#!/usr/bin/env python3
"""
Phase 2: Behavioural Evaluation
================================
Runs Pythia-2.8B on all 5 prompt cells (A–E) for every dataset example,
computes exact-match accuracy, and identifies contrast examples
(Cell A wrong ∧ Cell C correct) for downstream activation patching.

Usage:
    python scripts/phase_2_behaviour/run_evaluation.py
    python scripts/phase_2_behaviour/run_evaluation.py --max-new-tokens 20 --device cuda

Outputs:
    results/phase_2_behaviour/evaluation_results.csv   – one row per (example, cell)
    results/phase_2_behaviour/accuracy_summary.csv     – per-cell accuracy table
    dataset/processed/contrast_examples.json           – contrast pairs for Phase 3
"""

import argparse
import gc
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

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
# Dataset loading
# ---------------------------------------------------------------------------

CELL_NAMES = ["A", "B", "C", "D", "E"]
REQUIRED_FIELDS = {"id", "domain", "answer", "cells"}


def load_dataset(path: str) -> list[dict]:
    """Load and validate the dataset JSON."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list) or len(data) == 0:
        raise ValueError("Dataset must be a non-empty JSON list.")

    # Validate structure of every example
    for i, ex in enumerate(data):
        missing = REQUIRED_FIELDS - set(ex.keys())
        if missing:
            raise ValueError(
                f"Example index {i} (id={ex.get('id', '?')}) missing fields: {missing}"
            )
        cells = ex["cells"]
        if not isinstance(cells, dict):
            raise ValueError(f"Example {ex['id']}: 'cells' must be a dict.")
        missing_cells = set(CELL_NAMES) - set(cells.keys())
        if missing_cells:
            raise ValueError(
                f"Example {ex['id']}: missing cells {missing_cells}"
            )

    print(f"[dataset] Loaded {len(data)} examples from {path}")
    return data


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_model(model_name: str, device: str):
    """Load model via TransformerLens HookedTransformer."""
    from transformer_lens import HookedTransformer

    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    print(f"[model] Loading {model_name} on {device} ({dtype}) ...")
    t0 = time.time()
    model = HookedTransformer.from_pretrained(
        model_name,
        device=device,
        dtype=dtype,
    )
    model.eval()
    elapsed = time.time() - t0
    print(f"[model] Loaded in {elapsed:.1f}s  |  "
          f"n_layers={model.cfg.n_layers}  n_heads={model.cfg.n_heads}  "
          f"d_model={model.cfg.d_model}")
    return model


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def generate_answer(
    model,
    prompt: str,
    max_new_tokens: int,
    device: str,
) -> tuple[str, int, int]:
    """
    Run greedy generation on *prompt* and return:
        (raw_continuation, input_token_count, output_token_count)
    """
    tokenizer = model.tokenizer
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    input_len = input_ids.shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    # output_ids may be a tensor of shape (1, seq_len) or (seq_len,)
    if output_ids.dim() == 2:
        output_ids = output_ids[0]

    new_ids = output_ids[input_len:]
    raw_continuation = tokenizer.decode(new_ids, skip_special_tokens=True)
    return raw_continuation, input_len, len(new_ids)


# ---------------------------------------------------------------------------
# Answer normalisation and scoring
# ---------------------------------------------------------------------------

# Prefixes the model might produce before the actual answer
_ANSWER_PREFIXES = re.compile(
    r"^(the answer is|answer:)\s*", re.IGNORECASE
)


def normalise_answer(text: str) -> str:
    """Light normalisation: first line, strip, lowercase, remove common prefixes."""
    # Take only the first non-empty line (model may ramble)
    for line in text.split("\n"):
        line = line.strip()
        if line:
            text = line
            break
    else:
        return ""

    text = text.strip()
    text = text.lower()
    # Remove trailing punctuation
    text = text.rstrip(".,;:!?")
    # Remove leading answer prefixes
    text = _ANSWER_PREFIXES.sub("", text).strip()
    return text


def is_correct(pred_normalised: str, gold: str) -> bool:
    """Exact match after normalisation of both sides."""
    gold_norm = gold.strip().lower().rstrip(".,;:!?")
    return pred_normalised == gold_norm


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------


def evaluate_dataset(
    dataset: list[dict],
    model,
    max_new_tokens: int,
    device: str,
    max_examples: int | None = None,
) -> pd.DataFrame:
    """
    Run all cells for every example. Returns a DataFrame with one row per
    (example, cell) pair.
    """
    if max_examples is not None:
        dataset = dataset[:max_examples]
        print(f"[eval] Limiting to first {max_examples} examples.")

    rows: list[dict] = []
    errors: list[str] = []
    total = len(dataset) * len(CELL_NAMES)
    done = 0

    for ex in dataset:
        ex_id = ex["id"]
        domain = ex["domain"]
        gold = ex["answer"]

        for cell in CELL_NAMES:
            cell_data = ex["cells"][cell]
            # Materialise the prompt from the stored schema
            prompt = materialise_prompt(cell_data, model.tokenizer)
            try:
                raw, in_tok, out_tok = generate_answer(
                    model, prompt, max_new_tokens, device
                )
                norm = normalise_answer(raw)
                correct = is_correct(norm, gold)

                rows.append({
                    "example_id": ex_id,
                    "domain": domain,
                    "cell": cell,
                    "generated_answer_raw": raw,
                    "generated_answer_normalised": norm,
                    "gold_answer": gold,
                    "correct": correct,
                    "error": False,
                    "input_token_count": in_tok,
                    "output_token_count": out_tok,
                })
            except Exception as exc:
                errors.append(f"{ex_id}/{cell}: {exc}")
                rows.append({
                    "example_id": ex_id,
                    "domain": domain,
                    "cell": cell,
                    "generated_answer_raw": f"ERROR: {exc}",
                    "generated_answer_normalised": "",
                    "gold_answer": gold,
                    "correct": False,
                    "error": True,
                    "input_token_count": -1,
                    "output_token_count": -1,
                })

            done += 1
            if done % 50 == 0 or done == total:
                print(f"  [{done}/{total}] completed")

            # VRAM hygiene (only when running on CUDA)
            if device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    if errors:
        print(f"\n[eval] WARNING: {len(errors)} errors encountered:")
        for e in errors[:20]:
            print(f"  - {e}")
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more")

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Summaries and contrast detection
# ---------------------------------------------------------------------------


def build_accuracy_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per-cell accuracy table."""
    summary = (
        df.groupby("cell")
        .agg(total_examples=("correct", "size"),
             num_correct=("correct", "sum"))
        .reset_index()
    )
    summary["accuracy"] = summary["num_correct"] / summary["total_examples"]
    # Ensure consistent cell ordering A–E
    summary["cell"] = pd.Categorical(summary["cell"], categories=CELL_NAMES, ordered=True)
    summary = summary.sort_values("cell").reset_index(drop=True)
    return summary


def find_contrast_examples(
    df: pd.DataFrame,
    dataset_by_id: dict[str, dict],
) -> list[dict]:
    """
    Contrast examples: same example where Cell A is wrong AND Cell C is correct.
    Returns a list of dicts with the cell schema (prompt + metadata) for both cells.
    """
    # Pivot to one row per example
    cell_a = df[df["cell"] == "A"].set_index("example_id")
    cell_c = df[df["cell"] == "C"].set_index("example_id")

    common_ids = cell_a.index.intersection(cell_c.index)

    contrasts = []
    for eid in common_ids:
        a_row = cell_a.loc[eid]
        c_row = cell_c.loc[eid]
        if (not a_row["correct"]) and c_row["correct"]:
            ex = dataset_by_id[eid]
            # Store the clean cell schema (dict with prompt + metadata),
            # NOT the materialised prompt with EOS padding.
            cell_a_data = ex["cells"]["A"]
            cell_c_data = ex["cells"]["C"]

            # Ensure cell data is a dict (handle legacy string format)
            if isinstance(cell_a_data, str):
                cell_a_data = {"prompt": cell_a_data, "prefix_eos_pad": 0}
            if isinstance(cell_c_data, str):
                cell_c_data = {"prompt": cell_c_data, "prefix_eos_pad": 0}

            contrasts.append({
                "example_id": eid,
                "domain": ex["domain"],
                "gold_answer": ex["answer"],
                "cell_A": {
                    **cell_a_data,
                    "generated_answer_raw": a_row["generated_answer_raw"],
                    "generated_answer_normalised": a_row["generated_answer_normalised"],
                    "correct": False,
                },
                "cell_C": {
                    **cell_c_data,
                    "generated_answer_raw": c_row["generated_answer_raw"],
                    "generated_answer_normalised": c_row["generated_answer_normalised"],
                    "correct": True,
                },
            })

    return contrasts


# ---------------------------------------------------------------------------
# CLI and main
# ---------------------------------------------------------------------------


def _model_slug(model_name: str) -> str:
    """'EleutherAI/pythia-2.8b' -> 'pythia-2.8b', 'gpt2-large' -> 'gpt2-large'"""
    return model_name.split("/")[-1].lower()


def main():
    parser = argparse.ArgumentParser(
        description="Phase 2: Behavioural evaluation on 5-cell dataset"
    )
    parser.add_argument("--dataset", type=str,
                        default=None,
                        help="Path to dataset.json "
                             "(default: dataset/processed/<model-slug>/dataset.json)")
    parser.add_argument("--output-dir", type=str,
                        default=None,
                        help="Directory for output CSV files "
                             "(default: results/phase_2_behaviour/<model-slug>/)")
    parser.add_argument("--model", type=str, default="EleutherAI/pythia-2.8b",
                        help="HuggingFace model name for HookedTransformer")
    parser.add_argument("--max-new-tokens", type=int, default=16,
                        help="Max tokens to generate per prompt")
    parser.add_argument("--max-examples", type=int, default=None,
                        help="Limit to first N examples (for debugging)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run on (cuda or cpu)")
    args = parser.parse_args()

    # ---- Resolve model-namespaced defaults ----
    slug = _model_slug(args.model)
    dataset_path = args.dataset or f"dataset/processed/{slug}/dataset.json"
    out_dir_path = args.output_dir or f"results/phase_2_behaviour/{slug}"

    # ---- Setup ----
    out_dir = Path(out_dir_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load data ----
    dataset = load_dataset(dataset_path)
    dataset_by_id = {ex["id"]: ex for ex in dataset}

    # ---- Load model ----
    model = load_model(args.model, args.device)

    # ---- Run evaluation ----
    n_examples = min(len(dataset), args.max_examples) if args.max_examples else len(dataset)
    n_inferences = n_examples * len(CELL_NAMES)
    print(f"\n{'='*60}")
    print(f"Running evaluation: {n_examples} examples × {len(CELL_NAMES)} cells "
          f"= {n_inferences} inferences")
    print(f"max_new_tokens={args.max_new_tokens}  device={args.device}")
    print(f"{'='*60}\n")

    t0 = time.time()
    results_df = evaluate_dataset(
        dataset, model, args.max_new_tokens, args.device, args.max_examples
    )
    elapsed = time.time() - t0
    print(f"\n[eval] Finished in {elapsed:.1f}s")

    # ---- Save detailed results ----
    eval_path = out_dir / "evaluation_results.csv"
    results_df.to_csv(eval_path, index=False, encoding="utf-8")
    print(f"[save] {eval_path}  ({len(results_df)} rows)")

    # ---- Accuracy summary ----
    summary_df = build_accuracy_summary(results_df)
    summary_path = out_dir / "accuracy_summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8")
    print(f"[save] {summary_path}")

    # ---- Contrast examples ----
    contrasts = find_contrast_examples(results_df, dataset_by_id)
    # Route contrast examples to the model-specific processed dir so downstream
    # scripts find them alongside the model's dataset.json.
    dataset_dir = Path(dataset_path).parent
    dataset_dir.mkdir(parents=True, exist_ok=True)
    contrast_path = dataset_dir / "contrast_examples.json"
    with open(contrast_path, "w", encoding="utf-8") as f:
        json.dump(contrasts, f, indent=2, ensure_ascii=False)
    print(f"[save] {contrast_path}  ({len(contrasts)} contrast examples)")

    # ---- Console summary ----
    print(f"\n{'='*60}")
    print("ACCURACY SUMMARY")
    print(f"{'='*60}")
    for _, row in summary_df.iterrows():
        bar = "#" * int(row["accuracy"] * 40)
        print(f"  Cell {row['cell']}: {row['num_correct']:3.0f}/{row['total_examples']:3.0f} "
              f"= {row['accuracy']:.1%}  {bar}")

    n_contrast = len(contrasts)
    status = "PASS" if n_contrast >= 20 else "BELOW TARGET"
    print(f"\nContrast examples (A wrong ^ C correct): {n_contrast}  [{status}]")
    if n_contrast < 20:
        print("  → Consider expanding dataset or loosening contrast criterion "
              "(e.g., higher p(correct) under C even if not EM-correct).")

    # Error count
    n_errors = int(results_df["error"].sum())
    if n_errors > 0:
        print(f"\nRuntime errors: {n_errors}/{len(results_df)} inferences failed")
    else:
        print(f"\nRuntime errors: 0")

    # Domain breakdown for contrasts
    if contrasts:
        from collections import Counter
        dom_counts = Counter(c["domain"] for c in contrasts)
        print(f"  Domain breakdown: {dict(dom_counts)}")

if __name__ == "__main__":
    main()