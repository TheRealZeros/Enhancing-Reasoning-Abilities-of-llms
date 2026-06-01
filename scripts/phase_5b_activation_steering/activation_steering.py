#!/usr/bin/env python3
"""
Phase 5b: Activation Steering
=============================

Final average activation steering test for whether the calibrated late-layer donor-minus-source signal identified by
activation patching can be reused as a held-out intervention.

Core method:

    steering_vector = mean(donor_activation - source_activation)

The vector is computed on train examples and injected into source-condition
runs on held-out test examples. Injection is restricted to the final prompt
token position, matching the Phase 3 final-token scoring setup.

First implementation scope:
    - hook: resid_post only
    - score-only metrics by default
    - optional qualitative generation via --generate-examples
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

try:
    from scripts.utils.contrast_config import contrast_path_for, model_file_prefix, output_prefix_for
except ModuleNotFoundError:
    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))
    from scripts.utils.contrast_config import contrast_path_for, model_file_prefix, output_prefix_for


np = None
pd = None
plt = None
torch = None

SUPPORTED_HOOKS = {
    "resid_post": "blocks.{layer}.hook_resid_post",
}

TOKEN_POSITION_LABEL = "final_prompt_token"

BASE_RESULT_FIELDS = [
    "model",
    "example_id",
    "domain",
    "source_cell",
    "donor_cell",
    "layer",
    "hook",
    "token_position",
    "control",
    "alpha",
    "split",
    "gold_answer",
    "gold_token_id",
    "baseline_gold_logit",
    "steered_gold_logit",
    "delta_gold_logit",
    "baseline_gold_rank",
    "steered_gold_rank",
    "delta_gold_rank",
    "baseline_top1",
    "steered_top1",
]

GENERATION_RESULT_FIELDS = [
    "baseline_generated_answer",
    "steered_generated_answer",
    "baseline_exact_match",
    "steered_exact_match",
    "baseline_contains_answer",
    "steered_contains_answer",
    "repetition_detected",
    "malformed_output",
]

SUMMARY_FIELDS = [
    "model",
    "source_cell",
    "donor_cell",
    "contrast_file",
    "layer",
    "hook",
    "token_position",
    "control",
    "n_train",
    "n_test",
    "vector_l2_norm",
    "alpha",
    "mean_delta_gold_logit",
    "median_delta_gold_logit",
    "mean_delta_gold_rank",
    "baseline_top1_rate",
    "steered_top1_rate",
    "top1_improvement",
    "best_alpha_by_delta_logit",
    "best_alpha_by_top1_improvement",
]


def ensure_runtime_imports() -> None:
    """Import ML/plotting dependencies only when a real run needs them."""
    global np, pd, plt, torch
    if torch is not None:
        return

    import matplotlib
    import numpy as _np
    import pandas as _pd
    import torch as _torch

    matplotlib.use("Agg")
    import matplotlib.pyplot as _plt

    np = _np
    pd = _pd
    plt = _plt
    torch = _torch


def log(message: str) -> None:
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def format_seconds(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes}m {secs:.1f}s"


def model_slug(model_name: str) -> str:
    return model_name.split("/")[-1].lower()


def resolve_dtype(dtype_name: str, device: str) -> torch.dtype:
    if dtype_name == "auto":
        return torch.float16 if device.startswith("cuda") else torch.float32

    mapping = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    try:
        return mapping[dtype_name]
    except KeyError as exc:
        supported = ", ".join(["auto", *mapping.keys()])
        raise ValueError(f"Unsupported dtype {dtype_name!r}. Supported: {supported}") from exc


def cuda_mem_string(device: str) -> str:
    if not device.startswith("cuda") or not torch.cuda.is_available():
        return "cuda_mem=n/a"
    try:
        alloc = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        max_alloc = torch.cuda.max_memory_allocated() / (1024**3)
        return (
            f"cuda_mem alloc={alloc:.2f}GB "
            f"reserved={reserved:.2f}GB "
            f"max_alloc={max_alloc:.2f}GB"
        )
    except Exception as exc:
        return f"cuda_mem=unavailable ({exc})"


def reset_cuda_peak_memory_stats(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass


def clear_memory(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def materialise_prompt(cell_dict: Any, tokenizer) -> str:
    """
    Reconstruct the exact runnable model input from the stored cell schema.

    This mirrors the Phase 1/2/3/4 helper, including prefix EOS padding and
    optional inline filler before the final answer marker.
    """
    if isinstance(cell_dict, str):
        return cell_dict

    eos = tokenizer.eos_token or ""
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


def load_json_list(path: Path, label: str) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"{label} file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or not data:
        raise ValueError(f"{label} file must be a non-empty JSON list: {path}")
    valid = [row for row in data if isinstance(row, dict)]
    if len(valid) != len(data):
        log(f"[load] WARNING: {label} skipped {len(data) - len(valid)} non-dict rows")
    return valid


def load_dataset_index(path: Path) -> dict[str, dict]:
    data = load_json_list(path, "dataset")
    by_id = {}
    for ex in data:
        eid = ex.get("id") or ex.get("example_id")
        if eid:
            by_id[str(eid)] = ex
    return by_id


def get_cell_schema(
    contrast_example: dict,
    dataset_index: dict[str, dict],
    cell: str,
) -> Any | None:
    direct_key = f"cell_{cell}"
    if direct_key in contrast_example:
        return contrast_example[direct_key]

    eid = str(contrast_example.get("example_id", ""))
    dataset_example = dataset_index.get(eid)
    if dataset_example and isinstance(dataset_example.get("cells"), dict):
        return dataset_example["cells"].get(cell)

    return None


def validate_contrast_examples(
    examples: list[dict],
    dataset_index: dict[str, dict],
    source_cell: str,
    donor_cell: str,
) -> list[dict]:
    valid = []
    for idx, ex in enumerate(examples):
        eid = ex.get("example_id")
        gold = ex.get("gold_answer") or ex.get("answer")
        if not eid or not isinstance(gold, str) or not gold.strip():
            log(f"[load] WARNING: skipping contrast index {idx}: missing example_id or gold_answer")
            continue

        source_schema = get_cell_schema(ex, dataset_index, source_cell)
        donor_schema = get_cell_schema(ex, dataset_index, donor_cell)
        if source_schema is None or donor_schema is None:
            log(
                f"[load] WARNING: skipping {eid}: missing Cell {source_cell} "
                f"or Cell {donor_cell}"
            )
            continue
        valid.append(ex)
    return valid


def resolve_prompt_pair(
    ex: dict,
    dataset_index: dict[str, dict],
    source_cell: str,
    donor_cell: str,
    tokenizer,
) -> dict[str, Any] | None:
    eid = str(ex["example_id"])
    source_schema = get_cell_schema(ex, dataset_index, source_cell)
    donor_schema = get_cell_schema(ex, dataset_index, donor_cell)
    if source_schema is None or donor_schema is None:
        return None

    source_prompt = materialise_prompt(source_schema, tokenizer)
    donor_prompt = materialise_prompt(donor_schema, tokenizer)
    if not source_prompt or not donor_prompt:
        return None

    gold = ex.get("gold_answer") or ex.get("answer")
    return {
        "example_id": eid,
        "domain": ex.get("domain", ""),
        "gold_answer": gold,
        "source_prompt": source_prompt,
        "donor_prompt": donor_prompt,
    }


def split_examples(
    examples: list[dict],
    train_frac: float,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    if not 0.0 < train_frac < 1.0:
        raise ValueError("--train-frac must be between 0 and 1")

    indices = list(range(len(examples)))
    rng = random.Random(seed)
    rng.shuffle(indices)

    if len(indices) == 1:
        n_train = 1
    else:
        n_train = int(math.floor(len(indices) * train_frac))
        n_train = max(1, min(len(indices) - 1, n_train))

    train_indices = set(indices[:n_train])
    train = [examples[i] for i in range(len(examples)) if i in train_indices]
    test = [examples[i] for i in range(len(examples)) if i not in train_indices]
    return train, test


def hook_name_for(hook: str, layer: int) -> str:
    if hook not in SUPPORTED_HOOKS:
        supported = ", ".join(SUPPORTED_HOOKS)
        raise ValueError(
            f"Unsupported hook {hook!r}. Phase 5b currently supports: {supported}"
        )
    return SUPPORTED_HOOKS[hook].format(layer=layer)


def load_model(model_name: str, device: str, dtype_name: str):
    from transformer_lens import HookedTransformer

    dtype = resolve_dtype(dtype_name, device)
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
        f"{cuda_mem_string(device)}"
    )
    return model


def tokens_for_prompt(model, prompt: str, device: str) -> torch.Tensor:
    tokens = model.to_tokens(prompt)
    return tokens.to(device)


def token_lengths_match(model, source_prompt: str, donor_prompt: str, device: str) -> tuple[bool, int, int]:
    source_len = tokens_for_prompt(model, source_prompt, device).shape[1]
    donor_len = tokens_for_prompt(model, donor_prompt, device).shape[1]
    return source_len == donor_len, source_len, donor_len


def get_gold_first_token_id(model, gold_answer: str) -> tuple[int, str]:
    spaced = " " + gold_answer.strip()
    token_ids = model.tokenizer.encode(spaced, add_special_tokens=False)
    if not token_ids:
        raise ValueError(f"Gold answer {gold_answer!r} tokenises to zero tokens")
    token_id = int(token_ids[0])
    token_str = model.tokenizer.decode([token_id])
    return token_id, token_str


def extract_final_activation(
    model,
    tokens: torch.Tensor,
    hook_name: str,
) -> torch.Tensor:
    with torch.no_grad():
        _, cache = model.run_with_cache(
            tokens,
            names_filter=lambda name: name == hook_name,
        )
    activation = cache[hook_name][0, -1, :].detach().float().cpu()
    del cache
    return activation


def compute_steering_vector(
    model,
    train_examples: list[dict],
    dataset_index: dict[str, dict],
    source_cell: str,
    donor_cell: str,
    hook_name: str,
    device: str,
) -> tuple[torch.Tensor, list[str]]:
    differences = []
    skipped = []
    t0 = time.time()

    for idx, ex in enumerate(train_examples):
        resolved = resolve_prompt_pair(ex, dataset_index, source_cell, donor_cell, model.tokenizer)
        if resolved is None:
            skipped.append(str(ex.get("example_id", f"index_{idx}")))
            continue

        eid = resolved["example_id"]
        aligned, source_len, donor_len = token_lengths_match(
            model,
            resolved["source_prompt"],
            resolved["donor_prompt"],
            device,
        )
        if not aligned:
            log(
                f"[train] WARNING: skipping {eid}: Cell {source_cell}={source_len} tokens, "
                f"Cell {donor_cell}={donor_len} tokens"
            )
            skipped.append(eid)
            continue

        source_tokens = tokens_for_prompt(model, resolved["source_prompt"], device)
        donor_tokens = tokens_for_prompt(model, resolved["donor_prompt"], device)
        source_activation = extract_final_activation(model, source_tokens, hook_name)
        donor_activation = extract_final_activation(model, donor_tokens, hook_name)
        differences.append(donor_activation - source_activation)

        del source_tokens, donor_tokens, source_activation, donor_activation
        clear_memory(device)

        if idx == 0 or (idx + 1) % 10 == 0 or idx == len(train_examples) - 1:
            log(
                f"[train] {idx + 1}/{len(train_examples)} examples processed | "
                f"valid_vectors={len(differences)} | {cuda_mem_string(device)}"
            )

    if not differences:
        raise RuntimeError("No valid train examples were available for steering vector computation")

    stacked = torch.stack(differences, dim=0).float()
    steering_vector = stacked.mean(dim=0).float()
    elapsed = time.time() - t0
    log(
        f"[train] Steering vector computed from {len(differences)} examples "
        f"in {format_seconds(elapsed)}"
    )
    return steering_vector, skipped


def make_random_matched_norm_vector(
    learned_vector: torch.Tensor,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    random_vector = torch.randn(
        learned_vector.shape,
        generator=generator,
        dtype=torch.float32,
    )
    learned_norm = learned_vector.float().norm()
    random_norm = random_vector.norm()
    if random_norm.item() == 0:
        return random_vector
    return random_vector * (learned_norm / random_norm)


def make_final_token_steering_hook(
    steering_vector: torch.Tensor,
    alpha: float,
    fixed_position: int | None = None,
):
    """
    Add alpha * steering_vector at the final prompt token only.

    For score-only runs fixed_position is None and the input contains only the
    prompt, so -1 is the final prompt token. For qualitative generation,
    fixed_position is set to the original prompt-final index so later decoding
    steps do not move the intervention onto generated tokens.
    """

    def hook_fn(activation, hook):
        if activation.ndim != 3:
            raise ValueError(
                f"Expected activation with shape [batch, position, d_model], got {tuple(activation.shape)}"
            )
        pos = fixed_position if fixed_position is not None else activation.shape[1] - 1
        if pos >= activation.shape[1]:
            return activation
        vector = steering_vector.to(device=activation.device, dtype=activation.dtype)
        updated = activation.clone()
        updated[:, pos, :] = updated[:, pos, :] + (float(alpha) * vector)
        return updated

    return hook_fn


def score_logits(logits: torch.Tensor, gold_token_id: int) -> dict[str, Any]:
    last_logits = logits[0, -1, :].float()
    gold_logit = float(last_logits[gold_token_id].item())
    gold_rank = int((last_logits > last_logits[gold_token_id]).sum().item())
    top1_id = int(torch.argmax(last_logits).item())
    return {
        "gold_logit": gold_logit,
        "gold_rank": gold_rank,
        "top1": int(top1_id == gold_token_id),
    }


def run_baseline_logits(model, tokens: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return model(tokens)


def run_steered_logits(
    model,
    tokens: torch.Tensor,
    hook_name: str,
    steering_vector: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    hook_fn = make_final_token_steering_hook(steering_vector, alpha)
    with torch.no_grad():
        return model.run_with_hooks(tokens, fwd_hooks=[(hook_name, hook_fn)])


_ANSWER_PREFIXES = re.compile(r"^(the answer is|answer:)\s*", re.IGNORECASE)


def normalise_answer(text: str) -> str:
    for line in (text or "").split("\n"):
        line = line.strip()
        if line:
            text = line
            break
    else:
        return ""

    text = text.strip().lower()
    text = text.rstrip(".,;:!?")
    text = _ANSWER_PREFIXES.sub("", text).strip()
    return text


def exact_match(generated: str, gold: str) -> bool:
    gold_norm = gold.strip().lower().rstrip(".,;:!?")
    return normalise_answer(generated) == gold_norm


def contains_answer(generated: str, gold: str) -> bool:
    gold_norm = normalise_answer(gold)
    generated_norm = normalise_answer(generated)
    return bool(gold_norm and gold_norm in generated_norm)


def detect_simple_repetition(text: str) -> bool:
    normalised = re.sub(r"\s+", " ", (text or "").strip().lower())
    if not normalised:
        return False

    lines = [line.strip().lower() for line in text.splitlines() if line.strip()]
    if len(lines) != len(set(lines)):
        return True

    words = normalised.split()
    if len(words) < 9:
        return False
    seen = set()
    for idx in range(len(words) - 2):
        tri = tuple(words[idx : idx + 3])
        if tri in seen:
            return True
        seen.add(tri)
    return False


def is_malformed_generation(text: str) -> bool:
    return normalise_answer(text) == ""


def greedy_generate(
    model,
    prompt: str,
    device: str,
    max_new_tokens: int,
    hook_name: str | None = None,
    steering_vector: torch.Tensor | None = None,
    alpha: float = 0.0,
) -> str:
    tokens = tokens_for_prompt(model, prompt, device)
    prompt_final_index = tokens.shape[1] - 1
    generated = tokens
    eos_id = model.tokenizer.eos_token_id
    new_ids = []

    for _ in range(max_new_tokens):
        with torch.no_grad():
            if hook_name is not None and steering_vector is not None:
                hook_fn = make_final_token_steering_hook(
                    steering_vector,
                    alpha,
                    fixed_position=prompt_final_index,
                )
                logits = model.run_with_hooks(generated, fwd_hooks=[(hook_name, hook_fn)])
            else:
                logits = model(generated)

        next_id = int(torch.argmax(logits[0, -1, :]).item())
        new_ids.append(next_id)
        next_tensor = torch.tensor([[next_id]], device=generated.device, dtype=generated.dtype)
        generated = torch.cat([generated, next_tensor], dim=1)
        if eos_id is not None and next_id == eos_id:
            break

    return model.tokenizer.decode(new_ids, skip_special_tokens=True)


def should_generate_for_row(generated_so_far: int, generation_limit: int) -> bool:
    return generated_so_far < generation_limit


def evaluate_test_examples(
    model,
    test_examples: list[dict],
    dataset_index: dict[str, dict],
    model_name: str,
    source_cell: str,
    donor_cell: str,
    layer: int,
    hook: str,
    hook_name: str,
    control: str,
    alphas: list[float],
    steering_vectors: list[tuple[str | None, torch.Tensor]],
    device: str,
    generate_examples: bool,
    generation_limit: int,
    max_new_tokens: int,
) -> tuple[list[dict], list[dict], list[str]]:
    rows = []
    qualitative_examples = []
    skipped = []
    generated_count = 0
    t0 = time.time()

    for idx, ex in enumerate(test_examples):
        resolved = resolve_prompt_pair(ex, dataset_index, source_cell, donor_cell, model.tokenizer)
        if resolved is None:
            skipped.append(str(ex.get("example_id", f"index_{idx}")))
            continue

        eid = resolved["example_id"]
        aligned, source_len, donor_len = token_lengths_match(
            model,
            resolved["source_prompt"],
            resolved["donor_prompt"],
            device,
        )
        if not aligned:
            log(
                f"[test] WARNING: skipping {eid}: Cell {source_cell}={source_len} tokens, "
                f"Cell {donor_cell}={donor_len} tokens"
            )
            skipped.append(eid)
            continue

        try:
            gold_token_id, _ = get_gold_first_token_id(model, resolved["gold_answer"])
        except ValueError as exc:
            log(f"[test] WARNING: skipping {eid}: {exc}")
            skipped.append(eid)
            continue

        source_tokens = tokens_for_prompt(model, resolved["source_prompt"], device)
        baseline_logits = run_baseline_logits(model, source_tokens)
        baseline = score_logits(baseline_logits, gold_token_id)

        baseline_generation = ""
        if generate_examples and should_generate_for_row(generated_count, generation_limit):
            baseline_generation = greedy_generate(
                model,
                resolved["source_prompt"],
                device,
                max_new_tokens=max_new_tokens,
            )

        for vector_label, vector in steering_vectors:
            random_seed_label = vector_label or ""
            for alpha in alphas:
                steered_logits = run_steered_logits(
                    model,
                    source_tokens,
                    hook_name,
                    vector,
                    alpha,
                )
                steered = score_logits(steered_logits, gold_token_id)

                row = {
                    "model": model_name,
                    "example_id": eid,
                    "domain": resolved.get("domain", ""),
                    "source_cell": source_cell,
                    "donor_cell": donor_cell,
                    "layer": layer,
                    "hook": hook,
                    "token_position": TOKEN_POSITION_LABEL,
                    "control": control,
                    "random_seed": random_seed_label,
                    "alpha": alpha,
                    "split": "test",
                    "gold_answer": resolved["gold_answer"],
                    "gold_token_id": gold_token_id,
                    "baseline_gold_logit": baseline["gold_logit"],
                    "steered_gold_logit": steered["gold_logit"],
                    "delta_gold_logit": steered["gold_logit"] - baseline["gold_logit"],
                    "baseline_gold_rank": baseline["gold_rank"],
                    "steered_gold_rank": steered["gold_rank"],
                    "delta_gold_rank": baseline["gold_rank"] - steered["gold_rank"],
                    "baseline_top1": baseline["top1"],
                    "steered_top1": steered["top1"],
                }

                if generate_examples:
                    if should_generate_for_row(generated_count, generation_limit):
                        steered_generation = greedy_generate(
                            model,
                            resolved["source_prompt"],
                            device,
                            max_new_tokens=max_new_tokens,
                            hook_name=hook_name,
                            steering_vector=vector,
                            alpha=alpha,
                        )
                        repetition = detect_simple_repetition(baseline_generation) or detect_simple_repetition(
                            steered_generation
                        )
                        malformed = is_malformed_generation(baseline_generation) or is_malformed_generation(
                            steered_generation
                        )
                        row.update(
                            {
                                "baseline_generated_answer": baseline_generation,
                                "steered_generated_answer": steered_generation,
                                "baseline_exact_match": int(
                                    exact_match(baseline_generation, resolved["gold_answer"])
                                ),
                                "steered_exact_match": int(
                                    exact_match(steered_generation, resolved["gold_answer"])
                                ),
                                "baseline_contains_answer": int(
                                    contains_answer(baseline_generation, resolved["gold_answer"])
                                ),
                                "steered_contains_answer": int(
                                    contains_answer(steered_generation, resolved["gold_answer"])
                                ),
                                "repetition_detected": int(repetition),
                                "malformed_output": int(malformed),
                            }
                        )
                        qualitative_examples.append(row.copy())
                    else:
                        row.update({field: "" for field in GENERATION_RESULT_FIELDS})

                rows.append(row)
                del steered_logits

        if generate_examples and baseline_generation:
            generated_count += 1

        del source_tokens, baseline_logits
        clear_memory(device)

        if idx == 0 or (idx + 1) % 10 == 0 or idx == len(test_examples) - 1:
            log(
                f"[test] {idx + 1}/{len(test_examples)} examples processed | "
                f"rows={len(rows)} | {cuda_mem_string(device)}"
            )

    elapsed = time.time() - t0
    log(f"[test] Evaluation finished in {format_seconds(elapsed)} with {len(rows)} rows")
    return rows, qualitative_examples, skipped


def build_alpha_sweep(
    rows: list[dict],
    model_name: str,
    source_cell: str,
    donor_cell: str,
    contrast_file: Path,
    layer: int,
    hook: str,
    control: str,
    n_train: int,
    n_test: int,
    vector_l2_norm: float,
) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=SUMMARY_FIELDS)

    df = pd.DataFrame(rows)
    grouped = []
    for (row_control, alpha), group in df.groupby(["control", "alpha"], sort=True):
        baseline_top1_rate = float(group["baseline_top1"].mean())
        steered_top1_rate = float(group["steered_top1"].mean())
        random_seed_mean = np.nan
        random_seed_std = np.nan
        random_seed_top1_mean = np.nan
        random_seed_top1_std = np.nan
        if "random_seed" in group.columns and group["random_seed"].astype(str).str.len().gt(0).any():
            seed_rows = []
            for _, seed_group in group.groupby("random_seed", sort=True):
                seed_baseline_top1 = float(seed_group["baseline_top1"].mean())
                seed_steered_top1 = float(seed_group["steered_top1"].mean())
                seed_rows.append(
                    {
                        "mean_delta_gold_logit": float(seed_group["delta_gold_logit"].mean()),
                        "top1_improvement": seed_steered_top1 - seed_baseline_top1,
                    }
                )
            seed_df = pd.DataFrame(seed_rows)
            random_seed_mean = float(seed_df["mean_delta_gold_logit"].mean())
            random_seed_std = float(seed_df["mean_delta_gold_logit"].std(ddof=0))
            random_seed_top1_mean = float(seed_df["top1_improvement"].mean())
            random_seed_top1_std = float(seed_df["top1_improvement"].std(ddof=0))
        grouped.append(
            {
                "model": model_name,
                "source_cell": source_cell,
                "donor_cell": donor_cell,
                "contrast_file": str(contrast_file),
                "layer": layer,
                "hook": hook,
                "token_position": TOKEN_POSITION_LABEL,
                "control": row_control,
                "n_train": n_train,
                "n_test": n_test,
                "vector_l2_norm": vector_l2_norm,
                "alpha": alpha,
                "mean_delta_gold_logit": float(group["delta_gold_logit"].mean()),
                "median_delta_gold_logit": float(group["delta_gold_logit"].median()),
                "std_delta_gold_logit": float(group["delta_gold_logit"].std(ddof=0)),
                "mean_delta_gold_rank": float(group["delta_gold_rank"].mean()),
                "median_delta_gold_rank": float(group["delta_gold_rank"].median()),
                "std_delta_gold_rank": float(group["delta_gold_rank"].std(ddof=0)),
                "baseline_top1_rate": baseline_top1_rate,
                "steered_top1_rate": steered_top1_rate,
                "top1_improvement": steered_top1_rate - baseline_top1_rate,
                "n_rows": int(len(group)),
                "random_control_mean_delta_gold_logit": random_seed_mean,
                "random_control_std_delta_gold_logit": random_seed_std,
                "random_control_mean_top1_improvement": random_seed_top1_mean,
                "random_control_std_top1_improvement": random_seed_top1_std,
            }
        )

    sweep = pd.DataFrame(grouped)
    if sweep.empty:
        return sweep

    best_delta_by_control = {}
    best_top1_by_control = {}
    for row_control, group in sweep.groupby("control", sort=False):
        best_delta_idx = group["mean_delta_gold_logit"].idxmax()
        best_top1_idx = group["top1_improvement"].idxmax()
        best_delta_by_control[row_control] = sweep.loc[best_delta_idx, "alpha"]
        best_top1_by_control[row_control] = sweep.loc[best_top1_idx, "alpha"]

    sweep["best_alpha_by_delta_logit"] = sweep["control"].map(best_delta_by_control)
    sweep["best_alpha_by_top1_improvement"] = sweep["control"].map(best_top1_by_control)
    return sweep


def save_csv(path: Path, df: pd.DataFrame, columns: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if columns is not None:
        ordered = [col for col in columns if col in df.columns]
        extra = [col for col in df.columns if col not in ordered]
        df = df[ordered + extra]
    df.to_csv(path, index=False, encoding="utf-8")
    log(f"[save] {path} ({len(df)} rows)")


def write_vector_stats(
    path: Path,
    model_name: str,
    source_cell: str,
    donor_cell: str,
    layer: int,
    requested_layer: int,
    hook: str,
    n_train: int,
    steering_vector: torch.Tensor,
    train_frac: float,
    seed: int,
    control: str,
    skipped_train: list[str],
    random_seed_values: list[int] | None = None,
) -> None:
    stats = {
        "model": model_name,
        "source_cell": source_cell,
        "donor_cell": donor_cell,
        "layer": layer,
        "requested_layer": requested_layer,
        "hook": hook,
        "token_position": TOKEN_POSITION_LABEL,
        "n_train": n_train,
        "vector_shape": list(steering_vector.shape),
        "vector_l2_norm": float(steering_vector.float().norm().item()),
        "vector_mean": float(steering_vector.float().mean().item()),
        "vector_std": float(steering_vector.float().std(unbiased=False).item()),
        "train_frac": train_frac,
        "seed": seed,
        "control": control,
        "random_seed_info": random_seed_values or [],
        "skipped_train_examples": skipped_train,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    log(f"[save] {path}")


def write_filler_not_implemented_report(
    report_path: Path,
    stats_path: Path,
    model_name: str,
    source_cell: str,
    donor_cell: str,
    layer: int,
    hook: str,
    contrast_file: Path,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Phase 5b Activation Steering Report",
        "",
        f"Model: `{model_name}`",
        f"Contrast: Cell {source_cell} -> Cell {donor_cell}",
        f"Layer: `{layer}`",
        f"Hook: `{hook}`",
        f"Contrast file: `{contrast_file}`",
        "",
        "## Status",
        "",
        "The filler control was requested, but it is not implemented in this Phase 5b script.",
        "",
        "This is intentional: the project plan marks filler as optional and says to document it as a limitation if it is not straightforward. The script does not invent a new filler methodology.",
        "",
        "No steering metrics were run for this control.",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    stats = {
        "model": model_name,
        "source_cell": source_cell,
        "donor_cell": donor_cell,
        "layer": layer,
        "hook": hook,
        "token_position": TOKEN_POSITION_LABEL,
        "control": "filler",
        "status": "not_implemented",
        "reason": "Filler control requires a separately specified contrast direction; the Phase 5 plan permits documenting it as a limitation.",
    }
    stats_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    log("[control] WARNING: filler control is not implemented; wrote limitation report")


def plot_alpha_sweep(summary: pd.DataFrame, path: Path) -> None:
    if summary.empty:
        log("[figure] WARNING: no alpha sweep data to plot")
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for control, group in summary.groupby("control", sort=False):
        group = group.sort_values("alpha")
        yerr = group["std_delta_gold_logit"] if "std_delta_gold_logit" in group else None
        ax.errorbar(
            group["alpha"],
            group["mean_delta_gold_logit"],
            yerr=yerr,
            marker="o",
            linewidth=1.6,
            capsize=3,
            label=str(control),
        )
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Alpha")
    ax.set_ylabel("Mean delta gold logit")
    ax.set_title("Phase 5b Steering Alpha Sweep")
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log(f"[figure] Saved {path}")


def select_best_rows(results: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    if results.empty or summary.empty:
        return pd.DataFrame()
    best_parts = []
    for control, group in summary.groupby("control", sort=False):
        best_alpha = group.loc[group["mean_delta_gold_logit"].idxmax(), "alpha"]
        best_parts.append(results[(results["control"] == control) & (results["alpha"] == best_alpha)])
    if not best_parts:
        return pd.DataFrame()
    return pd.concat(best_parts, ignore_index=True)


def alpha_zero_sanity(results: pd.DataFrame, tolerance: float = 1e-5) -> dict[str, Any] | None:
    if results.empty or "alpha" not in results.columns:
        return None

    zero_rows = results[np.isclose(results["alpha"].astype(float), 0.0)]
    if zero_rows.empty:
        return None

    max_abs_delta_logit = float(zero_rows["delta_gold_logit"].abs().max())
    max_abs_delta_rank = int(zero_rows["delta_gold_rank"].abs().max())
    top1_mismatch_count = int((zero_rows["baseline_top1"] != zero_rows["steered_top1"]).sum())
    return {
        "n_rows": int(len(zero_rows)),
        "max_abs_delta_gold_logit": max_abs_delta_logit,
        "max_abs_delta_gold_rank": max_abs_delta_rank,
        "top1_mismatch_count": top1_mismatch_count,
        "passed": (
            max_abs_delta_logit <= tolerance
            and max_abs_delta_rank == 0
            and top1_mismatch_count == 0
        ),
    }


def plot_logit_shift(results: pd.DataFrame, summary: pd.DataFrame, path: Path) -> None:
    selected = select_best_rows(results, summary)
    if selected.empty:
        log("[figure] WARNING: no logit shift data to plot")
        return
    means = [
        float(selected["baseline_gold_logit"].mean()),
        float(selected["steered_gold_logit"].mean()),
    ]
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.bar(["Baseline", "Steered"], means, color=["#667085", "#2c7bb6"])
    ax.set_ylabel("Mean gold-answer logit")
    ax.set_title("Gold Logit Shift at Best Alpha")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log(f"[figure] Saved {path}")


def plot_rank_shift(results: pd.DataFrame, summary: pd.DataFrame, path: Path) -> None:
    selected = select_best_rows(results, summary)
    if selected.empty:
        log("[figure] WARNING: no rank shift data to plot")
        return
    means = [
        float(selected["baseline_gold_rank"].mean()),
        float(selected["steered_gold_rank"].mean()),
    ]
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.bar(["Baseline", "Steered"], means, color=["#667085", "#2c7bb6"])
    ax.set_ylabel("Mean gold-token rank (lower is better)")
    ax.set_title("Gold Rank Shift at Best Alpha")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log(f"[figure] Saved {path}")


def md_escape(value: Any) -> str:
    return str(value).replace("\n", "<br>").replace("|", "\\|")


def write_markdown_report(
    path: Path,
    model_name: str,
    source_cell: str,
    donor_cell: str,
    contrast_file: Path,
    layer: int,
    requested_layer: int,
    hook: str,
    control: str,
    train_frac: float,
    seed: int,
    n_train: int,
    n_test: int,
    vector_l2_norm: float,
    summary: pd.DataFrame,
    qualitative_examples: list[dict],
    generate_examples: bool,
    skipped_train: list[str],
    skipped_test: list[str],
    random_seed_values: list[int] | None,
    alpha_zero_sanity: dict[str, Any] | None,
) -> None:
    lines = [
        "# Phase 5b Activation Steering Report",
        "",
        "## Experiment Configuration",
        "",
        f"- Model: `{model_name}`",
        f"- Source cell: `{source_cell}`",
        f"- Donor cell: `{donor_cell}`",
        f"- Contrast file: `{contrast_file}`",
        f"- Requested layer: `{requested_layer}`",
        f"- Applied layer: `{layer}`",
        f"- Hook: `{hook}`",
        f"- Token position: `{TOKEN_POSITION_LABEL}`",
        f"- Control: `{control}`",
        f"- Train fraction: `{train_frac}`",
        f"- Seed: `{seed}`",
        f"- Train examples used for split: `{n_train}`",
        f"- Held-out test examples: `{n_test}`",
        f"- Steering vector L2 norm: `{vector_l2_norm:.6f}`",
        "",
        "## Final-Token-Only Injection Confirmation",
        "",
        "Steering was injected only into the final prompt token position of the selected activation tensor. For score-only runs, the source prompt is the whole input and the injection index is `-1`. For optional generation, the injection stays fixed at the original prompt-final index rather than moving onto generated tokens.",
        "",
    ]

    if generate_examples:
        lines.extend(
            [
                "## Score and Generation Status",
                "",
                "This run includes score-only metrics plus a small qualitative generation sample. The generation rows are diagnostic only; exact-match remains the primary behavioural metric and answer-containment remains secondary.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "## Score-Only Status",
                "",
                "This is a score-only run. No generation was performed because `--generate-examples` was not provided.",
                "",
            ]
        )

    lines.extend(
        [
            "## Alpha Sweep Summary",
            "",
            "| control | alpha | mean delta gold logit | median delta gold logit | mean delta gold rank | baseline top1 | steered top1 | top1 improvement |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )

    if summary.empty:
        lines.append("| n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |")
    else:
        for _, row in summary.sort_values(["control", "alpha"]).iterrows():
            lines.append(
                f"| {md_escape(row['control'])} | {row['alpha']} | "
                f"{row['mean_delta_gold_logit']:.4f} | "
                f"{row['median_delta_gold_logit']:.4f} | "
                f"{row['mean_delta_gold_rank']:.2f} | "
                f"{row['baseline_top1_rate']:.3f} | "
                f"{row['steered_top1_rate']:.3f} | "
                f"{row['top1_improvement']:.3f} |"
            )

    lines.extend(["", "## Best Alpha", ""])
    if summary.empty:
        lines.append("No best alpha could be selected because no score rows were produced.")
    else:
        for row_control, group in summary.groupby("control", sort=False):
            best_delta = group.loc[group["mean_delta_gold_logit"].idxmax()]
            best_top1 = group.loc[group["top1_improvement"].idxmax()]
            lines.append(
                f"- `{row_control}` best by mean delta gold logit: alpha `{best_delta['alpha']}` "
                f"(mean delta `{best_delta['mean_delta_gold_logit']:.4f}`)."
            )
            lines.append(
                f"- `{row_control}` best by top-1 improvement: alpha `{best_top1['alpha']}` "
                f"(improvement `{best_top1['top1_improvement']:.3f}`)."
            )

    lines.extend(["", "## Alpha 0.0 Sanity Check", ""])
    if alpha_zero_sanity is None:
        lines.append("Alpha `0.0` was not included in this run.")
    else:
        lines.append(
            f"Rows at alpha `0.0`: `{alpha_zero_sanity['n_rows']}`. "
            f"Maximum absolute logit delta: `{alpha_zero_sanity['max_abs_delta_gold_logit']:.8f}`. "
            f"Maximum absolute rank delta: `{alpha_zero_sanity['max_abs_delta_gold_rank']}`. "
            f"Top-1 mismatches: `{alpha_zero_sanity['top1_mismatch_count']}`."
        )
        if alpha_zero_sanity["passed"]:
            lines.append("The alpha `0.0` sanity check passed: steered and baseline scores were identical or numerically negligible.")
        else:
            lines.append("The alpha `0.0` sanity check did not pass. Inspect the per-example rows before interpreting non-zero alphas.")

    lines.extend(["", "## Control Notes", ""])
    if control == "none":
        lines.append("The run used the learned donor-minus-source steering vector.")
    elif control == "random":
        seeds = ", ".join(str(seed_value) for seed_value in (random_seed_values or []))
        lines.append(
            "The run used Gaussian random vectors rescaled to match the learned vector L2 norm. "
            f"Random seeds: {seeds}."
        )
        if "std_delta_gold_logit" in summary.columns:
            lines.append(
                "The alpha sweep CSV includes standard deviations across the pooled random-control rows."
            )
    elif control == "early_layer":
        lines.append(
            f"The run computed and applied the same donor-minus-source procedure at early layer `{layer}`. "
            "Compare it against the main late-layer run before making strong late-layer claims."
        )
    elif control == "filler":
        lines.append(
            "The filler control is not implemented in this script. It is documented as a limitation rather than replaced by an invented proxy."
        )

    if qualitative_examples:
        lines.extend(
            [
                "",
                "## Qualitative Generation Examples",
                "",
                "| example_id | alpha | gold answer | baseline generation | steered generation | baseline exact | steered exact | baseline contains | steered contains |",
                "|---|---:|---|---|---|---:|---:|---:|---:|",
            ]
        )
        for row in qualitative_examples:
            lines.append(
                f"| {md_escape(row['example_id'])} | {row['alpha']} | "
                f"{md_escape(row['gold_answer'])} | "
                f"{md_escape(row.get('baseline_generated_answer', ''))} | "
                f"{md_escape(row.get('steered_generated_answer', ''))} | "
                f"{row.get('baseline_exact_match', '')} | "
                f"{row.get('steered_exact_match', '')} | "
                f"{row.get('baseline_contains_answer', '')} | "
                f"{row.get('steered_contains_answer', '')} |"
            )

    lines.extend(
        [
            "",
            "## Interpretation Guidance",
            "",
            "A positive mean delta gold logit indicates that steering increased the gold answer's first-token logit on held-out source prompts. A positive delta gold rank means the gold token moved upward in the vocabulary ranking because rank improvement is recorded as baseline rank minus steered rank.",
            "",
            "Strong evidence requires the learned late-layer vector to improve gold logit, rank, and top-1 rate, while outperforming random matched-norm and early-layer controls.",
            "",
            "## Limitations",
            "",
            "- Activation patching is example-specific, while this steering vector is an average direction.",
            "- Logit and rank recovery may not translate into full generated-answer recovery.",
            "- The first implementation supports `resid_post` only.",
            "- The filler control is not implemented in this version.",
        ]
    )

    if skipped_train or skipped_test:
        lines.extend(["", "## Skipped Examples", ""])
        lines.append(f"- Skipped train examples: `{len(skipped_train)}`")
        lines.append(f"- Skipped test examples: `{len(skipped_test)}`")

    lines.extend(
        [
            "",
            "## Thesis-Safe Language",
            "",
            "Activation steering was introduced as an intervention experiment based on the late-layer localisation found through activation patching. The steering vector was computed as the average donor-minus-source activation difference over training examples and evaluated on held-out examples. Steering was applied only at the final prompt token, matching the final-token scoring methodology used in activation patching. The primary outcome is gold-answer logit, rank, and top-1 recovery.",
            "",
            "Avoid claiming that the vector is a reasoning circuit or that steering fully fixes the model. A negative or mixed result remains informative because it tests whether example-specific patching effects transfer into a reusable average direction.",
        ]
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"[save] {path}")


def resolve_paths(args) -> dict[str, Path | str]:
    slug = model_slug(args.model)
    source_cell = args.source_cell.upper()
    donor_cell = args.donor_cell.upper()
    contrast_prefix = output_prefix_for(source_cell, donor_cell, args.output_prefix)
    control_prefixes = {
        "none": "",
        "random": "random_",
        "early_layer": "early_layer_",
        "filler": "filler_",
    }
    control_prefix = control_prefixes[args.control]
    prefix = model_file_prefix(slug, f"{contrast_prefix}{control_prefix}")

    dataset_path = Path(args.dataset or f"dataset/processed/{slug}/dataset.json")
    contrast_path = Path(args.contrast_file or contrast_path_for(slug, source_cell, donor_cell))
    result_dir = Path(args.output_dir or f"results/phase_5b_activation_steering/{slug}")
    figure_dir = Path(args.figure_dir or f"figures/phase_5b_activation_steering/{slug}")

    return {
        "slug": slug,
        "prefix": prefix,
        "contrast_prefix": contrast_prefix,
        "control_prefix": control_prefix,
        "dataset_path": dataset_path,
        "contrast_path": contrast_path,
        "result_dir": result_dir,
        "figure_dir": figure_dir,
        "results_path": result_dir / f"{prefix}steering_results.csv",
        "summary_path": result_dir / f"{prefix}steering_summary.csv",
        "alpha_path": result_dir / f"{prefix}steering_alpha_sweep.csv",
        "stats_path": result_dir / f"{prefix}steering_vector_stats.json",
        "report_path": result_dir / f"{prefix}steering_report.md",
        "alpha_fig_path": figure_dir / f"{prefix}steering_alpha_sweep.png",
        "logit_fig_path": figure_dir / f"{prefix}steering_logit_shift.png",
        "rank_fig_path": figure_dir / f"{prefix}steering_rank_shift.png",
    }


def print_dry_run(args, paths: dict[str, Path | str]) -> None:
    dataset_path = paths["dataset_path"]
    contrast_path = paths["contrast_path"]
    dataset_index = load_dataset_index(dataset_path)
    examples = load_json_list(contrast_path, "contrast")
    valid = validate_contrast_examples(
        examples,
        dataset_index,
        args.source_cell.upper(),
        args.donor_cell.upper(),
    )
    if args.max_examples is not None:
        valid = valid[: args.max_examples]
    train, test = split_examples(valid, args.train_frac, args.seed)

    applied_layer = args.early_layer if args.control == "early_layer" else args.layer
    hook_name = hook_name_for(args.hook, applied_layer)

    log("[dry-run] Phase 5b Activation Steering")
    log(f"[dry-run] model: {args.model}")
    log(f"[dry-run] source/donor: Cell {args.source_cell.upper()} -> Cell {args.donor_cell.upper()}")
    log(f"[dry-run] dataset: {dataset_path}")
    log(f"[dry-run] contrast: {contrast_path}")
    log(f"[dry-run] valid examples: {len(valid)}")
    log(f"[dry-run] train/test: {len(train)}/{len(test)} using train_frac={args.train_frac} seed={args.seed}")
    log(f"[dry-run] control: {args.control}")
    log(f"[dry-run] layer: {applied_layer} (requested layer {args.layer})")
    log(f"[dry-run] hook: {args.hook} -> {hook_name}")
    log(f"[dry-run] token position: {TOKEN_POSITION_LABEL}")
    log(f"[dry-run] alphas: {' '.join(str(alpha) for alpha in args.alphas)}")
    log(f"[dry-run] output results: {paths['results_path']}")
    log(f"[dry-run] output summary: {paths['summary_path']}")
    log(f"[dry-run] output report: {paths['report_path']}")
    log("[dry-run] no model was loaded and no GPU steering was run")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase 5b: final-token average activation steering intervention"
    )
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B")
    parser.add_argument("--source-cell", type=str, required=True)
    parser.add_argument("--donor-cell", type=str, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--hook", type=str, default="resid_post", choices=sorted(SUPPORTED_HOOKS))
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--control",
        type=str,
        default="none",
        choices=["none", "random", "early_layer", "filler"],
    )
    parser.add_argument("--early-layer", type=int, default=8)
    parser.add_argument("--random-seeds", type=int, default=3)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "float32", "bfloat16"],
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--generate-examples", action="store_true")
    parser.add_argument("--generation-limit", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--contrast-file", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--figure-dir", type=str, default=None)
    parser.add_argument("--output-prefix", type=str, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_cell = args.source_cell.upper()
    donor_cell = args.donor_cell.upper()
    paths = resolve_paths(args)

    if args.control == "random" and args.random_seeds < 1:
        raise ValueError("--random-seeds must be at least 1 for random control")
    if args.generation_limit < 0:
        raise ValueError("--generation-limit must be non-negative")

    if args.dry_run:
        print_dry_run(args, paths)
        return 0

    result_dir = paths["result_dir"]
    figure_dir = paths["figure_dir"]
    result_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    applied_layer = args.early_layer if args.control == "early_layer" else args.layer
    hook_name = hook_name_for(args.hook, applied_layer)

    if args.control == "filler":
        write_filler_not_implemented_report(
            paths["report_path"],
            paths["stats_path"],
            args.model,
            source_cell,
            donor_cell,
            applied_layer,
            args.hook,
            paths["contrast_path"],
        )
        return 0

    dataset_index = load_dataset_index(paths["dataset_path"])
    raw_examples = load_json_list(paths["contrast_path"], "contrast")
    examples = validate_contrast_examples(raw_examples, dataset_index, source_cell, donor_cell)
    if args.max_examples is not None:
        examples = examples[: args.max_examples]
        log(f"[main] Limiting to first {args.max_examples} contrast examples")

    if len(examples) < 2:
        raise RuntimeError("Need at least two valid examples for a train/test split")

    train_examples, test_examples = split_examples(examples, args.train_frac, args.seed)

    log("=" * 70)
    log("Phase 5b: Activation Steering")
    log(f"  model:          {args.model}")
    log(f"  contrast:       Cell {source_cell} -> Cell {donor_cell}")
    log(f"  contrast file:  {paths['contrast_path']}")
    log(f"  control:        {args.control}")
    log(f"  layer:          {applied_layer} (requested {args.layer})")
    log(f"  hook:           {args.hook} -> {hook_name}")
    log(f"  token position: {TOKEN_POSITION_LABEL}")
    log(f"  train/test:     {len(train_examples)}/{len(test_examples)}")
    log(f"  alphas:         {args.alphas}")
    log("=" * 70)

    ensure_runtime_imports()
    model = load_model(args.model, args.device, args.dtype)
    if applied_layer < 0 or applied_layer >= model.cfg.n_layers:
        raise ValueError(
            f"Layer {applied_layer} is out of range for {args.model} "
            f"(n_layers={model.cfg.n_layers})"
        )

    reset_cuda_peak_memory_stats(args.device)
    steering_vector, skipped_train = compute_steering_vector(
        model,
        train_examples,
        dataset_index,
        source_cell,
        donor_cell,
        hook_name,
        args.device,
    )
    vector_norm = float(steering_vector.float().norm().item())

    random_seed_values = None
    if args.control == "random":
        random_seed_values = [args.seed + offset for offset in range(args.random_seeds)]
        steering_vectors = [
            (f"seed_{seed_value}", make_random_matched_norm_vector(steering_vector, seed_value))
            for seed_value in random_seed_values
        ]
    else:
        steering_vectors = [(None, steering_vector)]

    rows, qualitative_examples, skipped_test = evaluate_test_examples(
        model=model,
        test_examples=test_examples,
        dataset_index=dataset_index,
        model_name=args.model,
        source_cell=source_cell,
        donor_cell=donor_cell,
        layer=applied_layer,
        hook=args.hook,
        hook_name=hook_name,
        control=args.control,
        alphas=args.alphas,
        steering_vectors=steering_vectors,
        device=args.device,
        generate_examples=args.generate_examples,
        generation_limit=args.generation_limit,
        max_new_tokens=args.max_new_tokens,
    )

    if not rows:
        raise RuntimeError("No test rows were produced")

    results_df = pd.DataFrame(rows)
    result_columns = BASE_RESULT_FIELDS + (GENERATION_RESULT_FIELDS if args.generate_examples else [])
    save_csv(paths["results_path"], results_df, result_columns)

    summary_df = build_alpha_sweep(
        rows,
        args.model,
        source_cell,
        donor_cell,
        paths["contrast_path"],
        applied_layer,
        args.hook,
        args.control,
        len(train_examples) - len(skipped_train),
        len(test_examples) - len(skipped_test),
        vector_norm,
    )
    save_csv(paths["alpha_path"], summary_df)
    save_csv(paths["summary_path"], summary_df, SUMMARY_FIELDS)

    write_vector_stats(
        paths["stats_path"],
        args.model,
        source_cell,
        donor_cell,
        applied_layer,
        args.layer,
        args.hook,
        len(train_examples) - len(skipped_train),
        steering_vector,
        args.train_frac,
        args.seed,
        args.control,
        skipped_train,
        random_seed_values=random_seed_values,
    )

    plot_alpha_sweep(summary_df, paths["alpha_fig_path"])
    plot_logit_shift(results_df, summary_df, paths["logit_fig_path"])
    plot_rank_shift(results_df, summary_df, paths["rank_fig_path"])
    alpha_zero_check = alpha_zero_sanity(results_df)

    write_markdown_report(
        paths["report_path"],
        args.model,
        source_cell,
        donor_cell,
        paths["contrast_path"],
        applied_layer,
        args.layer,
        args.hook,
        args.control,
        args.train_frac,
        args.seed,
        len(train_examples) - len(skipped_train),
        len(test_examples) - len(skipped_test),
        vector_norm,
        summary_df,
        qualitative_examples,
        args.generate_examples,
        skipped_train,
        skipped_test,
        random_seed_values,
        alpha_zero_check,
    )

    log("=" * 70)
    log("Phase 5b activation steering complete")
    if alpha_zero_check is not None:
        status = "passed" if alpha_zero_check["passed"] else "FAILED"
        log(
            "Alpha 0.0 sanity check "
            f"{status}: max_abs_delta_logit={alpha_zero_check['max_abs_delta_gold_logit']:.8f}, "
            f"max_abs_delta_rank={alpha_zero_check['max_abs_delta_gold_rank']}, "
            f"top1_mismatches={alpha_zero_check['top1_mismatch_count']}"
        )
    if not summary_df.empty:
        best = summary_df.loc[summary_df["mean_delta_gold_logit"].idxmax()]
        log(
            f"Best mean delta gold logit: control={best['control']} "
            f"alpha={best['alpha']} delta={best['mean_delta_gold_logit']:+.4f}"
        )
    log(f"Results: {paths['results_path']}")
    log(f"Report:  {paths['report_path']}")
    log("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
