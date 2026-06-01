#!/usr/bin/env python3
"""
Phase 5a Steering Calibration
=============================

Calibration diagnostics for Phase 5 steering. These routines run before the
final average-steering intervention so the selected layer and alpha range are
grounded in oracle and late-layer sweep evidence.

This script does not change the Phase 5b steering method. It reuses
activation_steering.py helpers for model loading, prompt materialisation,
train/test splitting, activation extraction, final-token-only injection, and
gold-token scoring.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

try:
    import activation_steering as steering
except ModuleNotFoundError:
    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root / "scripts" / "phase_5b_activation_steering"))
    import activation_steering as steering

try:
    from scripts.utils.contrast_config import contrast_path_for, model_file_prefix, output_prefix_for
except ModuleNotFoundError:
    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))
    from scripts.utils.contrast_config import contrast_path_for, model_file_prefix, output_prefix_for


DEFAULT_DIAGNOSTIC_ALPHAS = [0.25, 0.5, 0.75, 1.0, 1.25]
DEFAULT_ORACLE_ALPHAS = [0.25, 0.5, 0.75, 1.0]
DEFAULT_LATE_LAYERS = [31, 32, 33, 34, 35]
DEFAULT_FINAL_ALPHA_RANGE = [0.0, 0.25, 0.5, 0.75, 1.0]
HELPED_HURT_EPS = 1e-6


def log(message: str) -> None:
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def model_slug(model_name: str) -> str:
    return model_name.split("/")[-1].lower()


def ensure_runtime_imports():
    steering.ensure_runtime_imports()
    return steering.np, steering.pd, steering.plt, steering.torch


def resolve_common_paths(args) -> dict[str, Path | str]:
    slug = model_slug(args.model)
    source_cell = args.source_cell.upper()
    donor_cell = args.donor_cell.upper()
    prefix = model_file_prefix(slug, output_prefix_for(source_cell, donor_cell, args.output_prefix))
    result_dir = Path(args.output_dir or f"results/phase_5a_steering_calibration/{slug}")
    figure_dir = Path(args.figure_dir or f"figures/phase_5a_steering_calibration/{slug}")
    analysis_result_dir = Path(args.analysis_output_dir or f"results/phase_5c_steering_analysis/{slug}")
    analysis_figure_dir = Path(args.analysis_figure_dir or f"figures/phase_5c_steering_analysis/{slug}")
    dataset_path = Path(args.dataset or f"dataset/processed/{slug}/dataset.json")
    contrast_path = Path(args.contrast_file or contrast_path_for(slug, source_cell, donor_cell))

    return {
        "slug": slug,
        "prefix": prefix,
        "result_dir": result_dir,
        "figure_dir": figure_dir,
        "dataset_path": dataset_path,
        "contrast_path": contrast_path,
        "oracle_results": result_dir / f"{prefix}oracle_steering_results.csv",
        "oracle_summary": result_dir / f"{prefix}oracle_steering_summary.csv",
        "oracle_figure": figure_dir / f"{prefix}oracle_steering_alpha_sweep.png",
        "layer_sweep_results": result_dir / f"{prefix}layer_sweep_steering_results.csv",
        "layer_sweep_summary": result_dir / f"{prefix}layer_sweep_steering_summary.csv",
        "layer_sweep_figure": figure_dir / f"{prefix}layer_sweep_steering_heatmap.png",
        "recommended_config": result_dir / f"{prefix}recommended_steering_config.json",
        "calibration_report": result_dir / f"{prefix}steering_calibration_report.md",
        "analysis_result_dir": analysis_result_dir,
        "analysis_figure_dir": analysis_figure_dir,
        "helped_hurt_csv": analysis_result_dir / f"{prefix}helped_hurt_analysis.csv",
        "helped_hurt_report": analysis_result_dir / f"{prefix}helped_hurt_report.md",
        "final_interpretation": analysis_result_dir / f"{prefix}final_steering_interpretation.md",
        "default_steering_results": Path(f"results/phase_5b_activation_steering/{slug}/{prefix}steering_results.csv"),
    }


def load_valid_examples(args, paths: dict[str, Path | str]) -> tuple[dict[str, dict], list[dict], list[dict], list[dict]]:
    dataset_index = steering.load_dataset_index(paths["dataset_path"])
    raw_examples = steering.load_json_list(paths["contrast_path"], "contrast")
    examples = steering.validate_contrast_examples(
        raw_examples,
        dataset_index,
        args.source_cell.upper(),
        args.donor_cell.upper(),
    )
    if args.max_examples is not None:
        examples = examples[: args.max_examples]
    if len(examples) < 2:
        raise RuntimeError("Need at least two valid examples for diagnostics")
    train_examples, test_examples = steering.split_examples(examples, args.train_frac, args.seed)
    return dataset_index, examples, train_examples, test_examples


def print_dry_run(args, paths: dict[str, Path | str]) -> None:
    dataset_index, examples, train_examples, test_examples = load_valid_examples(args, paths)
    del dataset_index

    layers = args.layers or [args.layer]
    log("[dry-run] Phase 5a Steering Calibration / Phase 5c Analysis")
    log(f"[dry-run] diagnostic: {args.diagnostic}")
    log(f"[dry-run] model: {args.model}")
    log(f"[dry-run] source/donor: Cell {args.source_cell.upper()} -> Cell {args.donor_cell.upper()}")
    log(f"[dry-run] dataset: {paths['dataset_path']}")
    log(f"[dry-run] contrast: {paths['contrast_path']}")
    log(f"[dry-run] valid examples: {len(examples)}")
    log(f"[dry-run] train/test: {len(train_examples)}/{len(test_examples)} using train_frac={args.train_frac} seed={args.seed}")
    log(f"[dry-run] layer: {args.layer}")
    log(f"[dry-run] layers: {' '.join(str(layer) for layer in layers)}")
    log(f"[dry-run] hook: {args.hook}")
    log(f"[dry-run] token position: {steering.TOKEN_POSITION_LABEL}")
    log(f"[dry-run] alphas: {' '.join(str(alpha) for alpha in args.alphas)}")
    log(f"[dry-run] oracle results: {paths['oracle_results']}")
    log(f"[dry-run] layer sweep results: {paths['layer_sweep_results']}")
    log(f"[dry-run] recommended config: {paths['recommended_config']}")
    log(f"[dry-run] calibration report: {paths['calibration_report']}")
    log(f"[dry-run] helped/hurt report: {paths['helped_hurt_report']}")
    log(f"[dry-run] final interpretation: {paths['final_interpretation']}")
    log("[dry-run] no model was loaded and no GPU diagnostics were run")


def score_with_vector(model, tokens, hook_name: str, vector, alpha: float, gold_token_id: int) -> dict[str, Any]:
    logits = steering.run_steered_logits(model, tokens, hook_name, vector, alpha)
    scores = steering.score_logits(logits, gold_token_id)
    del logits
    return scores


def run_oracle(args, paths: dict[str, Path | str]) -> None:
    np, pd, plt, torch = ensure_runtime_imports()
    source_cell = args.source_cell.upper()
    donor_cell = args.donor_cell.upper()
    hook_name = steering.hook_name_for(args.hook, args.layer)
    dataset_index, _examples, train_examples, test_examples = load_valid_examples(args, paths)

    model = steering.load_model(args.model, args.device, args.dtype)
    if args.layer < 0 or args.layer >= model.cfg.n_layers:
        raise ValueError(f"Layer {args.layer} is out of range for {args.model} (n_layers={model.cfg.n_layers})")

    rows = []
    skipped = []
    log(f"[oracle] Running oracle per-example steering on {len(test_examples)} held-out examples")
    for idx, ex in enumerate(test_examples):
        resolved = steering.resolve_prompt_pair(ex, dataset_index, source_cell, donor_cell, model.tokenizer)
        if resolved is None:
            skipped.append(str(ex.get("example_id", f"index_{idx}")))
            continue

        aligned, source_len, donor_len = steering.token_lengths_match(
            model,
            resolved["source_prompt"],
            resolved["donor_prompt"],
            args.device,
        )
        if not aligned:
            log(
                f"[oracle] WARNING: skipping {resolved['example_id']}: "
                f"Cell {source_cell}={source_len}, Cell {donor_cell}={donor_len}"
            )
            skipped.append(resolved["example_id"])
            continue

        gold_token_id, _ = steering.get_gold_first_token_id(model, resolved["gold_answer"])
        source_tokens = steering.tokens_for_prompt(model, resolved["source_prompt"], args.device)
        donor_tokens = steering.tokens_for_prompt(model, resolved["donor_prompt"], args.device)
        baseline_logits = steering.run_baseline_logits(model, source_tokens)
        baseline = steering.score_logits(baseline_logits, gold_token_id)
        source_activation = steering.extract_final_activation(model, source_tokens, hook_name)
        donor_activation = steering.extract_final_activation(model, donor_tokens, hook_name)
        per_example_vector = donor_activation - source_activation
        vector_norm = float(per_example_vector.float().norm().item())

        for alpha in args.alphas:
            steered = score_with_vector(model, source_tokens, hook_name, per_example_vector, alpha, gold_token_id)
            rows.append(
                {
                    "model": args.model,
                    "diagnostic": "oracle",
                    "example_id": resolved["example_id"],
                    "domain": resolved.get("domain", ""),
                    "source_cell": source_cell,
                    "donor_cell": donor_cell,
                    "layer": args.layer,
                    "hook": args.hook,
                    "token_position": steering.TOKEN_POSITION_LABEL,
                    "alpha": alpha,
                    "split": "test",
                    "gold_answer": resolved["gold_answer"],
                    "gold_token_id": gold_token_id,
                    "per_example_vector_l2_norm": vector_norm,
                    "baseline_gold_logit": baseline["gold_logit"],
                    "steered_gold_logit": steered["gold_logit"],
                    "delta_gold_logit": steered["gold_logit"] - baseline["gold_logit"],
                    "baseline_gold_rank": baseline["gold_rank"],
                    "steered_gold_rank": steered["gold_rank"],
                    "delta_gold_rank": baseline["gold_rank"] - steered["gold_rank"],
                    "baseline_top1": baseline["top1"],
                    "steered_top1": steered["top1"],
                }
            )

        del source_tokens, donor_tokens, baseline_logits, source_activation, donor_activation, per_example_vector
        steering.clear_memory(args.device)
        if idx == 0 or (idx + 1) % 10 == 0 or idx == len(test_examples) - 1:
            log(f"[oracle] {idx + 1}/{len(test_examples)} examples processed | rows={len(rows)}")

    if not rows:
        raise RuntimeError("Oracle diagnostic produced no rows")

    results = pd.DataFrame(rows)
    summary = summarise_by_layer_alpha(
        results,
        diagnostic="oracle",
        model_name=args.model,
        source_cell=source_cell,
        donor_cell=donor_cell,
        contrast_file=paths["contrast_path"],
        n_train=len(train_examples),
        n_test=len(test_examples) - len(skipped),
    )
    save_df(results, paths["oracle_results"])
    save_df(summary, paths["oracle_summary"])
    plot_oracle_alpha_sweep(summary, paths["oracle_figure"])
    write_overall_report(paths, args)


def summarise_by_layer_alpha(
    results,
    diagnostic: str,
    model_name: str,
    source_cell: str,
    donor_cell: str,
    contrast_file: Path,
    n_train: int,
    n_test: int,
):
    _np, pd, _plt, _torch = ensure_runtime_imports()
    rows = []
    for (layer, alpha), group in results.groupby(["layer", "alpha"], sort=True):
        baseline_top1 = float(group["baseline_top1"].mean())
        steered_top1 = float(group["steered_top1"].mean())
        vector_col = "vector_l2_norm" if "vector_l2_norm" in group.columns else "per_example_vector_l2_norm"
        rows.append(
            {
                "model": model_name,
                "diagnostic": diagnostic,
                "source_cell": source_cell,
                "donor_cell": donor_cell,
                "contrast_file": str(contrast_file),
                "layer": int(layer),
                "hook": str(group["hook"].iloc[0]),
                "token_position": steering.TOKEN_POSITION_LABEL,
                "n_train": n_train,
                "n_test": n_test,
                "alpha": float(alpha),
                "mean_vector_l2_norm": float(group[vector_col].mean()) if vector_col in group else "",
                "mean_delta_gold_logit": float(group["delta_gold_logit"].mean()),
                "median_delta_gold_logit": float(group["delta_gold_logit"].median()),
                "std_delta_gold_logit": float(group["delta_gold_logit"].std(ddof=0)),
                "mean_delta_gold_rank": float(group["delta_gold_rank"].mean()),
                "median_delta_gold_rank": float(group["delta_gold_rank"].median()),
                "std_delta_gold_rank": float(group["delta_gold_rank"].std(ddof=0)),
                "baseline_top1_rate": baseline_top1,
                "steered_top1_rate": steered_top1,
                "top1_improvement": steered_top1 - baseline_top1,
                "n_rows": int(len(group)),
            }
        )

    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    best_delta = {}
    best_top1 = {}
    for layer, group in summary.groupby("layer", sort=False):
        best_delta[layer] = group.loc[group["mean_delta_gold_logit"].idxmax(), "alpha"]
        best_top1[layer] = group.loc[group["top1_improvement"].idxmax(), "alpha"]
    summary["best_alpha_by_delta_logit_for_layer"] = summary["layer"].map(best_delta)
    summary["best_alpha_by_top1_for_layer"] = summary["layer"].map(best_top1)
    return summary


def recommended_config_from_layer_sweep(summary, args, paths: dict[str, Path | str]) -> dict[str, Any] | None:
    if summary.empty:
        return None
    candidates = summary[summary["alpha"].astype(float) <= 1.0].copy()
    if candidates.empty:
        candidates = summary.copy()
    candidates = candidates.sort_values(
        ["mean_delta_gold_logit", "alpha"],
        ascending=[False, True],
        kind="mergesort",
    )
    best = candidates.iloc[0]
    return {
        "model": args.model,
        "source_cell": args.source_cell.upper(),
        "donor_cell": args.donor_cell.upper(),
        "hook": args.hook,
        "recommended_layer": int(best["layer"]),
        "recommended_alpha": float(best["alpha"]),
        "alpha_range_for_final": list(DEFAULT_FINAL_ALPHA_RANGE),
        "selection_metric": "highest mean_delta_gold_logit from layer_sweep with alpha <= 1.0; ties prefer lower alpha",
        "mean_delta_gold_logit": float(best["mean_delta_gold_logit"]),
        "mean_delta_gold_rank": float(best["mean_delta_gold_rank"]),
        "top1_improvement": float(best["top1_improvement"]),
        "contrast_file": str(paths["contrast_path"]),
        "layer_sweep_summary": str(paths["layer_sweep_summary"]),
        "note": "This recommendation is for held-out average steering, not oracle per-example steering.",
    }


def write_recommended_config(summary, args, paths: dict[str, Path | str]) -> dict[str, Any] | None:
    config = recommended_config_from_layer_sweep(summary, args, paths)
    if config is None:
        return None
    path = Path(paths["recommended_config"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    log(f"[save] {path}")
    return config


def run_layer_sweep(args, paths: dict[str, Path | str]) -> None:
    _np, pd, _plt, _torch = ensure_runtime_imports()
    source_cell = args.source_cell.upper()
    donor_cell = args.donor_cell.upper()
    layers = args.layers or DEFAULT_LATE_LAYERS
    dataset_index, _examples, train_examples, test_examples = load_valid_examples(args, paths)
    model = steering.load_model(args.model, args.device, args.dtype)

    all_rows = []
    for layer in layers:
        if layer < 0 or layer >= model.cfg.n_layers:
            raise ValueError(f"Layer {layer} is out of range for {args.model} (n_layers={model.cfg.n_layers})")
        hook_name = steering.hook_name_for(args.hook, layer)
        log(f"[layer-sweep] Computing average vector for layer {layer}")
        vector, skipped_train = steering.compute_steering_vector(
            model,
            train_examples,
            dataset_index,
            source_cell,
            donor_cell,
            hook_name,
            args.device,
        )
        vector_norm = float(vector.float().norm().item())
        rows, _qual, skipped_test = steering.evaluate_test_examples(
            model=model,
            test_examples=test_examples,
            dataset_index=dataset_index,
            model_name=args.model,
            source_cell=source_cell,
            donor_cell=donor_cell,
            layer=layer,
            hook=args.hook,
            hook_name=hook_name,
            control="layer_sweep",
            alphas=args.alphas,
            steering_vectors=[(None, vector)],
            device=args.device,
            generate_examples=False,
            generation_limit=0,
            max_new_tokens=0,
        )
        for row in rows:
            row["diagnostic"] = "layer_sweep"
            row["vector_l2_norm"] = vector_norm
            row["n_train_valid"] = len(train_examples) - len(skipped_train)
            row["n_test_valid"] = len(test_examples) - len(skipped_test)
        all_rows.extend(rows)
        del vector
        steering.clear_memory(args.device)

    if not all_rows:
        raise RuntimeError("Layer-sweep diagnostic produced no rows")

    results = pd.DataFrame(all_rows)
    summary = summarise_by_layer_alpha(
        results,
        diagnostic="layer_sweep",
        model_name=args.model,
        source_cell=source_cell,
        donor_cell=donor_cell,
        contrast_file=paths["contrast_path"],
        n_train=int(results["n_train_valid"].max()) if "n_train_valid" in results else len(train_examples),
        n_test=int(results["n_test_valid"].max()) if "n_test_valid" in results else len(test_examples),
    )
    save_df(results, paths["layer_sweep_results"])
    save_df(summary, paths["layer_sweep_summary"])
    plot_layer_sweep_heatmap(summary, paths["layer_sweep_figure"])
    write_recommended_config(summary, args, paths)
    write_overall_report(paths, args)


def save_df(df, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")
    log(f"[save] {path} ({len(df)} rows)")


def plot_oracle_alpha_sweep(summary, path: Path) -> None:
    _np, _pd, plt, _torch = ensure_runtime_imports()
    if summary.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for layer, group in summary.groupby("layer", sort=True):
        group = group.sort_values("alpha")
        ax.plot(
            group["alpha"],
            group["mean_delta_gold_logit"],
            marker="o",
            linewidth=1.6,
            label=f"layer {layer}",
        )
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Alpha")
    ax.set_ylabel("Mean delta gold logit")
    ax.set_title("Oracle Per-Example Steering")
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log(f"[figure] Saved {path}")


def plot_layer_sweep_heatmap(summary, path: Path) -> None:
    np, _pd, plt, _torch = ensure_runtime_imports()
    if summary.empty:
        return
    layers = sorted(summary["layer"].unique())
    alphas = sorted(summary["alpha"].unique())
    heat = np.full((len(layers), len(alphas)), np.nan)
    for i, layer in enumerate(layers):
        for j, alpha in enumerate(alphas):
            match = summary[(summary["layer"] == layer) & (summary["alpha"] == alpha)]
            if not match.empty:
                heat[i, j] = float(match["mean_delta_gold_logit"].iloc[0])

    fig, ax = plt.subplots(figsize=(7, 4.8))
    im = ax.imshow(heat, aspect="auto", cmap="coolwarm")
    ax.set_xticks(range(len(alphas)), [str(alpha) for alpha in alphas])
    ax.set_yticks(range(len(layers)), [str(layer) for layer in layers])
    ax.set_xlabel("Alpha")
    ax.set_ylabel("Layer")
    ax.set_title("Late-Layer Average Steering Sweep")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Mean delta gold logit")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log(f"[figure] Saved {path}")


def read_csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def as_float(row: dict, key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    if value in ("", None):
        return default
    return float(value)


def classify_delta(delta: float, eps: float = HELPED_HURT_EPS) -> str:
    if delta > eps:
        return "helped"
    if delta < -eps:
        return "hurt"
    return "unchanged"


def run_helped_hurt(args, paths: dict[str, Path | str]) -> None:
    source_path = Path(args.steering_results or paths["default_steering_results"])
    rows = read_csv_rows(source_path)
    if not rows:
        raise RuntimeError(f"No rows in steering results: {source_path}")

    detail_rows = []
    grouped = defaultdict(list)
    for row in rows:
        delta = as_float(row, "delta_gold_logit")
        label = classify_delta(delta)
        out = {
            "source_file": str(source_path),
            "example_id": row.get("example_id", ""),
            "domain": row.get("domain", ""),
            "layer": row.get("layer", ""),
            "hook": row.get("hook", ""),
            "control": row.get("control", ""),
            "alpha": row.get("alpha", ""),
            "gold_answer": row.get("gold_answer", ""),
            "baseline_gold_logit": row.get("baseline_gold_logit", ""),
            "steered_gold_logit": row.get("steered_gold_logit", ""),
            "delta_gold_logit": row.get("delta_gold_logit", ""),
            "baseline_gold_rank": row.get("baseline_gold_rank", ""),
            "steered_gold_rank": row.get("steered_gold_rank", ""),
            "delta_gold_rank": row.get("delta_gold_rank", ""),
            "baseline_top1": row.get("baseline_top1", ""),
            "steered_top1": row.get("steered_top1", ""),
            "helped_hurt_label": label,
        }
        detail_rows.append(out)
        grouped[row.get("alpha", "")].append(out)

    write_csv(paths["helped_hurt_csv"], detail_rows, list(detail_rows[0].keys()))
    write_helped_hurt_report(paths["helped_hurt_report"], source_path, detail_rows, grouped)
    write_final_interpretation(paths["final_interpretation"], source_path, detail_rows, grouped)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    log(f"[save] {path} ({len(rows)} rows)")


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def format_float(value: float) -> str:
    return f"{value:.4f}"


def write_helped_hurt_report(
    path: Path,
    source_path: Path,
    detail_rows: list[dict],
    grouped: dict[str, list[dict]],
) -> None:
    lines = [
        "# Phase 5c Helped vs Hurt Analysis",
        "",
        f"Source results: `{source_path}`",
        "",
        "A row is labelled `helped` when `delta_gold_logit > 0`, `hurt` when `delta_gold_logit < 0`, and `unchanged` when the absolute delta is near zero.",
        "",
        "## Alpha Summary",
        "",
        "| alpha | helped | hurt | unchanged | mean baseline logit helped | mean baseline logit hurt | mean baseline rank helped | mean baseline rank hurt |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for alpha in sorted(grouped, key=lambda x: float(x) if x not in ("", None) else -999):
        rows = grouped[alpha]
        counts = Counter(row["helped_hurt_label"] for row in rows)
        helped = [row for row in rows if row["helped_hurt_label"] == "helped"]
        hurt = [row for row in rows if row["helped_hurt_label"] == "hurt"]
        lines.append(
            f"| {alpha} | {counts['helped']} | {counts['hurt']} | {counts['unchanged']} | "
            f"{format_float(mean([as_float(r, 'baseline_gold_logit') for r in helped]))} | "
            f"{format_float(mean([as_float(r, 'baseline_gold_logit') for r in hurt]))} | "
            f"{format_float(mean([as_float(r, 'baseline_gold_rank') for r in helped]))} | "
            f"{format_float(mean([as_float(r, 'baseline_gold_rank') for r in hurt]))} |"
        )

    lines.extend(["", "## Domain Breakdown", ""])
    domain_counts = defaultdict(Counter)
    for row in detail_rows:
        domain_counts[row.get("domain", "")][row["helped_hurt_label"]] += 1
    lines.extend(["| domain | helped | hurt | unchanged |", "|---|---:|---:|---:|"])
    for domain in sorted(domain_counts):
        counts = domain_counts[domain]
        lines.append(f"| {domain} | {counts['helped']} | {counts['hurt']} | {counts['unchanged']} |")

    sorted_helped = sorted(detail_rows, key=lambda r: as_float(r, "delta_gold_logit"), reverse=True)[:10]
    sorted_hurt = sorted(detail_rows, key=lambda r: as_float(r, "delta_gold_logit"))[:10]
    lines.extend(["", "## Top Examples Helped", "", "| example_id | alpha | domain | gold | delta logit | baseline rank | steered rank |", "|---|---:|---|---|---:|---:|---:|"])
    for row in sorted_helped:
        lines.append(
            f"| {md(row['example_id'])} | {row['alpha']} | {md(row['domain'])} | {md(row['gold_answer'])} | "
            f"{as_float(row, 'delta_gold_logit'):.4f} | {row['baseline_gold_rank']} | {row['steered_gold_rank']} |"
        )
    lines.extend(["", "## Top Examples Harmed", "", "| example_id | alpha | domain | gold | delta logit | baseline rank | steered rank |", "|---|---:|---|---|---:|---:|---:|"])
    for row in sorted_hurt:
        lines.append(
            f"| {md(row['example_id'])} | {row['alpha']} | {md(row['domain'])} | {md(row['gold_answer'])} | "
            f"{as_float(row, 'delta_gold_logit'):.4f} | {row['baseline_gold_rank']} | {row['steered_gold_rank']} |"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"[save] {path}")


def write_final_interpretation(
    path: Path,
    source_path: Path,
    detail_rows: list[dict],
    grouped: dict[str, list[dict]],
) -> None:
    total = len(detail_rows)
    helped = sum(1 for row in detail_rows if row["helped_hurt_label"] == "helped")
    hurt = sum(1 for row in detail_rows if row["helped_hurt_label"] == "hurt")
    unchanged = sum(1 for row in detail_rows if row["helped_hurt_label"] == "unchanged")
    alpha_summaries = []
    for alpha in sorted(grouped, key=lambda x: float(x) if x not in ("", None) else -999):
        rows = grouped[alpha]
        counts = Counter(row["helped_hurt_label"] for row in rows)
        mean_delta = mean([as_float(row, "delta_gold_logit") for row in rows])
        mean_rank_delta = mean([as_float(row, "delta_gold_rank") for row in rows])
        alpha_summaries.append((alpha, counts, mean_delta, mean_rank_delta))

    lines = [
        "# Phase 5c Final Steering Interpretation",
        "",
        f"Source results: `{source_path}`",
        "",
        "This report interprets the final average-steering run after calibration. It is post-steering analysis, not a new intervention.",
        "",
        "## Overall Helped/Hurt Balance",
        "",
        f"- Rows analysed: `{total}`",
        f"- Helped rows: `{helped}`",
        f"- Hurt rows: `{hurt}`",
        f"- Unchanged rows: `{unchanged}`",
        "",
        "## Alpha Safety",
        "",
        "| alpha | helped | hurt | unchanged | mean delta gold logit | mean delta gold rank |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for alpha, counts, mean_delta, mean_rank_delta in alpha_summaries:
        lines.append(
            f"| {alpha} | {counts['helped']} | {counts['hurt']} | {counts['unchanged']} | "
            f"{mean_delta:.4f} | {mean_rank_delta:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Thesis-Safe Interpretation",
            "",
            "- If average steering improves gold-answer logits or ranks but does not improve top-1/generation, describe it as representation-level recovery.",
            "- Do not claim that average steering fully fixes Qwen unless top-1 and generation clearly improve.",
            "- If helped and hurt examples are both common, describe the steering direction as partially useful and example-sensitive.",
            "- The calibrated layer and alpha range should be presented as selected from held-out score diagnostics, not hand-picked from the final steering outcome.",
        ]
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"[save] {path}")


def md(value: Any) -> str:
    return str(value).replace("\n", "<br>").replace("|", "\\|")


def best_row_from_csv(path: Path) -> dict[str, str] | None:
    if not path.exists():
        return None
    rows = read_csv_rows(path)
    if not rows:
        return None
    return max(rows, key=lambda row: as_float(row, "mean_delta_gold_logit"))


def best_top1_row_from_csv(path: Path) -> dict[str, str] | None:
    if not path.exists():
        return None
    rows = read_csv_rows(path)
    if not rows:
        return None
    return max(rows, key=lambda row: as_float(row, "top1_improvement"))


def load_recommended_config(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def write_overall_report(paths: dict[str, Path | str], args) -> None:
    learned_summary = Path(f"results/phase_5b_activation_steering/{paths['slug']}/{paths['prefix']}steering_summary.csv")
    oracle_best = best_row_from_csv(paths["oracle_summary"])
    oracle_best_top1 = best_top1_row_from_csv(paths["oracle_summary"])
    layer_best = best_row_from_csv(paths["layer_sweep_summary"])
    layer_best_top1 = best_top1_row_from_csv(paths["layer_sweep_summary"])
    learned_best = best_row_from_csv(learned_summary)
    learned_best_top1 = best_top1_row_from_csv(learned_summary)
    recommended = load_recommended_config(Path(paths["recommended_config"]))

    lines = [
        "# Phase 5a Steering Calibration Report",
        "",
        f"Model: `{args.model}`",
        f"Contrast: Cell {args.source_cell.upper()} -> Cell {args.donor_cell.upper()}",
        f"Contrast file: `{paths['contrast_path']}`",
        "",
        "Phase 5a runs calibration before the final average-steering intervention. Oracle steering is an upper bound on final-token intervention strength, and the layer sweep selects the average-steering layer and alpha range for Phase 5b.",
        "",
        "## Questions",
        "",
        "### Did oracle steering outperform average steering?",
        "",
    ]

    if oracle_best and learned_best:
        lines.append(
            f"Best oracle mean delta gold logit is `{as_float(oracle_best, 'mean_delta_gold_logit'):.4f}` "
            f"at layer `{oracle_best.get('layer')}`, alpha `{oracle_best.get('alpha')}`. "
            f"Best average-steering mean delta gold logit is `{as_float(learned_best, 'mean_delta_gold_logit'):.4f}` "
            f"at alpha `{learned_best.get('alpha')}`."
        )
    else:
        lines.append("Oracle and/or Phase 5b learned steering summaries are not both available yet. This is expected before the final steering run.")

    lines.extend(["", "### Does average steering peak at layer 34 or another late layer?", ""])
    if layer_best:
        lines.append(
            f"The current layer-sweep best by mean delta gold logit is layer `{layer_best.get('layer')}`, "
            f"alpha `{layer_best.get('alpha')}`, with mean delta `{as_float(layer_best, 'mean_delta_gold_logit'):.4f}`."
        )
    else:
        lines.append("Layer-sweep summary is not available yet.")

    lines.extend(["", "### Recommended Phase 5b configuration", ""])
    if recommended:
        alpha_range = " ".join(str(alpha) for alpha in recommended.get("alpha_range_for_final", []))
        lines.append(
            f"Recommended layer `{recommended.get('recommended_layer')}`, hook `{recommended.get('hook')}`, "
            f"with final alpha range `{alpha_range}`. Selection metric: {recommended.get('selection_metric')}."
        )
        lines.append("This recommendation is for average steering, not oracle per-example steering.")
    else:
        lines.append("No recommended config has been written yet. Run the layer-sweep diagnostic to create it.")

    lines.extend(["", "### Which alpha range is safest?", ""])
    candidates = [row for row in [oracle_best, layer_best, learned_best] if row]
    if candidates:
        alphas = sorted({row.get("alpha") for row in candidates})
        lines.append(
            "The safest alpha range should be judged from positive mean delta gold logit without top-1 degradation. "
            f"Current best-logit alphas observed in available summaries: `{', '.join(alphas)}`."
        )
    else:
        lines.append("No summary rows are available yet to identify a safe alpha range.")

    lines.extend(["", "### Are helped examples different from hurt examples?", ""])
    if Path(paths["helped_hurt_report"]).exists():
        lines.append(f"See `{paths['helped_hurt_report']}` for helped/hurt counts, baseline-rank comparisons, domain breakdowns, and top examples.")
    else:
        lines.append("Helped/hurt analysis has not been generated yet. It belongs to Phase 5c and should run after Phase 5b final steering.")

    lines.extend(["", "### Does the evidence support representation-level recovery?", ""])
    if learned_best:
        if as_float(learned_best, "mean_delta_gold_logit") > 0:
            lines.append("Available learned-steering results support representation-level recovery because the mean gold-answer logit improves.")
        else:
            lines.append("Available learned-steering results do not yet show positive mean gold-logit recovery.")
    else:
        lines.append("Learned-steering summary is not available.")

    lines.extend(["", "### Does the evidence support behavioural/top-1 recovery?", ""])
    top1_rows = [row for row in [learned_best_top1, oracle_best_top1, layer_best_top1] if row]
    if top1_rows and max(as_float(row, "top1_improvement") for row in top1_rows) > 0:
        best = max(top1_rows, key=lambda row: as_float(row, "top1_improvement"))
        lines.append(
            f"Some available diagnostic rows show top-1 improvement, with best observed improvement "
            f"`{as_float(best, 'top1_improvement'):.4f}`. This should still be framed carefully unless generation also improves."
        )
    else:
        lines.append("Available summaries do not yet support a strong behavioural/top-1 recovery claim.")

    lines.extend(
        [
            "",
            "## Thesis-Safe Claims",
            "",
            "- If logit/rank improves but top-1/generation does not, describe the result as representation-level recovery.",
            "- If oracle steering is much stronger than average steering, describe the effect as partly example-specific and say the average vector is too blunt.",
            "- If late layers outperform early layers or the selected late layer improves logit/rank recovery, this supports the late-layer mediation story from activation patching.",
            "",
            "## What Not To Claim",
            "",
            "- Do not claim steering fully fixes Qwen unless top-1 and generation clearly improve.",
            "- Do not claim the steering vector is the complete reasoning circuit.",
            "- Do not claim one average vector should work across all examples or models.",
        ]
    )

    report_path = Path(paths["calibration_report"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"[save] {report_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 5a steering calibration diagnostics and Phase 5c analysis")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B")
    parser.add_argument("--source-cell", type=str, default="B")
    parser.add_argument("--donor-cell", type=str, default="D")
    parser.add_argument("--layer", type=int, default=34)
    parser.add_argument("--layers", type=int, nargs="+", default=None)
    parser.add_argument("--hook", type=str, default="resid_post", choices=sorted(steering.SUPPORTED_HOOKS))
    parser.add_argument(
        "--diagnostic",
        type=str,
        required=True,
        choices=["oracle", "layer_sweep", "helped_hurt", "report"],
    )
    parser.add_argument("--alphas", type=float, nargs="+", default=DEFAULT_DIAGNOSTIC_ALPHAS)
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "float16", "float32", "bfloat16"])
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--contrast-file", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--figure-dir", type=str, default=None)
    parser.add_argument("--analysis-output-dir", type=str, default=None)
    parser.add_argument("--analysis-figure-dir", type=str, default=None)
    parser.add_argument("--output-prefix", type=str, default=None)
    parser.add_argument("--steering-results", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = resolve_common_paths(args)
    if args.dry_run:
        print_dry_run(args, paths)
        return 0

    if args.diagnostic == "oracle":
        run_oracle(args, paths)
    elif args.diagnostic == "layer_sweep":
        run_layer_sweep(args, paths)
    elif args.diagnostic == "helped_hurt":
        run_helped_hurt(args, paths)
    elif args.diagnostic == "report":
        write_overall_report(paths, args)
    else:
        raise ValueError(f"Unknown diagnostic: {args.diagnostic}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
