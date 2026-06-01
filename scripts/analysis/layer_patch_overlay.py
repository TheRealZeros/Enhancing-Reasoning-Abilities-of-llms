"""
Phase 5: Cross-model layer-level activation patching overlay.

Plots Pythia-2.8B A->C (clean structured recovery) and Qwen2.5-3B B->D
(noisy structured recovery) on the same axes.

NOTE: The two contrasts are NOT identical experimental conditions.
  Pythia A->C : Direct/Clean (wrong) vs Structured/Clean (correct), n=38
  Qwen  B->D  : Direct/Noisy (wrong) vs Structured/Noisy (correct), n=104
Differences in mean delta reflect contrast severity, sample size, AND
architecture -- not architecture alone.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def log(msg: str) -> None:
    print(msg, flush=True)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_summary(csv_path: str, label: str) -> pd.DataFrame:
    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(f"Layer patch summary not found: {p}")
    df = pd.read_csv(p)
    required = {"layer", "mean_delta", "std_delta", "n_examples"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path}: missing columns {missing}")
    df = df.sort_values("layer").reset_index(drop=True)
    df["label"] = label
    n_layers = int(df["layer"].max()) + 1
    df["relative_depth"] = df["layer"] / (n_layers - 1)
    return df


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

_COLORS = {
    "Pythia-2.8B A->C": "#1f77b4",
    "Qwen2.5-3B B->D":  "#d62728",
}
_LINESTYLES = {
    "Pythia-2.8B A->C": "-",
    "Qwen2.5-3B B->D":  "--",
}
_MARKERS = {
    "Pythia-2.8B A->C": "o",
    "Qwen2.5-3B B->D":  "s",
}


def _plot_overlay(
    pythia_df: pd.DataFrame,
    qwen_df: pd.DataFrame,
    x_col: str,
    x_label: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))

    for df in (pythia_df, qwen_df):
        label = df["label"].iloc[0]
        n = int(df["n_examples"].iloc[0])
        color = _COLORS[label]
        ax.plot(
            df[x_col],
            df["mean_delta"],
            color=color,
            linestyle=_LINESTYLES[label],
            marker=_MARKERS[label],
            markersize=4,
            linewidth=1.5,
            label=f"{label}  (n={n})",
        )
        ax.fill_between(
            df[x_col],
            df["mean_delta"] - df["std_delta"],
            df["mean_delta"] + df["std_delta"],
            color=color,
            alpha=0.12,
        )

    ax.axhline(0, color="black", linewidth=0.8, linestyle=":")
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel("Mean delta-logit (patched - baseline)", fontsize=12)
    ax.set_title("Cross-model layer-level activation patching", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, linewidth=0.4, alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log(f"[save] {out_path.resolve()}")


def plot_raw_depth(pythia_df, qwen_df, out_path):
    _plot_overlay(
        pythia_df, qwen_df,
        x_col="layer",
        x_label="Layer index (absolute)",
        out_path=out_path,
    )


def plot_relative_depth(pythia_df, qwen_df, out_path):
    _plot_overlay(
        pythia_df, qwen_df,
        x_col="relative_depth",
        x_label="Relative depth  (layer / (n_layers - 1))",
        out_path=out_path,
    )


# ---------------------------------------------------------------------------
# Summary CSV
# ---------------------------------------------------------------------------

def build_overlay_summary(pythia_df: pd.DataFrame, qwen_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for df in (pythia_df, qwen_df):
        label = df["label"].iloc[0]
        peak_row = df.loc[df["mean_delta"].idxmax()]
        first_pos_layers = df[df["mean_delta"] > 0]["layer"]
        first_pos = int(first_pos_layers.min()) if not first_pos_layers.empty else None
        n = int(df["n_examples"].iloc[0])
        n_layers = int(df["layer"].max()) + 1
        rows.append({
            "model_contrast":       label,
            "n_layers":             n_layers,
            "n_examples":           n,
            "peak_layer":           int(peak_row["layer"]),
            "peak_mean_delta":      round(float(peak_row["mean_delta"]), 4),
            "peak_relative_depth":  round(float(peak_row["relative_depth"]), 4),
            "first_positive_layer": first_pos,
            "mean_delta_layer_0":   round(float(df.loc[df["layer"] == 0, "mean_delta"].values[0]), 4),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Console interpretation
# ---------------------------------------------------------------------------

def print_interpretation(summary_df: pd.DataFrame) -> None:
    log("\n" + "=" * 70)
    log("CROSS-MODEL LAYER PATCHING OVERLAY -- INTERPRETATION")
    log("=" * 70)
    log("")
    log("Caution: the two series use different contrasts and sample sizes.")
    log("  Pythia-2.8B A->C : Direct/Clean (wrong) vs Structured/Clean (correct)")
    log("  Qwen2.5-3B  B->D : Direct/Noisy (wrong) vs Structured/Noisy (correct)")
    log("  Do not interpret magnitude differences as purely architectural.")
    log("")

    for _, row in summary_df.iterrows():
        log(f"  [{row['model_contrast']}]")
        log(f"    n_layers            : {row['n_layers']}")
        log(f"    n_examples          : {row['n_examples']}")
        log(f"    first positive layer: {row['first_positive_layer']}")
        log(f"    peak layer          : {row['peak_layer']}  (rel depth {row['peak_relative_depth']:.2f})")
        log(f"    peak mean_delta     : {row['peak_mean_delta']:+.4f}")
        log(f"    layer-0 delta       : {row['mean_delta_layer_0']:+.4f}")
        log("")

    log("Preliminary interpretation (treat cautiously):")
    log("  Both models show negative or near-zero deltas in early layers,")
    log("  transitioning to positive effect at mid-to-late depth.")
    log("  In relative depth the transition is broadly similar (~0.55-0.65),")
    log("  but Qwen B->D peaks sharply at the final 3 layers (rel ~0.94-1.0),")
    log("  while Pythia A->C rises more gradually from ~layer 10 onwards.")
    log("  The late-layer concentration in Qwen may reflect the harder noisy-")
    log("  context contrast (B->D) requiring deeper intervention, or genuine")
    log("  architectural differences in where answer information is processed.")
    log("=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 5: Cross-model layer patch overlay figures"
    )
    parser.add_argument(
        "--pythia-csv",
        type=str,
        default="results/phase_3a_layer_patching/pythia-2.8b/pythia-2.8b_layer_patch_summary.csv",
    )
    parser.add_argument(
        "--qwen-csv",
        type=str,
        default="results/phase_3a_layer_patching/qwen2.5-3b/qwen2.5-3b_noisy_layer_patch_summary.csv",
    )
    parser.add_argument(
        "--figure-dir",
        type=str,
        default="figures/analysis/layer_patch_overlay",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/analysis/layer_patch_overlay",
    )
    args = parser.parse_args()

    fig_dir = Path(args.figure_dir)
    out_dir = Path(args.output_dir)
    ensure_dir(fig_dir)
    ensure_dir(out_dir)
    file_prefix = "pythia-2.8b_qwen2.5-3b_"

    log(f"[load] Pythia CSV : {args.pythia_csv}")
    pythia_df = load_summary(args.pythia_csv, "Pythia-2.8B A->C")
    log(f"       {len(pythia_df)} layers loaded, n_examples={int(pythia_df['n_examples'].iloc[0])}")

    log(f"[load] Qwen CSV   : {args.qwen_csv}")
    qwen_df = load_summary(args.qwen_csv, "Qwen2.5-3B B->D")
    log(f"       {len(qwen_df)} layers loaded, n_examples={int(qwen_df['n_examples'].iloc[0])}")

    plot_raw_depth(
        pythia_df, qwen_df,
        out_path=fig_dir / f"{file_prefix}layer_patch_overlay_raw_depth.png",
    )
    plot_relative_depth(
        pythia_df, qwen_df,
        out_path=fig_dir / f"{file_prefix}layer_patch_overlay_relative_depth.png",
    )

    summary_df = build_overlay_summary(pythia_df, qwen_df)
    summary_path = out_dir / f"{file_prefix}layer_patch_overlay_summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8")
    log(f"[save] {summary_path.resolve()}")

    print_interpretation(summary_df)


if __name__ == "__main__":
    main()
