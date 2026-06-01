#!/usr/bin/env python3
"""
Phase 4b: Attention Pattern Visualisation
=========================================
Generates qualitative attention heatmaps for contrast examples to inspect
whether structured prompting changes how the model routes information at the
final answer position. Defaults to Cell A -> Cell C, and supports Cell B ->
Cell D for noisy contrasts.

For a small subset of contrast examples, the script:
  - loads Pythia-2.8B via TransformerLens
  - reconstructs token-aligned source / donor prompts
  - caches attention patterns at selected layers
  - extracts final-token attention over all prior tokens
  - saves side-by-side head-averaged heatmaps for source vs donor cells
  - saves per-head attention values as JSON for later inspection

Outputs:
  figures/phase_4b_attention/<model>/*.png
  results/phase_4b_attention/<model>/*.json
  results/phase_4b_attention/<model>/attention_manifest.json
"""

import argparse
import gc
import json
import re
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    from scripts.utils.contrast_config import contrast_path_for, model_file_prefix, output_prefix_for
except ModuleNotFoundError:
    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))
    from scripts.utils.contrast_config import contrast_path_for, model_file_prefix, output_prefix_for


VERBOSE = False
REGION_ORDER = [
    "few_shot_prefix",
    "final_example_fact_1",
    "final_example_fact_2",
    "final_example_question",
    "final_example_step_1",
    "final_example_step_2",
    "final_example_answer_cue",
]
REGION_LABELS = {
    "few_shot_prefix": "Few-shot prefix",
    "final_example_fact_1": "Fact 1",
    "final_example_fact_2": "Fact 2",
    "final_example_question": "Question",
    "final_example_step_1": "Step 1",
    "final_example_step_2": "Step 2",
    "final_example_answer_cue": "Answer cue",
}
REGION_COLOURS = {
    "few_shot_prefix": "#d9d9d9",
    "final_example_fact_1": "#f4a261",
    "final_example_fact_2": "#2a9d8f",
    "final_example_question": "#457b9d",
    "final_example_step_1": "#e76f51",
    "final_example_step_2": "#8ab17d",
    "final_example_answer_cue": "#6d597a",
}


def materialise_prompt(cell_dict, tokenizer) -> str:
    """
    Reconstruct the exact runnable model input from the stored cell schema.

    Supports both the new schema (dict with 'prompt', 'prefix_eos_pad',
    optional 'inline_eos_filler') and legacy format (plain string).
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


def log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def vlog(msg: str) -> None:
    if VERBOSE:
        log(msg)


def fmt_s(seconds: float) -> str:
    return f"{seconds:.1f}s" if seconds < 60 else f"{int(seconds // 60)}m {seconds % 60:.1f}s"


def cuda_mem(device: str) -> str:
    if not device.startswith("cuda") or not torch.cuda.is_available():
        return ""
    try:
        alloc = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        return f"  [VRAM alloc={alloc:.2f}GB res={reserved:.2f}GB]"
    except Exception:
        return ""


def sanitise_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("_") or "example"


def format_token_label(token: str, max_len: int = 18) -> str:
    token = token.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
    if token == "":
        token = "<empty>"
    if token.isspace():
        token = token.replace(" ", "·")
    if len(token) > max_len:
        token = token[: max_len - 3] + "..."
    return token


def build_comparison_labels(
    labels_a: list[str],
    labels_c: list[str],
    source_cell: str = "A",
    donor_cell: str = "C",
) -> list[str]:
    combined = []
    for la, lc in zip(labels_a, labels_c):
        if la == lc:
            combined.append(la)
        else:
            combined.append(f"{source_cell}:{la}\n{donor_cell}:{lc}")
    return combined


def find_line_end(prompt: str, start_idx: int) -> int:
    line_end = prompt.find("\n", start_idx)
    return len(prompt) if line_end == -1 else line_end


def parse_prompt_regions(prompt: str) -> list[dict]:
    final_fact_1 = prompt.rfind("Fact 1:")
    if final_fact_1 == -1:
        return []

    final_slice = prompt[final_fact_1:]
    relative_fact_2 = final_slice.find("\nFact 2:")
    relative_q = final_slice.find("\n\nQ:")
    relative_step_1 = final_slice.find("\nStep 1:")
    relative_step_2 = final_slice.find("\nStep 2:")
    relative_answer = final_slice.find("\nAnswer:")
    relative_a = final_slice.find("\nA:")

    if relative_fact_2 == -1 or relative_q == -1:
        return []

    final_fact_2 = final_fact_1 + relative_fact_2 + 1
    final_question = final_fact_1 + relative_q + 2

    if relative_answer != -1:
        answer_marker = final_fact_1 + relative_answer + 1
    elif relative_a != -1:
        answer_marker = final_fact_1 + relative_a + 1
    else:
        answer_marker = len(prompt)

    step_1 = final_fact_1 + relative_step_1 + 1 if relative_step_1 != -1 else None
    step_2 = final_fact_1 + relative_step_2 + 1 if relative_step_2 != -1 else None

    regions = []
    if final_fact_1 > 0:
        regions.append({
            "name": "few_shot_prefix",
            "char_start": 0,
            "char_end": final_fact_1,
        })

    regions.extend([
        {
            "name": "final_example_fact_1",
            "char_start": final_fact_1,
            "char_end": final_fact_2,
        },
        {
            "name": "final_example_fact_2",
            "char_start": final_fact_2,
            "char_end": final_question,
        },
        {
            "name": "final_example_question",
            "char_start": final_question,
            "char_end": step_1 if step_1 is not None else answer_marker,
        },
    ])

    if step_1 is not None and step_2 is not None:
        regions.append({
            "name": "final_example_step_1",
            "char_start": step_1,
            "char_end": step_2,
        })
        regions.append({
            "name": "final_example_step_2",
            "char_start": step_2,
            "char_end": answer_marker,
        })

    regions.append({
        "name": "final_example_answer_cue",
        "char_start": answer_marker,
        "char_end": find_line_end(prompt, answer_marker),
    })
    return regions


def char_span_to_token_span(
    char_start: int,
    char_end: int,
    token_char_spans: list[tuple[int, int]],
) -> tuple[int, int] | None:
    token_start = None
    token_end = None
    for idx, (tok_start, tok_end) in enumerate(token_char_spans):
        if tok_end <= char_start:
            continue
        if tok_start >= char_end:
            break
        if token_start is None:
            token_start = idx
        token_end = idx + 1
    if token_start is None or token_end is None or token_end <= token_start:
        return None
    return token_start, token_end


def build_region_boundaries(prompt: str, token_char_spans: list[tuple[int, int]]) -> list[dict]:
    boundaries = []
    for region in parse_prompt_regions(prompt):
        token_span = char_span_to_token_span(region["char_start"], region["char_end"], token_char_spans)
        if token_span is None:
            continue
        boundaries.append({
            "name": region["name"],
            "label": REGION_LABELS.get(region["name"], region["name"]),
            "char_start": region["char_start"],
            "char_end": region["char_end"],
            "token_start": token_span[0],
            "token_end": token_span[1],
        })
    return boundaries


def summarise_region_attention(attention: np.ndarray, region_boundaries: list[dict]) -> dict:
    summary = {}
    for region in region_boundaries:
        summary[region["name"]] = round(
            float(attention[region["token_start"]:region["token_end"]].sum()),
            6,
        )

    summary["support_facts_total"] = round(
        float(summary.get("final_example_fact_1", 0.0) + summary.get("final_example_fact_2", 0.0)),
        6,
    )
    summary["structured_scaffold_total"] = round(
        float(summary.get("final_example_step_1", 0.0) + summary.get("final_example_step_2", 0.0)),
        6,
    )
    return summary


def lookup_region_name(token_index: int, region_boundaries: list[dict]) -> str | None:
    for region in region_boundaries:
        if region["token_start"] <= token_index < region["token_end"]:
            return region["name"]
    return None


def top_attention_tokens(
    attention: np.ndarray,
    token_labels: list[str],
    token_text: list[str],
    region_boundaries: list[dict],
    eos_token: str,
    top_k: int = 5,
) -> list[dict]:
    ranked_indices = np.argsort(attention)[::-1]
    top_tokens = []
    for token_index in ranked_indices:
        raw_text = token_text[token_index]
        if eos_token and raw_text == eos_token:
            continue
        top_tokens.append({
            "token_index": int(token_index),
            "token_label": token_labels[token_index],
            "token_text": raw_text,
            "attention": round(float(attention[token_index]), 6),
            "region": lookup_region_name(int(token_index), region_boundaries),
        })
        if len(top_tokens) >= top_k:
            break
    return top_tokens


def build_condition_summary(
    condition_name: str,
    prompt: str,
    token_ids: list[int],
    token_text: list[str],
    token_labels: list[str],
    token_char_spans: list[tuple[int, int]],
    attention: np.ndarray,
    eos_token: str,
) -> dict:
    region_boundaries = build_region_boundaries(prompt, token_char_spans)
    region_attention_summary = summarise_region_attention(attention, region_boundaries)
    max_token_index = int(np.argmax(attention))
    return {
        "condition": condition_name,
        "token_ids": token_ids,
        "token_text": token_text,
        "token_labels": token_labels,
        "region_boundaries": region_boundaries,
        "region_attention_summary": region_attention_summary,
        "max_attended_token_index": max_token_index,
        "max_attended_token_label": token_labels[max_token_index],
        "max_attended_token_text": token_text[max_token_index],
        "max_attended_token_region": lookup_region_name(max_token_index, region_boundaries),
        "top_tokens": top_attention_tokens(
            attention=attention,
            token_labels=token_labels,
            token_text=token_text,
            region_boundaries=region_boundaries,
            eos_token=eos_token,
        ),
        "head_averaged_attention": attention.tolist(),
    }


def load_contrast_file(path: str, source_cell: str = "A", donor_cell: str = "C") -> list:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Contrast file not found: {p}")

    source_key = f"cell_{source_cell}"
    donor_key = f"cell_{donor_cell}"

    log(f"[load] Reading Cell {source_cell} -> Cell {donor_cell} contrast examples from {p}")
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list) or len(data) == 0:
        raise ValueError("Contrast file must be a non-empty JSON list.")

    valid = []
    for i, ex in enumerate(data):
        if not isinstance(ex, dict):
            log(f"  WARN: contrast index {i}: not a dict, skipping")
            continue
        missing = [k for k in ("example_id", "gold_answer", source_key, donor_key) if k not in ex]
        if missing:
            log(f"  WARN: contrast index {i}: missing keys {missing}, skipping")
            continue
        if not ex["gold_answer"]:
            log(f"  WARN: contrast index {i}: empty gold_answer, skipping")
            continue
        valid.append(ex)

    log(f"[load] {len(valid)}/{len(data)} contrast examples validated")
    return valid


def load_dataset_index(path: str) -> dict:
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
            by_id[str(eid)] = ex
    log(f"[load] {len(by_id)} examples indexed from dataset")
    return by_id


def load_data(
    contrast_path: str,
    dataset_path: str,
    tokenizer,
    source_cell: str = "A",
    donor_cell: str = "C",
) -> list:
    """
    Resolve prompts for contrast examples, preferring embedded prompts and
    falling back to dataset.json when available.
    """
    examples = load_contrast_file(contrast_path, source_cell, donor_cell)

    dataset_index = None
    if Path(dataset_path).exists():
        dataset_index = load_dataset_index(dataset_path)
    else:
        log(f"[load] Dataset fallback not found at {dataset_path}; continuing with embedded prompts only")

    def _extract(cell_value):
        if cell_value is None:
            return None
        try:
            return materialise_prompt(cell_value, tokenizer)
        except Exception:
            return None

    resolved = []
    source_key = f"cell_{source_cell}"
    donor_key = f"cell_{donor_cell}"
    for ex in examples:
        eid = str(ex["example_id"])
        gold = ex["gold_answer"]

        prompt_source = _extract(ex.get(source_key))
        prompt_donor = _extract(ex.get(donor_key))

        if (prompt_source is None or prompt_donor is None) and dataset_index is not None:
            ds = dataset_index.get(eid)
            cells = ds.get("cells", {}) if isinstance(ds, dict) else {}
            if prompt_source is None:
                prompt_source = _extract(cells.get(source_cell))
            if prompt_donor is None:
                prompt_donor = _extract(cells.get(donor_cell))

        if not prompt_source:
            log(f"  WARN: {eid} - no runnable prompt for Cell {source_cell}, skipping")
            continue
        if not prompt_donor:
            log(f"  WARN: {eid} - no runnable prompt for Cell {donor_cell}, skipping")
            continue

        resolved.append({
            "example_id": eid,
            "gold_answer": gold,
            "domain": ex.get("domain", ""),
            "prompt_source": prompt_source,
            "prompt_donor": prompt_donor,
        })

    if not resolved:
        raise ValueError(
            f"No valid contrast examples with resolved Cell {source_cell} / Cell {donor_cell} prompts."
        )

    return resolved


def load_model(model_name: str, device: str):
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

    log(
        f"[model] Loaded in {fmt_s(time.time() - t0)} | "
        f"n_layers={model.cfg.n_layers} n_heads={model.cfg.n_heads} d_model={model.cfg.d_model}"
        f"{cuda_mem(device)}"
    )
    return model


def tokenise_prompt(
    model,
    prompt: str,
) -> tuple[torch.Tensor, list[int], list[str], list[str], list[tuple[int, int]]]:
    tokens = model.to_tokens(prompt)
    token_ids = tokens[0].tolist()
    token_text = [model.tokenizer.decode([tid]) for tid in token_ids]
    token_labels = [format_token_label(tok) for tok in token_text]
    token_char_spans = []
    cursor = 0
    for text in token_text:
        start = cursor
        cursor += len(text)
        token_char_spans.append((start, cursor))
    return tokens, token_ids, token_text, token_labels, token_char_spans


def extract_attention(model, prompt: str, layers: list[int]) -> dict:
    tokens, token_ids, token_text, token_labels, token_char_spans = tokenise_prompt(model, prompt)
    hook_names = {f"blocks.{layer}.attn.hook_pattern" for layer in layers}

    with torch.no_grad():
        _, cache = model.run_with_cache(
            tokens,
            names_filter=lambda name: name in hook_names,
        )

    seq_len = tokens.shape[1]
    final_token_index = seq_len - 1
    layer_data = {}
    for layer in layers:
        pattern = cache[f"blocks.{layer}.attn.hook_pattern"][0]  # [heads, dest, src]
        final_slice = pattern[:, final_token_index, :].detach().float().cpu()
        average = final_slice.mean(dim=0)
        layer_data[layer] = {
            "per_head": final_slice.numpy(),
            "average": average.numpy(),
        }

    del cache, tokens
    return {
        "token_ids": token_ids,
        "token_text": token_text,
        "token_labels": token_labels,
        "token_char_spans": token_char_spans,
        "final_token_index": final_token_index,
        "layers": layer_data,
    }


def assert_token_alignment(
    example_id: str,
    token_ids_a: list[int],
    token_ids_c: list[int],
    source_cell: str = "A",
    donor_cell: str = "C",
) -> bool:
    if len(token_ids_a) != len(token_ids_c):
        log(
            f"  ALIGNMENT FAIL: {example_id} - Cell {source_cell}={len(token_ids_a)} tokens vs "
            f"Cell {donor_cell}={len(token_ids_c)} tokens, skipping"
        )
        return False
    return True


def sparse_tick_positions(seq_len: int, target_ticks: int = 8) -> list[int]:
    if seq_len <= target_ticks:
        return list(range(seq_len))
    return sorted(set(np.linspace(0, seq_len - 1, num=target_ticks, dtype=int).tolist()))


def choose_summary_regions(region_boundaries_a: list[dict], region_boundaries_c: list[dict]) -> list[str]:
    available = {region["name"] for region in region_boundaries_a} | {region["name"] for region in region_boundaries_c}
    return [name for name in REGION_ORDER if name in available and name != "few_shot_prefix"]


def overlay_region_boundaries(ax, region_boundaries: list[dict], y_pos: float) -> None:
    for region in region_boundaries:
        colour = REGION_COLOURS.get(region["name"], "#666666")
        start = region["token_start"]
        end = region["token_end"]
        ax.axvline(start - 0.5, color=colour, linewidth=1.0, alpha=0.75)
        ax.axvline(end - 0.5, color=colour, linewidth=1.0, alpha=0.75)
        center = (start + end - 1) / 2
        ax.text(
            center,
            y_pos,
            region["label"],
            ha="center",
            va="bottom",
            fontsize=8,
            color=colour,
            clip_on=False,
        )


def plot_heatmap(
    example_id: str,
    layer: int,
    labels_a: list[str],
    labels_c: list[str],
    attention_a: np.ndarray,
    attention_c: np.ndarray,
    region_boundaries_a: list[dict],
    region_boundaries_c: list[dict],
    region_summary_a: dict,
    region_summary_c: dict,
    output_path: Path,
    source_cell: str = "A",
    donor_cell: str = "C",
) -> None:
    data_a = attention_a[np.newaxis, :]
    data_c = attention_c[np.newaxis, :]
    vmax = max(float(data_a.max()), float(data_c.max()), 1e-9)
    summary_regions = choose_summary_regions(region_boundaries_a, region_boundaries_c)

    fig, axes = plt.subplots(
        nrows=3,
        ncols=1,
        figsize=(13.5, 8.8),
        gridspec_kw={"height_ratios": [1.0, 1.0, 1.45]},
    )

    for ax, data, title, labels, region_boundaries in [
        (axes[0], data_a, f"Cell {source_cell} (direct)", labels_a, region_boundaries_a),
        (axes[1], data_c, f"Cell {donor_cell} (structured)", labels_c, region_boundaries_c),
    ]:
        im = ax.imshow(data, aspect="auto", cmap="viridis", interpolation="nearest", vmin=0.0, vmax=vmax)
        ax.set_yticks([0])
        ax.set_yticklabels(["Final token"])
        ax.set_title(title, fontsize=11, loc="left")
        ticks = sparse_tick_positions(len(labels))
        ax.set_xticks(ticks)
        ax.set_xticklabels([labels[i] for i in ticks], fontsize=8)
        overlay_region_boundaries(ax, region_boundaries, y_pos=0.62)

    axes[0].tick_params(axis="x", labelbottom=False)
    axes[1].set_xlabel("Token position (sparse token labels)", fontsize=10)

    summary_ax = axes[2]
    if summary_regions:
        positions = np.arange(len(summary_regions))
        width = 0.36
        summary_ax.bar(
            positions - width / 2,
            [region_summary_a.get(name, 0.0) for name in summary_regions],
            width=width,
            color="#457b9d",
            label=f"Cell {source_cell}",
        )
        summary_ax.bar(
            positions + width / 2,
            [region_summary_c.get(name, 0.0) for name in summary_regions],
            width=width,
            color="#e76f51",
            label=f"Cell {donor_cell}",
        )
        summary_ax.set_xticks(positions)
        summary_ax.set_xticklabels(
            [REGION_LABELS.get(name, name) for name in summary_regions],
            rotation=25,
            ha="right",
        )
        summary_ax.set_ylabel("Attention mass", fontsize=10)
        summary_ax.set_title("Region-level final-token attention", fontsize=11, loc="left")
        summary_ax.legend(fontsize=9)
    else:
        summary_ax.axis("off")

    fig.suptitle(f"{example_id} | Layer {layer} | Final-token attention routing", fontsize=13, y=0.985)

    fig.subplots_adjust(
        left=0.08,
        right=0.88,
        top=0.93,
        bottom=0.13,
        hspace=0.72,
    )
    cbar_ax = fig.add_axes([0.905, 0.56, 0.018, 0.26])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Attention weight", fontsize=9)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def write_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _model_slug(model_name: str) -> str:
    return model_name.split("/")[-1].lower()


def main():
    global VERBOSE

    parser = argparse.ArgumentParser(
        description="Phase 4b: attention heatmaps for clean contrast examples"
    )
    parser.add_argument(
        "--contrast-file",
        type=str,
        default=None,
        help="Path to contrast examples JSON "
             "(default: clean A/C file, or noisy B/D file when --source-cell B --donor-cell D)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to full dataset JSON for prompt fallback "
             "(default: dataset/processed/<model-slug>/dataset.json)",
    )
    parser.add_argument(
        "--figdir",
        type=str,
        default=None,
        help="Output directory for PNG figures "
             "(default: figures/phase_4b_attention/<model-slug>/)",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=None,
        help="Output directory for JSON results "
             "(default: results/phase_4b_attention/<model-slug>/)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="EleutherAI/pythia-2.8b",
        help="HuggingFace model name for TransformerLens",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device: cuda or cpu",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=[20, 30, 31],
        help="Layers to inspect",
    )
    parser.add_argument(
        "--source-cell",
        type=str,
        default="A",
        choices=["A", "B", "C", "D", "E"],
        help="Baseline/source cell to visualise (default: A)",
    )
    parser.add_argument(
        "--donor-cell",
        type=str,
        default="C",
        choices=["A", "B", "C", "D", "E"],
        help="Structured/donor cell to visualise (default: C)",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default=None,
        help="Optional filename prefix override. Defaults come from source/donor contrast routing.",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=3,
        help="Number of clean contrast examples to visualise",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    args = parser.parse_args()
    VERBOSE = args.verbose

    slug = _model_slug(args.model)
    source_cell = args.source_cell.upper()
    donor_cell = args.donor_cell.upper()
    file_prefix = model_file_prefix(slug, output_prefix_for(source_cell, donor_cell, args.output_prefix))
    default_contrast = contrast_path_for(slug, source_cell, donor_cell)
    contrast_file = args.contrast_file or default_contrast
    dataset_path  = args.dataset       or f"dataset/processed/{slug}/dataset.json"
    fig_dir_path = args.figdir or f"figures/phase_4b_attention/{slug}"
    out_dir_path = args.outdir or f"results/phase_4b_attention/{slug}"

    fig_dir = Path(fig_dir_path)
    out_dir = Path(out_dir_path)
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    log("=" * 72)
    log("Phase 4b: Attention Pattern Visualisation")
    log("=" * 72)
    log(f"  Model:          {args.model}")
    log(f"  Contrast file:  {contrast_file}")
    log(f"  Dataset:        {dataset_path}")
    log(f"  Source cell:    {source_cell}")
    log(f"  Donor cell:     {donor_cell}")
    log(f"  Figure dir:     {fig_dir_path}")
    log(f"  Results dir:    {out_dir_path}")
    log(f"  Layers:         {args.layers}")
    log(f"  Num examples:   {args.num_examples}")
    log("")

    overall_t0 = time.time()
    if not Path(contrast_file).exists():
        raise FileNotFoundError(f"Contrast file not found: {contrast_file}")

    model = load_model(args.model, args.device)
    for layer in args.layers:
        if layer < 0 or layer >= model.cfg.n_layers:
            raise ValueError(
                f"Requested layer {layer} is out of range for {args.model} "
                f"(valid: 0-{model.cfg.n_layers - 1})."
            )
    examples = load_data(
        contrast_file,
        dataset_path,
        model.tokenizer,
        source_cell=source_cell,
        donor_cell=donor_cell,
    )
    examples = examples[: args.num_examples]

    log(f"[run] Analysing {len(examples)} Cell {source_cell} -> Cell {donor_cell} contrast examples")
    manifest = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": args.model,
        "device": args.device,
        "contrast_file": contrast_file,
        "dataset": dataset_path,
        "source_cell": source_cell,
        "donor_cell": donor_cell,
        "layers": args.layers,
        "num_examples_requested": args.num_examples,
        "examples_processed": [],
    }

    skipped_alignment = 0

    for idx, ex in enumerate(examples, start=1):
        ex_t0 = time.time()
        example_id = ex["example_id"]
        vlog(f"[example:{example_id}] Starting example {idx}/{len(examples)}")

        attention_a = extract_attention(model, ex["prompt_source"], args.layers)
        attention_c = extract_attention(model, ex["prompt_donor"], args.layers)

        if not assert_token_alignment(
            example_id,
            attention_a["token_ids"],
            attention_c["token_ids"],
            source_cell,
            donor_cell,
        ):
            skipped_alignment += 1
            continue

        example_manifest = {
            "example_id": example_id,
            "gold_answer": ex["gold_answer"],
            "domain": ex["domain"],
            "final_token_index": attention_a["final_token_index"],
            "sequence_length": len(attention_a["token_ids"]),
            "layers": [],
        }

        for layer in args.layers:
            slug = sanitise_filename(example_id)
            fig_path = fig_dir / f"{file_prefix}{slug}_layer_{layer}_comparison.png"
            json_path = out_dir / f"{file_prefix}{slug}_layer_{layer}_comparison.json"
            summary_path = out_dir / f"{file_prefix}{slug}_layer_{layer}_summary.json"

            avg_a = attention_a["layers"][layer]["average"]
            avg_c = attention_c["layers"][layer]["average"]
            summary_a = build_condition_summary(
                condition_name=f"Cell {source_cell} (direct)",
                prompt=ex["prompt_source"],
                token_ids=attention_a["token_ids"],
                token_text=attention_a["token_text"],
                token_labels=attention_a["token_labels"],
                token_char_spans=attention_a["token_char_spans"],
                attention=avg_a,
                eos_token=model.tokenizer.eos_token,
            )
            summary_c = build_condition_summary(
                condition_name=f"Cell {donor_cell} (structured)",
                prompt=ex["prompt_donor"],
                token_ids=attention_c["token_ids"],
                token_text=attention_c["token_text"],
                token_labels=attention_c["token_labels"],
                token_char_spans=attention_c["token_char_spans"],
                attention=avg_c,
                eos_token=model.tokenizer.eos_token,
            )

            plot_heatmap(
                example_id=example_id,
                layer=layer,
                labels_a=attention_a["token_labels"],
                labels_c=attention_c["token_labels"],
                attention_a=avg_a,
                attention_c=avg_c,
                region_boundaries_a=summary_a["region_boundaries"],
                region_boundaries_c=summary_c["region_boundaries"],
                region_summary_a=summary_a["region_attention_summary"],
                region_summary_c=summary_c["region_attention_summary"],
                output_path=fig_path,
                source_cell=source_cell,
                donor_cell=donor_cell,
            )

            source_key = f"cell_{source_cell}"
            donor_key = f"cell_{donor_cell}"
            source_top_key = f"top_tokens_cell_{source_cell}"
            donor_top_key = f"top_tokens_cell_{donor_cell}"

            payload = {
                "example_id": example_id,
                "gold_answer": ex["gold_answer"],
                "domain": ex["domain"],
                "layer": layer,
                "final_token_index": attention_a["final_token_index"],
                "token_ids": attention_a["token_ids"],
                "region_attention_summary": {
                    source_key: summary_a["region_attention_summary"],
                    donor_key: summary_c["region_attention_summary"],
                },
                "region_boundaries": {
                    source_key: summary_a["region_boundaries"],
                    donor_key: summary_c["region_boundaries"],
                },
                source_top_key: summary_a["top_tokens"],
                donor_top_key: summary_c["top_tokens"],
                source_key: {
                    "condition": f"Cell {source_cell} (direct)",
                    "per_head_attention": attention_a["layers"][layer]["per_head"].tolist(),
                    "head_averaged_attention": avg_a.tolist(),
                    "token_text": attention_a["token_text"],
                    "token_labels": attention_a["token_labels"],
                    "region_boundaries": summary_a["region_boundaries"],
                    "region_attention_summary": summary_a["region_attention_summary"],
                    "max_attended_token_index": summary_a["max_attended_token_index"],
                    "max_attended_token_label": summary_a["max_attended_token_label"],
                    "max_attended_token_text": summary_a["max_attended_token_text"],
                    "max_attended_token_region": summary_a["max_attended_token_region"],
                    source_top_key: summary_a["top_tokens"],
                },
                donor_key: {
                    "condition": f"Cell {donor_cell} (structured)",
                    "per_head_attention": attention_c["layers"][layer]["per_head"].tolist(),
                    "head_averaged_attention": avg_c.tolist(),
                    "token_text": attention_c["token_text"],
                    "token_labels": attention_c["token_labels"],
                    "region_boundaries": summary_c["region_boundaries"],
                    "region_attention_summary": summary_c["region_attention_summary"],
                    "max_attended_token_index": summary_c["max_attended_token_index"],
                    "max_attended_token_label": summary_c["max_attended_token_label"],
                    "max_attended_token_text": summary_c["max_attended_token_text"],
                    "max_attended_token_region": summary_c["max_attended_token_region"],
                    donor_top_key: summary_c["top_tokens"],
                },
            }
            write_json(payload, json_path)

            compact_summary = {
                "example_id": example_id,
                "layer": layer,
                "final_token_index": attention_a["final_token_index"],
                source_key: {
                    "region_attention_summary": summary_a["region_attention_summary"],
                    "max_attended_token_index": summary_a["max_attended_token_index"],
                    "max_attended_token_label": summary_a["max_attended_token_label"],
                    "max_attended_token_region": summary_a["max_attended_token_region"],
                    source_top_key: summary_a["top_tokens"],
                },
                donor_key: {
                    "region_attention_summary": summary_c["region_attention_summary"],
                    "max_attended_token_index": summary_c["max_attended_token_index"],
                    "max_attended_token_label": summary_c["max_attended_token_label"],
                    "max_attended_token_region": summary_c["max_attended_token_region"],
                    donor_top_key: summary_c["top_tokens"],
                },
                "region_boundaries": {
                    source_key: summary_a["region_boundaries"],
                    donor_key: summary_c["region_boundaries"],
                },
            }
            write_json(compact_summary, summary_path)

            example_manifest["layers"].append({
                "layer": layer,
                "figure_path": str(fig_path),
                "result_path": str(json_path),
                "summary_path": str(summary_path),
                f"cell_{source_cell}_attention_sum": round(float(avg_a.sum()), 6),
                f"cell_{donor_cell}_attention_sum": round(float(avg_c.sum()), 6),
                "region_attention_summary": {
                    source_key: summary_a["region_attention_summary"],
                    donor_key: summary_c["region_attention_summary"],
                },
                source_top_key: summary_a["top_tokens"],
                donor_top_key: summary_c["top_tokens"],
                "region_boundaries": {
                    source_key: summary_a["region_boundaries"],
                    donor_key: summary_c["region_boundaries"],
                },
            })

        manifest["examples_processed"].append(example_manifest)
        log(
            f"[{idx}/{len(examples)}] {example_id} complete in {fmt_s(time.time() - ex_t0)}"
            f"{cuda_mem(args.device)}"
        )

        if args.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    manifest["examples_completed"] = len(manifest["examples_processed"])
    manifest["examples_skipped_alignment"] = skipped_alignment
    manifest["total_runtime_seconds"] = round(time.time() - overall_t0, 2)

    manifest_path = out_dir / f"{file_prefix}attention_manifest.json"
    write_json(manifest, manifest_path)

    log("")
    log(f"[output] Manifest -> {manifest_path}")
    log(f"[done] Phase 4b complete in {fmt_s(time.time() - overall_t0)}")


if __name__ == "__main__":
    main()
