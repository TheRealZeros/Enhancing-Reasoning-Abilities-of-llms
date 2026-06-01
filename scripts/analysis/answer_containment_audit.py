#!/usr/bin/env python3
"""
Model-agnostic answer-containment audit for existing behavioural outputs.

This is a secondary diagnostic over Phase 2 evaluation_results.csv. It does
not rerun model inference and does not replace strict exact-match scoring.

python scripts/analysis/answer_containment_audit.py --model Qwen/Qwen2.5-3B
python scripts/analysis/answer_containment_audit.py --model EleutherAI/pythia-2.8b

"""

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Optional

try:
    from scripts.utils.contrast_config import model_file_prefix
except ModuleNotFoundError:
    import sys

    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))
    from scripts.utils.contrast_config import model_file_prefix


CELL_ORDER = ["A", "B", "C", "D", "E"]
LABEL_EXACT = "exact_match"
LABEL_CONTAINS = "contains_correct_answer"
LABEL_WRONG = "wrong"
LABELS = [LABEL_EXACT, LABEL_CONTAINS, LABEL_WRONG]


def _model_slug(model_name: str) -> str:
    return model_name.split("/")[-1].lower()


def normalise_answer(text: str) -> str:
    """Conservative text normalisation for exact/containment diagnostics."""
    text = (text or "").lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = text.rstrip(".,;:!?")

    for prefix in ("the ", "a ", "an "):
        if text.startswith(prefix) and len(text) > len(prefix):
            text = text[len(prefix):].strip()
            break

    return text


def robust_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    low = str(value).strip().lower()
    if low in {"true", "1", "yes"}:
        return True
    if low in {"false", "0", "no"}:
        return False
    raise ValueError(f"Cannot parse boolean value: {value!r}")


def load_dataset_domains(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {
        str(ex.get("id") or ex.get("example_id")): ex.get("domain", "")
        for ex in data
        if ex.get("id") or ex.get("example_id")
    }


def classify_answer(generated_raw: str, gold: str, exact_match_correct: bool) -> tuple[str, str, str]:
    normalised_gold = normalise_answer(gold)
    normalised_generated = normalise_answer(generated_raw)

    if exact_match_correct or normalised_generated == normalised_gold:
        return LABEL_EXACT, normalised_gold, normalised_generated
    if normalised_gold and normalised_gold in normalised_generated:
        return LABEL_CONTAINS, normalised_gold, normalised_generated
    return LABEL_WRONG, normalised_gold, normalised_generated


def load_audit_rows(evaluation_path: Path, dataset_path: Path) -> list[dict]:
    domains_by_id = load_dataset_domains(dataset_path)
    rows = []

    with open(evaluation_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {
            "example_id",
            "cell",
            "gold_answer",
            "correct",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{evaluation_path}: missing required columns {sorted(missing)}")

        for row in reader:
            example_id = row["example_id"]
            generated = row.get("generated_answer_raw") or row.get("generated_answer") or ""
            exact_correct = robust_bool(row["correct"])
            label, normalised_gold, normalised_generated = classify_answer(
                generated_raw=generated,
                gold=row["gold_answer"],
                exact_match_correct=exact_correct,
            )
            rows.append({
                "example_id": example_id,
                "domain": row.get("domain") or domains_by_id.get(example_id, ""),
                "cell": row["cell"],
                "gold_answer": row["gold_answer"],
                "generated_answer": generated,
                "normalised_gold": normalised_gold,
                "normalised_generated": normalised_generated,
                "exact_match_correct": str(exact_correct),
                "answer_containment_label": label,
            })

    return rows


def summarise(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["cell"]].append(row)

    summary_rows = []
    for cell in sorted(grouped, key=lambda c: CELL_ORDER.index(c) if c in CELL_ORDER else c):
        cell_rows = grouped[cell]
        total = len(cell_rows)
        counts = {label: sum(r["answer_containment_label"] == label for r in cell_rows) for label in LABELS}
        exact = counts[LABEL_EXACT]
        contains = counts[LABEL_CONTAINS]
        wrong = counts[LABEL_WRONG]
        summary_rows.append({
            "cell": cell,
            "total": total,
            "exact_match_count": exact,
            "contains_correct_answer_count": contains,
            "wrong_count": wrong,
            "strict_exact_accuracy": exact / total if total else 0.0,
            "contains_answer_accuracy": (exact + contains) / total if total else 0.0,
            "wrong_rate": wrong / total if total else 0.0,
        })
    return summary_rows


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt_pct(value: float) -> str:
    return f"{value:.1%}"


def md_escape(value: str) -> str:
    return str(value).replace("\n", "<br>").replace("|", "\\|")


def rows_for_label(rows: list[dict], label: str, limit: int = 12) -> list[dict]:
    return [row for row in rows if row["answer_containment_label"] == label][:limit]


def summary_by_cell(summary_rows: list[dict]) -> dict[str, dict]:
    return {row["cell"]: row for row in summary_rows}


def contrast_line(summary_map: dict[str, dict], left: str, right: str) -> str:
    if left not in summary_map or right not in summary_map:
        return f"Cells {left} and/or {right} were not available in the summary."

    lrow = summary_map[left]
    rrow = summary_map[right]
    exact_delta = rrow["strict_exact_accuracy"] - lrow["strict_exact_accuracy"]
    contain_delta = rrow["contains_answer_accuracy"] - lrow["contains_answer_accuracy"]
    return (
        f"Cell {left} strict exact accuracy is {fmt_pct(lrow['strict_exact_accuracy'])}; "
        f"Cell {right} strict exact accuracy is {fmt_pct(rrow['strict_exact_accuracy'])} "
        f"(delta {exact_delta:+.1%}). Containment-aware accuracy changes from "
        f"{fmt_pct(lrow['contains_answer_accuracy'])} to "
        f"{fmt_pct(rrow['contains_answer_accuracy'])} (delta {contain_delta:+.1%})."
    )


def write_markdown(path: Path, model_name: str, rows: list[dict], summary_rows: list[dict]) -> None:
    summary_map = summary_by_cell(summary_rows)
    lines = [
        "# Model-Agnostic Answer-Containment Audit",
        "",
        f"Model: `{model_name}`",
        "",
        "## Purpose",
        "",
        "This audit checks whether a generation that fails strict exact-match still contains the gold answer. It is an evaluation-framework expansion over existing behavioural outputs, not a new runnable experiment stage.",
        "",
        "Strict exact-match remains the primary behavioural metric used by the experiment. The containment-aware score is a secondary diagnostic for understanding answer quality, especially when structured prompts produce extra wording around an otherwise correct answer.",
        "",
        "Labels are mutually exclusive: `exact_match`, `contains_correct_answer`, and `wrong`.",
        "",
        "## Per-Cell Summary",
        "",
        "| Cell | Total | Exact match | Contains correct answer | Wrong | Strict exact accuracy | Contains-answer accuracy | Wrong rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for row in summary_rows:
        lines.append(
            f"| {row['cell']} | {row['total']} | {row['exact_match_count']} | "
            f"{row['contains_correct_answer_count']} | {row['wrong_count']} | "
            f"{fmt_pct(row['strict_exact_accuracy'])} | "
            f"{fmt_pct(row['contains_answer_accuracy'])} | "
            f"{fmt_pct(row['wrong_rate'])} |"
        )

    lines.extend([
        "",
        "## A vs C Interpretation",
        "",
        contrast_line(summary_map, "A", "C"),
        "",
        "Cell A is Direct/Clean and Cell C is Structured/Clean. A containment-aware comparison can reveal when structured prompting produces the right answer with additional text, even when strict exact-match marks it incorrect.",
        "",
        "## B vs D Interpretation",
        "",
        contrast_line(summary_map, "B", "D"),
        "",
        "Cell B is Direct/Noisy and Cell D is Structured/Noisy. This comparison is useful for checking whether structured reasoning recovers answers in noisy contexts beyond what strict exact-match alone captures.",
        "",
        "## Exact-Match Failures That Contain the Gold Answer",
        "",
        "| example_id | domain | cell | gold answer | generated answer |",
        "|---|---|---|---|---|",
    ])

    for row in rows_for_label(rows, LABEL_CONTAINS):
        lines.append(
            f"| {md_escape(row['example_id'])} | {md_escape(row['domain'])} | "
            f"{md_escape(row['cell'])} | {md_escape(row['gold_answer'])} | "
            f"{md_escape(row['generated_answer'])} |"
        )

    lines.extend([
        "",
        "## Examples Labelled Wrong",
        "",
        "| example_id | domain | cell | gold answer | generated answer |",
        "|---|---|---|---|---|",
    ])

    for row in rows_for_label(rows, LABEL_WRONG):
        lines.append(
            f"| {md_escape(row['example_id'])} | {md_escape(row['domain'])} | "
            f"{md_escape(row['cell'])} | {md_escape(row['gold_answer'])} | "
            f"{md_escape(row['generated_answer'])} |"
        )

    lines.extend([
        "",
        "## Thesis-Safe Interpretation",
        "",
        "Use this audit to separate two phenomena: genuinely wrong answers and answers that include the correct answer but fail strict formatting. This is especially relevant for structured prompts, which may encourage explanations or more specific noun phrases.",
        "",
        "This diagnostic should be reported alongside strict exact-match accuracy, not instead of it. Exact-match is still the primary behavioural score used to define contrast sets for downstream causal analyses.",
        "",
        "## Warning",
        "",
        "This is a simple containment audit, not a semantic equivalence judge. It only checks normalised substring containment and may miss paraphrases or count some ambiguous mentions as containing the answer.",
    ])

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_answer_containment_audit(
    model_name: str,
    dataset_path: Optional[str] = None,
    evaluation_path: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> dict:
    """
    Run the answer-containment audit over existing behavioural outputs.

    This function is importable by Phase 2 post-processing and by other
    analysis scripts. It does not rerun model inference and does not alter
    exact-match scoring.
    """
    slug = _model_slug(model_name)
    resolved_dataset = Path(dataset_path) if dataset_path else Path(f"dataset/processed/{slug}/dataset.json")
    resolved_evaluation = (
        Path(evaluation_path)
        if evaluation_path
        else Path(f"results/phase_2_behaviour/{slug}/{model_file_prefix(slug)}evaluation_results.csv")
    )
    resolved_output_dir = (
        Path(output_dir)
        if output_dir
        else Path(f"results/model_agnostic_evaluation/{slug}")
    )

    rows = load_audit_rows(resolved_evaluation, resolved_dataset)
    summary_rows = summarise(rows)

    file_prefix = model_file_prefix(slug)
    audit_path = resolved_output_dir / f"{file_prefix}answer_containment_audit.csv"
    summary_path = resolved_output_dir / f"{file_prefix}answer_containment_summary.csv"
    markdown_path = resolved_output_dir / f"{file_prefix}answer_containment_audit.md"

    write_csv(
        audit_path,
        rows,
        [
            "example_id",
            "domain",
            "cell",
            "gold_answer",
            "generated_answer",
            "normalised_gold",
            "normalised_generated",
            "exact_match_correct",
            "answer_containment_label",
        ],
    )
    write_csv(
        summary_path,
        summary_rows,
        [
            "cell",
            "total",
            "exact_match_count",
            "contains_correct_answer_count",
            "wrong_count",
            "strict_exact_accuracy",
            "contains_answer_accuracy",
            "wrong_rate",
        ],
    )
    write_markdown(markdown_path, model_name, rows, summary_rows)

    return {
        "model_name": model_name,
        "model_slug": slug,
        "dataset_path": str(resolved_dataset),
        "evaluation_path": str(resolved_evaluation),
        "output_dir": str(resolved_output_dir),
        "audit_path": str(audit_path),
        "summary_path": str(summary_path),
        "markdown_path": str(markdown_path),
        "summary_rows": summary_rows,
        "n_rows": len(rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Model-agnostic answer-containment audit over existing behavioural outputs."
    )
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--evaluation", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    result = run_answer_containment_audit(
        model_name=args.model,
        dataset_path=str(args.dataset) if args.dataset else None,
        evaluation_path=str(args.evaluation) if args.evaluation else None,
        output_dir=str(args.output_dir) if args.output_dir else None,
    )

    print(f"[audit] rows:    {result['audit_path']}")
    print(f"[audit] summary: {result['summary_path']}")
    print(f"[audit] report:  {result['markdown_path']}")
    for row in result["summary_rows"]:
        print(
            f"[audit] Cell {row['cell']}: strict={fmt_pct(row['strict_exact_accuracy'])} "
            f"contains-aware={fmt_pct(row['contains_answer_accuracy'])} "
            f"wrong={fmt_pct(row['wrong_rate'])}"
        )


if __name__ == "__main__":
    main()
