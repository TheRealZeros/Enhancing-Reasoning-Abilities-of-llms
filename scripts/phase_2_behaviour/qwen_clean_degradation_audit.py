#!/usr/bin/env python3
"""
Audit Qwen clean-contrast degradation cases.

Finds examples where Cell A (Direct/Clean) is correct and Cell C
(Structured/Clean) is incorrect, then writes a markdown report explaining
likely structured-clean failure modes without rerunning the model.
"""

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path


FAILURE_CATEGORIES = [
    "exact-match formatting failure",
    "over-generation",
    "wrong bridge entity",
    "wrong second-hop answer",
    "scaffold-following error",
    "answer appears but not in exact-match form",
    "other / unclear",
]


def _model_slug(model_name: str) -> str:
    return model_name.split("/")[-1].lower()


def robust_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    low = str(value).strip().lower()
    if low in {"true", "1", "yes"}:
        return True
    if low in {"false", "0", "no"}:
        return False
    raise ValueError(f"Cannot parse bool value: {value!r}")


def normalise(text: str) -> str:
    text = (text or "").strip().lower()
    text = re.sub(r"^(the answer is|answer:)\s*", "", text)
    return text.strip().rstrip(".,;:!?")


def first_nonempty_line(text: str) -> str:
    for line in (text or "").splitlines():
        line = line.strip()
        if line:
            return line
    return ""


def contains_wordish(haystack: str, needle: str) -> bool:
    return normalise(needle) in normalise(haystack)


def looks_overgenerated(raw: str, normalised: str, gold: str) -> bool:
    raw = raw or ""
    first = first_nonempty_line(raw)
    if "\n" in raw.strip():
        return True
    if len(first.split()) >= max(5, len(gold.split()) + 3):
        return True
    if len(normalised) > len(normalise(gold)) + 20:
        return True
    return False


def follows_scaffold(raw: str) -> bool:
    low = (raw or "").lower()
    return ("step 1:" in low and "step 2:" in low) or ("answer:" not in low)


def classify(
    cell_c_raw: str,
    cell_c_norm: str,
    gold: str,
    bridge: str,
    fact_2: str,
) -> tuple[str, dict]:
    gold_in_raw = contains_wordish(cell_c_raw, gold)
    gold_in_norm = contains_wordish(cell_c_norm, gold)
    bridge_in_raw = contains_wordish(cell_c_raw, bridge)
    overgenerated = looks_overgenerated(cell_c_raw, cell_c_norm, gold)
    scaffold_ok = follows_scaffold(cell_c_raw)
    final_answer_norm = normalise(cell_c_norm)
    gold_norm = normalise(gold)

    wrong_bridge = bool(bridge) and not bridge_in_raw and ("step 1" in cell_c_raw.lower())
    wrong_final = final_answer_norm and final_answer_norm != gold_norm and not gold_in_norm

    if gold_in_raw and final_answer_norm != gold_norm:
        category = "answer appears but not in exact-match form"
    elif overgenerated and gold_in_raw:
        category = "exact-match formatting failure"
    elif overgenerated:
        category = "over-generation"
    elif wrong_bridge:
        category = "wrong bridge entity"
    elif wrong_final:
        category = "wrong second-hop answer"
    elif not scaffold_ok:
        category = "scaffold-following error"
    else:
        category = "other / unclear"

    flags = {
        "cell_c_contains_gold": gold_in_raw or gold_in_norm,
        "cell_c_over_generates_sentence": overgenerated,
        "cell_c_selects_wrong_bridge_entity": wrong_bridge,
        "cell_c_selects_wrong_final_answer": wrong_final,
        "cell_c_follows_structured_scaffold": scaffold_ok,
        "fact_2_mentions_gold": contains_wordish(fact_2, gold),
    }
    return category, flags


def load_dataset(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return {ex["id"]: ex for ex in json.load(f)}


def load_eval_rows(path: Path) -> dict:
    rows_by_id = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows_by_id.setdefault(row["example_id"], {})[row["cell"]] = row
    return rows_by_id


def markdown_bool(value: bool) -> str:
    return "yes" if value else "no"


def write_report(audit_rows: list[dict], summary: Counter, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Qwen Clean Degradation Audit",
        "",
        "Contrast audited: Cell A correct and Cell C incorrect.",
        "",
        "Cell A = Direct/Clean. Cell C = Structured/Clean.",
        "",
        "## Summary By Failure Category",
        "",
        "| Failure category | Count |",
        "|---|---:|",
    ]
    for category in FAILURE_CATEGORIES:
        lines.append(f"| {category} | {summary.get(category, 0)} |")

    lines.extend([
        "",
        "## Audited Examples",
        "",
        "| example_id | domain | gold answer | Cell A generated | Cell C generated | C contains gold | C over-generates | wrong bridge | wrong final answer | scaffold ok | likely failure category |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ])

    for row in audit_rows:
        def esc(value: str) -> str:
            return str(value).replace("\n", "<br>").replace("|", "\\|")

        lines.append(
            "| {example_id} | {domain} | {gold_answer} | {cell_a} | {cell_c} | "
            "{contains_gold} | {overgen} | {wrong_bridge} | {wrong_final} | "
            "{scaffold_ok} | {category} |".format(
                example_id=esc(row["example_id"]),
                domain=esc(row["domain"]),
                gold_answer=esc(row["gold_answer"]),
                cell_a=esc(row["cell_a_generated"]),
                cell_c=esc(row["cell_c_generated"]),
                contains_gold=markdown_bool(row["cell_c_contains_gold"]),
                overgen=markdown_bool(row["cell_c_over_generates_sentence"]),
                wrong_bridge=markdown_bool(row["cell_c_selects_wrong_bridge_entity"]),
                wrong_final=markdown_bool(row["cell_c_selects_wrong_final_answer"]),
                scaffold_ok=markdown_bool(row["cell_c_follows_structured_scaffold"]),
                category=esc(row["likely_failure_category"]),
            )
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit Qwen examples where direct-clean succeeds but structured-clean fails."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-3B",
        help="HuggingFace model name used to resolve default model-specific paths.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Explicit dataset path override.",
    )
    parser.add_argument(
        "--evaluation",
        type=Path,
        default=None,
        help="Explicit evaluation_results.csv path override.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Explicit markdown output path override.",
    )
    args = parser.parse_args()

    slug = _model_slug(args.model)
    dataset_path = args.dataset or Path(f"dataset/processed/{slug}/dataset.json")
    evaluation_path = args.evaluation or Path(f"results/phase_2_behaviour/{slug}/evaluation_results.csv")
    output_path = args.output or Path(f"results/phase_2_behaviour/{slug}/qwen_clean_degradation_audit.md")

    dataset = load_dataset(dataset_path)
    eval_rows = load_eval_rows(evaluation_path)

    audit_rows = []
    summary = Counter()
    for eid, rows in sorted(eval_rows.items()):
        row_a = rows.get("A")
        row_c = rows.get("C")
        if not row_a or not row_c:
            continue
        if not (robust_bool(row_a["correct"]) and not robust_bool(row_c["correct"])):
            continue

        ex = dataset.get(eid, {})
        category, flags = classify(
            cell_c_raw=row_c.get("generated_answer_raw", ""),
            cell_c_norm=row_c.get("generated_answer_normalised", ""),
            gold=row_c.get("gold_answer", ex.get("answer", "")),
            bridge=ex.get("bridge_entity", ""),
            fact_2=ex.get("fact_2", ""),
        )
        summary[category] += 1
        audit_rows.append({
            "example_id": eid,
            "domain": row_c.get("domain", ex.get("domain", "")),
            "gold_answer": row_c.get("gold_answer", ex.get("answer", "")),
            "cell_a_generated": first_nonempty_line(row_a.get("generated_answer_raw", "")),
            "cell_c_generated": first_nonempty_line(row_c.get("generated_answer_raw", "")),
            **flags,
            "likely_failure_category": category,
        })

    write_report(audit_rows, summary, output_path)
    print(f"[audit] Wrote {len(audit_rows)} examples -> {output_path}")
    for category in FAILURE_CATEGORIES:
        print(f"[audit] {category}: {summary.get(category, 0)}")


if __name__ == "__main__":
    main()
