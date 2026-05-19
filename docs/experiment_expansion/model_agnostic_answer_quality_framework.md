# Model-Agnostic Answer Quality Framework

This document describes the answer-containment audit used as a secondary behavioural diagnostic. It is not a new phase. It is an evaluation-layer post-processing step that reads existing Phase 2 behavioural outputs.

## Primary Metric

Exact-match accuracy remains the strict primary metric. The `correct` field in `evaluation_results.csv` is still exact-match only, and contrast examples for downstream causal analyses are still defined using exact-match correctness.

Answer-containment accuracy does not replace exact-match. It helps diagnose whether a generation that failed exact-match nevertheless contained the gold answer with extra wording, context, or specificity.

## Diagnostic Labels

Each generated answer is classified into exactly one label:

| Label | Meaning |
|---|---|
| `exact_match` | The generated answer exactly matches the gold answer after normalisation. |
| `contains_correct_answer` | Exact-match failed, but the normalised generation contains the normalised gold answer. |
| `wrong` | The normalised generation does not contain the normalised gold answer. |

The label `contains_correct_answer` is used deliberately. The word padding is avoided because padding already has a technical meaning in the dataset construction and token-alignment pipeline.

## Why This Matters

Structured prompts can encourage the model to produce explanatory continuations or longer noun phrases. These outputs may fail strict exact-match while still containing the correct answer.

For example:

| Gold | Generated |
|---|---|
| `Spain` | `southern Spain` |
| `480 million` | `480 million alveoli` |
| `5500 degrees Celsius` | `the Sun has a surface temperature of about 5500 degrees Celsius` |

This matters for Qwen2.5-3B because Structured/Clean can be undercounted by exact-match when the answer appears inside a longer structured completion. The audit lets the thesis distinguish formatting or over-generation from genuinely wrong answers.

## Fair Model Comparison

The audit should be run for both active models:

- `EleutherAI/pythia-2.8b`
- `Qwen/Qwen2.5-3B`

For fair reporting, compare models using both:

- strict exact-match accuracy
- containment-aware accuracy

Containment-aware accuracy is:

```text
(exact_match_count + contains_correct_answer_count) / total
```

Wrong rate is:

```text
wrong_count / total
```

## How To Run

Standalone:

```powershell
python scripts/analysis/answer_containment_audit.py --model Qwen/Qwen2.5-3B
python scripts/analysis/answer_containment_audit.py --model EleutherAI/pythia-2.8b
```

As optional Phase 2 post-processing:

```powershell
python scripts/phase_2_behaviour/run_evaluation.py --model Qwen/Qwen2.5-3B --run-containment-audit
python scripts/phase_2_behaviour/run_evaluation.py --model EleutherAI/pythia-2.8b --run-containment-audit
```

The Phase 2 flag does not change model inference, exact-match scoring, `evaluation_results.csv`, or contrast extraction. It only runs the secondary audit after the standard Phase 2 outputs have been written.

## Outputs

Audit outputs are written under:

```text
results/model_agnostic_evaluation/<model_slug>/
```

Files:

```text
answer_containment_audit.csv
answer_containment_summary.csv
answer_containment_audit.md
```

## Relationship To Runnable Stages

The runnable experiment stages remain:

- Phase 1 dataset construction
- Phase 2 behavioural evaluation
- Phase 3a layer-level activation patching
- Phase 3b component-level patching
- Phase 4a logit lens
- Phase 4b attention visualisation

Answer-containment is not a new phase. It is a reusable post-processing diagnostic that sits above Phase 2 outputs and helps interpret behavioural answer quality before downstream causal analysis.
