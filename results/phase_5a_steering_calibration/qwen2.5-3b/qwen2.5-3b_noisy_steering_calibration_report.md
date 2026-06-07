# Phase 5a Steering Calibration Report

Model: `Qwen/Qwen2.5-3B`
Contrast: Cell B -> Cell D
Contrast file: `dataset\processed\qwen2.5-3b\noisy_contrast_examples.json`

Phase 5a runs calibration before the final average-steering intervention. Oracle steering is an upper bound on final-token intervention strength, and the layer sweep selects the average-steering layer and alpha range for Phase 5b.

## Questions

### Did oracle steering outperform average steering?

Oracle and/or Phase 5b learned steering summaries are not both available yet. This is expected before the final steering run.

### Does average steering peak at layer 34 or another late layer?

The current layer-sweep best by mean delta gold logit is layer `32`, alpha `1.0`, with mean delta `1.6938`.

### Recommended Phase 5b configuration

Recommended layer `32`, hook `resid_post`, with final alpha range `0.0 0.25 0.5 0.75 1.0`. Selection metric: highest mean_delta_gold_logit from layer_sweep with alpha <= 1.0; ties prefer lower alpha.
This recommendation is for average steering, not oracle per-example steering.

### Which alpha range is safest?

The safest alpha range should be judged from positive mean delta gold logit without top-1 degradation. Current best-logit alphas observed in available summaries: `1.0`.

### Are helped examples different from hurt examples?

Helped/hurt analysis has not been generated yet. It belongs to Phase 5c and should run after Phase 5b final steering.

### Does the evidence support representation-level recovery?

Learned-steering summary is not available.

### Does the evidence support behavioural/top-1 recovery?

Some available diagnostic rows show top-1 improvement, with best observed improvement `0.8125`. This should still be framed carefully unless generation also improves.

## Thesis-Safe Claims

- If logit/rank improves but top-1/generation does not, describe the result as representation-level recovery.
- If oracle steering is much stronger than average steering, describe the effect as partly example-specific and say the average vector is too blunt.
- If late layers outperform early layers or the selected late layer improves logit/rank recovery, this supports the late-layer mediation story from activation patching.

## What Not To Claim

- Do not claim steering fully fixes Qwen unless top-1 and generation clearly improve.
- Do not claim the steering vector is the complete reasoning circuit.
- Do not claim one average vector should work across all examples or models.
