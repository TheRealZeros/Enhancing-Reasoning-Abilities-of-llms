# Phase 5c Final Steering Interpretation

Source results: `results\phase_5b_activation_steering\qwen2.5-3b\noisy_steering_results.csv`

This report interprets the final average-steering run after calibration. It is post-steering analysis, not a new intervention.

## Overall Helped/Hurt Balance

- Rows analysed: `160`
- Helped rows: `122`
- Hurt rows: `6`
- Unchanged rows: `32`

## Alpha Safety

| alpha | helped | hurt | unchanged | mean delta gold logit | mean delta gold rank |
|---:|---:|---:|---:|---:|---:|
| 0.0 | 0 | 0 | 32 | 0.0000 | 0.0000 |
| 0.25 | 31 | 1 | 0 | 0.7178 | 5.4375 |
| 0.5 | 31 | 1 | 0 | 1.2952 | 6.7500 |
| 0.75 | 31 | 1 | 0 | 1.6357 | 6.8125 |
| 1.0 | 29 | 3 | 0 | 1.6938 | 6.6875 |

## Thesis-Safe Interpretation

- If average steering improves gold-answer logits or ranks but does not improve top-1/generation, describe it as representation-level recovery.
- Do not claim that average steering fully fixes Qwen unless top-1 and generation clearly improve.
- If helped and hurt examples are both common, describe the steering direction as partially useful and example-sensitive.
- The calibrated layer and alpha range should be presented as selected from held-out score diagnostics, not hand-picked from the final steering outcome.
