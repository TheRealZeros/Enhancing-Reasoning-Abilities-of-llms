# Phase 5b Activation Steering Report

## Experiment Configuration

- Model: `Qwen/Qwen2.5-3B`
- Source cell: `B`
- Donor cell: `D`
- Contrast file: `dataset\processed\qwen2.5-3b\noisy_contrast_examples.json`
- Requested layer: `32`
- Applied layer: `32`
- Hook: `resid_post`
- Token position: `final_prompt_token`
- Control: `none`
- Train fraction: `0.7`
- Seed: `42`
- Train examples used for split: `72`
- Held-out test examples: `32`
- Steering vector L2 norm: `94.508430`

## Final-Token-Only Injection Confirmation

Steering was injected only into the final prompt token position of the selected activation tensor. For score-only runs, the source prompt is the whole input and the injection index is `-1`. For optional generation, the injection stays fixed at the original prompt-final index rather than moving onto generated tokens.

## Score-Only Status

This is a score-only run. No generation was performed because `--generate-examples` was not provided.

## Alpha Sweep Summary

| control | alpha | mean delta gold logit | median delta gold logit | mean delta gold rank | baseline top1 | steered top1 | top1 improvement |
|---|---:|---:|---:|---:|---:|---:|---:|
| none | 0.0 | 0.0000 | 0.0000 | 0.00 | 0.094 | 0.094 | 0.000 |
| none | 0.25 | 0.7178 | 0.6172 | 5.44 | 0.094 | 0.156 | 0.062 |
| none | 0.5 | 1.2952 | 1.1875 | 6.75 | 0.094 | 0.125 | 0.031 |
| none | 0.75 | 1.6357 | 1.5234 | 6.81 | 0.094 | 0.125 | 0.031 |
| none | 1.0 | 1.6938 | 1.4453 | 6.69 | 0.094 | 0.125 | 0.031 |

## Best Alpha

- `none` best by mean delta gold logit: alpha `1.0` (mean delta `1.6938`).
- `none` best by top-1 improvement: alpha `0.25` (improvement `0.062`).

## Alpha 0.0 Sanity Check

Rows at alpha `0.0`: `32`. Maximum absolute logit delta: `0.00000000`. Maximum absolute rank delta: `0`. Top-1 mismatches: `0`.
The alpha `0.0` sanity check passed: steered and baseline scores were identical or numerically negligible.

## Control Notes

The run used the learned donor-minus-source steering vector.

## Interpretation Guidance

A positive mean delta gold logit indicates that steering increased the gold answer's first-token logit on held-out source prompts. A positive delta gold rank means the gold token moved upward in the vocabulary ranking because rank improvement is recorded as baseline rank minus steered rank.

Strong evidence requires the learned late-layer vector to improve gold logit, rank, and top-1 rate, while outperforming random matched-norm and early-layer controls.

## Limitations

- Activation patching is example-specific, while this steering vector is an average direction.
- Logit and rank recovery may not translate into full generated-answer recovery.
- The first implementation supports `resid_post` only.
- The filler control is not implemented in this version.

## Thesis-Safe Language

Activation steering was introduced as an intervention experiment based on the late-layer localisation found through activation patching. The steering vector was computed as the average donor-minus-source activation difference over training examples and evaluated on held-out examples. Steering was applied only at the final prompt token, matching the final-token scoring methodology used in activation patching. The primary outcome is gold-answer logit, rank, and top-1 recovery.

Avoid claiming that the vector is a reasoning circuit or that steering fully fixes the model. A negative or mixed result remains informative because it tests whether example-specific patching effects transfer into a reusable average direction.
