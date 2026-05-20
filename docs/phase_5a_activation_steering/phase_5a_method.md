# Phase 5a Activation Steering Method

Phase 5a is the first iteration activation-steering experiment. It tests whether the late-layer signal identified by activation patching can be reused as an intervention on held-out examples.

The scientific method is unchanged from the original Phase 5 steering script:

```text
steering_vector = mean(donor_activation - source_activation)
```

The vector is computed from train examples and injected into held-out source-condition runs. Injection remains final-prompt-token only.

## Primary Experiment

Qwen B->D noisy recovery steering is the primary Phase 5a experiment:

```text
model: Qwen/Qwen2.5-3B
source cell: B
donor cell: D
contrast file: noisy_contrast_examples.json
layer: 34
hook: resid_post
token position: final prompt token only
output prefix: noisy_
```

This tests whether the structured-noisy donor signal can partially recover the gold-answer signal in direct-noisy Qwen failures.

## Secondary Experiment

Pythia A->C clean structure steering remains the secondary experiment:

```text
model: EleutherAI/pythia-2.8b
source cell: A
donor cell: C
contrast file: contrast_examples.json
layer: 31
hook: resid_post
token position: final prompt token only
output prefix: base filenames
```

## Metrics

Phase 5a is score-only by default:

```text
baseline_gold_logit
steered_gold_logit
delta_gold_logit
baseline_gold_rank
steered_gold_rank
delta_gold_rank
baseline_top1
steered_top1
```

Exact-match remains the primary behavioural metric from Phase 2. Answer-containment remains a secondary diagnostic only when optional generation is requested.

## Controls

`--control none` runs the learned average steering vector.

`--control random` samples Gaussian random vectors and rescales them to the learned vector L2 norm.

`--control early_layer` computes the same donor-minus-source procedure at an early layer, default layer 8.

`--control filler` is documented as not implemented.

## Output Roots

Phase 5a outputs are written under:

```text
results/phase_5a_activation_steering/<model_slug>/
figures/phase_5a_activation_steering/<model_slug>/
```

For Qwen B->D learned steering:

```text
results/phase_5a_activation_steering/qwen2.5-3b/noisy_steering_summary.csv
figures/phase_5a_activation_steering/qwen2.5-3b/noisy_steering_alpha_sweep.png
```

Controls use separate filename prefixes such as `noisy_random_steering_*` and `noisy_early_layer_steering_*`.

## Thesis-Safe Interpretation

If logits and ranks improve but top-1 or generation does not, describe the result as representation-level recovery. Do not claim that steering fully fixes Qwen or that the average vector is a complete reasoning mechanism.
