# Phase 5b Activation Steering Method

Phase 5b is the final average activation-steering intervention. It runs after Phase 5a calibration and uses the calibrated layer, hook, and alpha range written to:

```text
results/phase_5a_steering_calibration/qwen2.5-3b/noisy_recommended_steering_config.json
```

The scientific method is unchanged from the first steering implementation:

```text
steering_vector = mean(donor_activation - source_activation)
```

The vector is computed on training contrast examples and injected into held-out source-cell runs. Injection is final-token-only at `blocks.<layer>.hook_resid_post`.

## Primary Qwen Experiment

```text
model: Qwen/Qwen2.5-3B
source cell: B
donor cell: D
contrast file: noisy_contrast_examples.json
hook: resid_post
token position: final prompt token only
layer: read from Phase 5a recommended config
alphas: read from Phase 5a recommended config
```

## Metrics

The primary metrics are score-only:

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

Exact-match remains primary for generation only when `--generate-examples` is explicitly used. Answer-containment remains a secondary diagnostic. Phase 5b does not change Phase 1 dataset logic, Phase 2 behavioural scoring, Phase 3 patching, or Phase 4 logit lens/attention methods.

## Controls

Random matched-norm controls sample Gaussian directions and rescale them to the learned vector norm. Early-layer controls compute the same average-vector procedure at the configured early layer, usually layer 8. Controls write separate files and must not overwrite learned steering outputs.

## Interpretation

If Phase 5b improves gold-answer logit or rank without strong top-1/generation recovery, describe the result as representation-level recovery. Do not claim that average steering fully fixes Qwen unless behavioural or generation metrics clearly improve.
