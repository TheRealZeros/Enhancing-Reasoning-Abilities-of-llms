# Phase 5a Steering Calibration

Phase 5a calibrates Qwen B->D steering before the final average-steering run. The goal is to test whether final-token steering is viable and to select a late layer and alpha range for Phase 5b without hand-picking from the final intervention.

## Research Logic

Phase 3 showed that late-layer activation patching can partially recover the target answer signal. Phase 5 asks whether that signal can be reused as an intervention. The calibration-first workflow is:

1. Run oracle steering as an upper bound.
2. Run a late-layer average-steering sweep.
3. Write a recommended average-steering config.
4. Use that config for Phase 5b final steering.

Oracle steering is not deployable steering because it uses each held-out example's own donor-source vector. It tells us whether the final-token intervention setup has enough causal leverage at all.

The layer sweep tests average vectors across layers 31, 32, 33, 34, and 35. The recommended config selects the row with highest `mean_delta_gold_logit`, restricted to `alpha <= 1.0`; ties prefer the lower alpha. Top-1 improvement is recorded but is not the sole selection criterion.

## Commands

Oracle calibration:

```powershell
python scripts/phase_5a_steering_calibration/steering_diagnostics.py --model Qwen/Qwen2.5-3B --source-cell B --donor-cell D --layer 34 --hook resid_post --diagnostic oracle --alphas 0.25 0.5 0.75 1.0 --seed 42
```

Late-layer sweep:

```powershell
python scripts/phase_5a_steering_calibration/steering_diagnostics.py --model Qwen/Qwen2.5-3B --source-cell B --donor-cell D --layers 31 32 33 34 35 --hook resid_post --diagnostic layer_sweep --alphas 0.25 0.5 0.75 1.0 --train-frac 0.7 --seed 42
```

Pipeline preset:

```powershell
python scripts/run_model_pipeline.py --preset qwen-steering-calibration
```

## Outputs

```text
results/phase_5a_steering_calibration/qwen2.5-3b/qwen2.5-3b_noisy_oracle_steering_results.csv
results/phase_5a_steering_calibration/qwen2.5-3b/qwen2.5-3b_noisy_oracle_steering_summary.csv
results/phase_5a_steering_calibration/qwen2.5-3b/qwen2.5-3b_noisy_layer_sweep_steering_results.csv
results/phase_5a_steering_calibration/qwen2.5-3b/qwen2.5-3b_noisy_layer_sweep_steering_summary.csv
results/phase_5a_steering_calibration/qwen2.5-3b/qwen2.5-3b_noisy_recommended_steering_config.json
results/phase_5a_steering_calibration/qwen2.5-3b/qwen2.5-3b_noisy_steering_calibration_report.md
figures/phase_5a_steering_calibration/qwen2.5-3b/qwen2.5-3b_noisy_oracle_steering_alpha_sweep.png
figures/phase_5a_steering_calibration/qwen2.5-3b/qwen2.5-3b_noisy_layer_sweep_steering_heatmap.png
```

## Thesis-Safe Wording

Use Phase 5a to say that the intervention site and final-token setup were calibrated before the final average-steering run. If oracle steering is much stronger than average steering, describe the effect as partly example-specific. If late layers outperform weaker layers or controls, describe this as support for late-layer mediation rather than proof of a complete reasoning circuit.
