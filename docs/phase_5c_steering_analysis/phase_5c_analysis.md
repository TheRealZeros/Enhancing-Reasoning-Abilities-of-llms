# Phase 5c Steering Analysis

Phase 5c runs after Phase 5b final average steering. It does not change the steering method. It explains which held-out examples benefited from the calibrated average vector and which were harmed.

## Command

```powershell
python scripts/run_model_pipeline.py --preset qwen-steering-analysis
```

Direct script command:

```powershell
python scripts/phase_5a_steering_calibration/steering_diagnostics.py --model Qwen/Qwen2.5-3B --source-cell B --donor-cell D --diagnostic helped_hurt --steering-results results/phase_5b_activation_steering/qwen2.5-3b/qwen2.5-3b_noisy_steering_results.csv
```

## Prerequisite

```text
results/phase_5b_activation_steering/qwen2.5-3b/qwen2.5-3b_noisy_steering_results.csv
```

If this file is missing, run:

```powershell
python scripts/run_model_pipeline.py --preset qwen-steering-final
```

## Outputs

```text
results/phase_5c_steering_analysis/qwen2.5-3b/qwen2.5-3b_noisy_helped_hurt_analysis.csv
results/phase_5c_steering_analysis/qwen2.5-3b/qwen2.5-3b_noisy_helped_hurt_report.md
results/phase_5c_steering_analysis/qwen2.5-3b/qwen2.5-3b_noisy_final_steering_interpretation.md
```

## Interpretation

Use helped/hurt counts, baseline ranks, baseline logits, and domain breakdowns to explain mixed steering results. If many rows improve in logit or rank while top-1 remains weak, frame the result as representation-level recovery rather than behavioural recovery.
