# Phase 5b Activation Steering Usage

Run Phase 5a calibration first:

```powershell
python scripts/run_model_pipeline.py --preset qwen-steering-calibration
```

Then run final average steering from the recommended config:

```powershell
python scripts/run_model_pipeline.py --preset qwen-steering-final
```

Run controls with the same calibrated layer and alpha range:

```powershell
python scripts/run_model_pipeline.py --preset qwen-steering-controls
```

Regenerate all Qwen Phase 5 outputs in the calibrated order:

```powershell
python scripts/run_model_pipeline.py --preset qwen-steering-full --clean-phase5 --yes
```

## Direct Script Command

If you want to run the script directly, read the layer and alpha range from `noisy_recommended_steering_config.json`, then call:

```powershell
python scripts/phase_5b_activation_steering/activation_steering.py --model Qwen/Qwen2.5-3B --source-cell B --donor-cell D --layer <recommended_layer> --hook resid_post --alphas 0.0 0.25 0.5 0.75 1.0 --train-frac 0.7 --seed 42
```

Random control:

```powershell
python scripts/phase_5b_activation_steering/activation_steering.py --model Qwen/Qwen2.5-3B --source-cell B --donor-cell D --layer <recommended_layer> --hook resid_post --alphas 0.0 0.25 0.5 0.75 1.0 --train-frac 0.7 --seed 42 --control random --random-seeds 3
```

Early-layer control:

```powershell
python scripts/phase_5b_activation_steering/activation_steering.py --model Qwen/Qwen2.5-3B --source-cell B --donor-cell D --layer <recommended_layer> --hook resid_post --alphas 0.0 0.25 0.5 0.75 1.0 --train-frac 0.7 --seed 42 --control early_layer --early-layer 8
```

## Outputs

```text
results/phase_5b_activation_steering/qwen2.5-3b/qwen2.5-3b_noisy_steering_results.csv
results/phase_5b_activation_steering/qwen2.5-3b/qwen2.5-3b_noisy_steering_summary.csv
results/phase_5b_activation_steering/qwen2.5-3b/qwen2.5-3b_noisy_steering_alpha_sweep.csv
results/phase_5b_activation_steering/qwen2.5-3b/qwen2.5-3b_noisy_steering_vector_stats.json
results/phase_5b_activation_steering/qwen2.5-3b/qwen2.5-3b_noisy_steering_report.md
figures/phase_5b_activation_steering/qwen2.5-3b/qwen2.5-3b_noisy_steering_alpha_sweep.png
figures/phase_5b_activation_steering/qwen2.5-3b/qwen2.5-3b_noisy_steering_logit_shift.png
figures/phase_5b_activation_steering/qwen2.5-3b/qwen2.5-3b_noisy_steering_rank_shift.png
```

Control outputs use separate prefixes:

```text
results/phase_5b_activation_steering/qwen2.5-3b/qwen2.5-3b_noisy_random_steering_*
results/phase_5b_activation_steering/qwen2.5-3b/qwen2.5-3b_noisy_early_layer_steering_*
```
