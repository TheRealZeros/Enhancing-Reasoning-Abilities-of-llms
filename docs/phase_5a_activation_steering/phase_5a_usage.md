# Phase 5a Activation Steering Usage

Phase 5a can be run directly or through `scripts/run_model_pipeline.py`.

## Direct Qwen Learned Steering

```powershell
python scripts/phase_5a_activation_steering/activation_steering.py --model Qwen/Qwen2.5-3B --source-cell B --donor-cell D --layer 34 --hook resid_post --alphas 0.0 0.5 1.0 2.0 --train-frac 0.7 --seed 42
```

Alpha `0.0` is a sanity baseline. Steered scores should match baseline scores.

## Direct Qwen Controls

Random matched-norm control:

```powershell
python scripts/phase_5a_activation_steering/activation_steering.py --model Qwen/Qwen2.5-3B --source-cell B --donor-cell D --layer 34 --hook resid_post --alphas 0.0 0.5 1.0 2.0 --train-frac 0.7 --seed 42 --control random --random-seeds 3
```

Early-layer control:

```powershell
python scripts/phase_5a_activation_steering/activation_steering.py --model Qwen/Qwen2.5-3B --source-cell B --donor-cell D --layer 34 --hook resid_post --alphas 0.0 0.5 1.0 2.0 --train-frac 0.7 --seed 42 --control early_layer --early-layer 8
```

## Pipeline Presets

Learned steering only:

```powershell
python scripts/run_model_pipeline.py --preset qwen-steering-5a
```

Controls:

```powershell
python scripts/run_model_pipeline.py --preset qwen-steering-5a-controls
```

Full Phase 5a/5b Qwen regeneration:

```powershell
python scripts/run_model_pipeline.py --preset qwen-steering-full --clean-phase5 --yes
```

## Output Paths

Qwen learned steering:

```text
results/phase_5a_activation_steering/qwen2.5-3b/noisy_steering_results.csv
results/phase_5a_activation_steering/qwen2.5-3b/noisy_steering_summary.csv
results/phase_5a_activation_steering/qwen2.5-3b/noisy_steering_alpha_sweep.csv
results/phase_5a_activation_steering/qwen2.5-3b/noisy_steering_vector_stats.json
results/phase_5a_activation_steering/qwen2.5-3b/noisy_steering_report.md
```

Qwen figures:

```text
figures/phase_5a_activation_steering/qwen2.5-3b/noisy_steering_alpha_sweep.png
figures/phase_5a_activation_steering/qwen2.5-3b/noisy_steering_logit_shift.png
figures/phase_5a_activation_steering/qwen2.5-3b/noisy_steering_rank_shift.png
```

Controls do not overwrite learned outputs:

```text
results/phase_5a_activation_steering/qwen2.5-3b/noisy_random_steering_*
results/phase_5a_activation_steering/qwen2.5-3b/noisy_early_layer_steering_*
```

Pythia A->C secondary outputs use base filenames under:

```text
results/phase_5a_activation_steering/pythia-2.8b/
figures/phase_5a_activation_steering/pythia-2.8b/
```
