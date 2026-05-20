# Phase 5b Steering Diagnostics

Phase 5b diagnoses why Phase 5a average activation steering partially works or fails. It does not change the Phase 5a steering method.

## Diagnostics

Oracle steering upper bound:

```powershell
python scripts/phase_5b_steering_diagnostics/steering_diagnostics.py --model Qwen/Qwen2.5-3B --source-cell B --donor-cell D --layer 34 --hook resid_post --diagnostic oracle --alphas 0.25 0.5 0.75 1.0 --seed 42
```

Late-layer sweep:

```powershell
python scripts/phase_5b_steering_diagnostics/steering_diagnostics.py --model Qwen/Qwen2.5-3B --source-cell B --donor-cell D --layers 31 32 33 34 35 --hook resid_post --diagnostic layer_sweep --alphas 0.25 0.5 0.75 1.0 --train-frac 0.7 --seed 42
```

Helped vs hurt analysis:

```powershell
python scripts/phase_5b_steering_diagnostics/steering_diagnostics.py --model Qwen/Qwen2.5-3B --source-cell B --donor-cell D --diagnostic helped_hurt --steering-results results/phase_5a_activation_steering/qwen2.5-3b/noisy_steering_results.csv
```

Aggregate report only:

```powershell
python scripts/phase_5b_steering_diagnostics/steering_diagnostics.py --model Qwen/Qwen2.5-3B --source-cell B --donor-cell D --diagnostic report
```

## Output Paths

Phase 5b outputs are written under:

```text
results/phase_5b_steering_diagnostics/<model_slug>/
figures/phase_5b_steering_diagnostics/<model_slug>/
```

Qwen outputs:

```text
results/phase_5b_steering_diagnostics/qwen2.5-3b/noisy_oracle_steering_results.csv
results/phase_5b_steering_diagnostics/qwen2.5-3b/noisy_oracle_steering_summary.csv
figures/phase_5b_steering_diagnostics/qwen2.5-3b/noisy_oracle_steering_alpha_sweep.png

results/phase_5b_steering_diagnostics/qwen2.5-3b/noisy_layer_sweep_steering_results.csv
results/phase_5b_steering_diagnostics/qwen2.5-3b/noisy_layer_sweep_steering_summary.csv
figures/phase_5b_steering_diagnostics/qwen2.5-3b/noisy_layer_sweep_steering_heatmap.png

results/phase_5b_steering_diagnostics/qwen2.5-3b/noisy_helped_hurt_analysis.csv
results/phase_5b_steering_diagnostics/qwen2.5-3b/noisy_helped_hurt_report.md
results/phase_5b_steering_diagnostics/qwen2.5-3b/noisy_steering_diagnostics_report.md
```

## Interpretation

If oracle steering greatly outperforms average steering, the effect is partly example-specific and the average vector is too blunt.

If the late-layer sweep peaks near layer 34, that supports the late-layer mediation story. If another late layer performs better, the late-layer story remains plausible but the intervention peak is shifted.

If logit/rank improves without top-1 recovery, use thesis-safe wording: representation-level recovery, not behavioural repair.
