# Phase 5c Steering Analysis

Helped/hurt analysis is currently implemented in:

```text
scripts/phase_5a_steering_calibration/steering_diagnostics.py --diagnostic helped_hurt
```

The runner routes Phase 5c outputs to:

```text
results/phase_5c_steering_analysis/<model_slug>/
```

Generated filenames are also prefixed with the model slug, for example:

```text
results/phase_5c_steering_analysis/qwen2.5-3b/qwen2.5-3b_noisy_helped_hurt_analysis.csv
results/phase_5c_steering_analysis/qwen2.5-3b/qwen2.5-3b_noisy_helped_hurt_report.md
results/phase_5c_steering_analysis/qwen2.5-3b/qwen2.5-3b_noisy_final_steering_interpretation.md
```

This keeps the analysis stage explicit without duplicating the shared steering diagnostics helpers.
