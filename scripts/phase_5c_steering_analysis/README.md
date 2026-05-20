# Phase 5c Steering Analysis

Helped/hurt analysis is currently implemented in:

```text
scripts/phase_5a_steering_calibration/steering_diagnostics.py --diagnostic helped_hurt
```

The runner routes Phase 5c outputs to:

```text
results/phase_5c_steering_analysis/<model_slug>/
```

This keeps the analysis stage explicit without duplicating the shared steering diagnostics helpers.
