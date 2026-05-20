# Model Pipeline Runner

`scripts/run_model_pipeline.py` is an orchestration helper. It is not a new experiment phase. It calls the existing runnable scripts, streams their output, writes logs, and checks expected outputs.

The runner does not rewrite or duplicate experiment logic. Dataset construction, behavioural evaluation, containment auditing, patching, logit lens analysis, attention visualisation, and overlay plotting remain in their existing scripts.

## Interactive Usage

Run without arguments:

```powershell
python scripts/run_model_pipeline.py
```

The menu offers:

```text
pythia-clean
qwen-noisy-recovery
qwen-clean-degradation
qwen-direct-noise
qwen-structured-noise
qwen-full-spread
qwen-steering-5a
qwen-steering-5a-controls
qwen-steering-5b
qwen-steering-full
```

For normal Qwen work, use `qwen-full-spread` when you want Phase 2 to run once and then reuse the same Phase 3/4 scripts across the Qwen contrasts.

## Presets

### `pythia-clean`

```text
model: EleutherAI/pythia-2.8b
source cell: A
donor cell: C
contrast file: contrast_examples.json
output prefix: base filenames
component layers: 24 25 29 30 31
attention layers: 20 30 31
```

### `qwen-noisy-recovery`

Alias-compatible with the older `qwen-noisy` preset.

```text
model: Qwen/Qwen2.5-3B
source cell: B
donor cell: D
contrast file: noisy_contrast_examples.json
output prefix: noisy_
component layers: 31 32 33 34 35
attention layers: 20 31 33 34 35
```

### `qwen-clean-degradation`

```text
source cell: C
donor cell: A
contrast file: clean_degradation_contrast_examples.json
output prefix: clean_degradation_
purpose: clean degradation / direct recovery
```

### `qwen-direct-noise`

```text
source cell: B
donor cell: A
contrast file: direct_noise_contrast_examples.json
output prefix: direct_noise_
purpose: direct noise damage
```

### `qwen-structured-noise`

```text
source cell: C
donor cell: D
contrast file: structured_noise_contrast_examples.json
output prefix: structured_noise_
purpose: structured noise stability
```

### `qwen-full-spread`

`qwen-full-spread` runs the Qwen setup once:

```text
dataset
evaluation with --run-containment-audit
```

Then it reuses the same Phase 3/4 scripts for selected contrasts:

```text
layer-patching
component-patching
logit-lens
attention
```

By default, expensive Phase 3/4 work runs only for contrasts with at least 20 examples. Low-n contrasts are still saved by Phase 2 and reported clearly.

## Contrast Definitions

Phase 2 writes all useful contrast files from the same `evaluation_results.csv`:

```text
A->C  A wrong and C correct  contrast_examples.json
B->D  B wrong and D correct  noisy_contrast_examples.json
B->A  B wrong and A correct  direct_noise_contrast_examples.json
C->D  C wrong and D correct  structured_noise_contrast_examples.json
C->A  C wrong and A correct  clean_degradation_contrast_examples.json
```

Output prefixes:

```text
Pythia A->C default: base filenames
Qwen A->C in full spread: clean_
Qwen B->D: noisy_
Qwen B->A: direct_noise_
Qwen C->D: structured_noise_
Qwen C->A: clean_degradation_
```

The flat folder convention is preserved. Outputs stay under:

```text
results/<stage>/<model_slug>/
figures/<stage>/<model_slug>/
```

No nested contrast folders such as `noisy_bd/` are used.

## Low-N Handling

If a contrast has fewer than 20 examples:

```text
Phase 2 saves the contrast file.
The runner prints a low-n warning.
qwen-full-spread skips it by default for expensive Phase 3/4 work.
Interactive mode lets you run low-n contrasts anyway.
```

This keeps the data visible without silently spending GPU time on fragile estimates.

## Thesis Interpretation

Recommended framing:

```text
B->D: noisy recovery
C->A: clean degradation / direct recovery
B->A: direct noise damage
C->D: structured noise stability
```

These are not new phases. They are contrast selections run through the existing phases.

## Non-Interactive Usage

```powershell
python scripts/run_model_pipeline.py --preset pythia-clean
python scripts/run_model_pipeline.py --preset qwen-noisy-recovery
python scripts/run_model_pipeline.py --preset qwen-clean-degradation
python scripts/run_model_pipeline.py --preset qwen-direct-noise
python scripts/run_model_pipeline.py --preset qwen-structured-noise
python scripts/run_model_pipeline.py --preset qwen-full-spread
python scripts/run_model_pipeline.py --preset qwen-steering-5a
python scripts/run_model_pipeline.py --preset qwen-steering-5a-controls
python scripts/run_model_pipeline.py --preset qwen-steering-5b
python scripts/run_model_pipeline.py --preset qwen-steering-full
```

The older alias still works:

```powershell
python scripts/run_model_pipeline.py --preset qwen-noisy
```

Select stages with comma-separated names:

```powershell
python scripts/run_model_pipeline.py --preset qwen-full-spread --stages dataset,evaluation,containment
```

Available stages:

```text
dataset
evaluation
containment
layer-patching
component-patching
logit-lens
attention
overlay
steering-5a
steering-5a-controls
steering-5b-oracle
steering-5b-layer-sweep
steering-5b-helped-hurt
steering-5b-all
```

## Dry Runs

```powershell
python scripts/run_model_pipeline.py --preset qwen-full-spread --dry-run
python scripts/run_model_pipeline.py --preset qwen-clean-degradation --dry-run
python scripts/run_model_pipeline.py --preset qwen-direct-noise --dry-run
python scripts/run_model_pipeline.py --preset qwen-structured-noise --dry-run
python scripts/run_model_pipeline.py --preset pythia-clean --dry-run
python scripts/run_model_pipeline.py --preset qwen-steering-full --dry-run
```

Dry runs print commands and create logs, but do not run GPU work.

## Logs and Timers

Every run creates a log under:

```text
logs/pipeline_runs/
```

Each streamed child-script line is prefixed with the current location:

```text
[2/7 evaluation | scripts/phase_2_behaviour/run_evaluation.py | stage 00:03:12 | total 00:04:01] ...
```

This keeps the current stage, script, stage runtime, and overall runtime visible during long runs.

## Expected Outputs

Evaluation:

```text
results/phase_2_behaviour/<model_slug>/evaluation_results.csv
results/phase_2_behaviour/<model_slug>/accuracy_summary.csv
dataset/processed/<model_slug>/contrast_examples.json
dataset/processed/<model_slug>/noisy_contrast_examples.json
dataset/processed/<model_slug>/direct_noise_contrast_examples.json
dataset/processed/<model_slug>/structured_noise_contrast_examples.json
dataset/processed/<model_slug>/clean_degradation_contrast_examples.json
```

Examples for Qwen full-spread Phase 3a:

```text
results/phase_3a_layer_patching/qwen2.5-3b/clean_layer_patch_summary.csv
results/phase_3a_layer_patching/qwen2.5-3b/noisy_layer_patch_summary.csv
results/phase_3a_layer_patching/qwen2.5-3b/direct_noise_layer_patch_summary.csv
results/phase_3a_layer_patching/qwen2.5-3b/structured_noise_layer_patch_summary.csv
results/phase_3a_layer_patching/qwen2.5-3b/clean_degradation_layer_patch_summary.csv
```

The same prefix pattern applies to component patching, logit lens, attention, and figures.

## Phase 5a / 5b Steering

Phase 5 is split into:

```text
Phase 5a: activation steering first iteration
Phase 5b: steering diagnostics / analysis
```

Qwen B->D remains the primary Phase 5 target.

### Steering Presets

`qwen-steering-5a` runs learned Qwen B->D steering:

```powershell
python scripts/run_model_pipeline.py --preset qwen-steering-5a
```

`qwen-steering-5a-controls` runs the random matched-norm and early-layer controls:

```powershell
python scripts/run_model_pipeline.py --preset qwen-steering-5a-controls
```

`qwen-steering-5b` runs oracle steering, late-layer sweep, and helped/hurt diagnostics:

```powershell
python scripts/run_model_pipeline.py --preset qwen-steering-5b
```

`qwen-steering-full` runs Phase 5a learned steering, Phase 5a controls, then Phase 5b diagnostics:

```powershell
python scripts/run_model_pipeline.py --preset qwen-steering-full
```

To regenerate clean Qwen Phase 5 outputs from existing Phase 1-4 prerequisites:

```powershell
python scripts/run_model_pipeline.py --preset qwen-steering-full --clean-phase5 --yes
```

Phase 5 can skip earlier phases only when required Phase 1/2 outputs and contrast files exist. Phase 5b requires Phase 5a learned steering outputs unless the same pipeline run includes `steering-5a` before the diagnostics.

### Prerequisites

Phase 5a requires:

```text
dataset/processed/qwen2.5-3b/dataset.json
dataset/processed/qwen2.5-3b/noisy_contrast_examples.json
results/phase_2_behaviour/qwen2.5-3b/evaluation_results.csv
```

Phase 5a recommends:

```text
results/phase_3a_layer_patching/qwen2.5-3b/noisy_layer_patch_summary.csv
```

If the recommended Phase 3a file is missing, the runner warns that layer 34 is based on prior known Qwen B->D results.

Phase 5b requires:

```text
dataset/processed/qwen2.5-3b/dataset.json
dataset/processed/qwen2.5-3b/noisy_contrast_examples.json
results/phase_5a_activation_steering/qwen2.5-3b/noisy_steering_results.csv
```

If `noisy_steering_results.csv` is missing for a Phase 5b-only run, the runner tells you to run `qwen-steering-5a` first.

### Phase 5 Outputs

Phase 5a learned steering:

```text
results/phase_5a_activation_steering/qwen2.5-3b/noisy_steering_results.csv
results/phase_5a_activation_steering/qwen2.5-3b/noisy_steering_summary.csv
results/phase_5a_activation_steering/qwen2.5-3b/noisy_steering_alpha_sweep.csv
results/phase_5a_activation_steering/qwen2.5-3b/noisy_steering_report.md
```

Phase 5a controls:

```text
results/phase_5a_activation_steering/qwen2.5-3b/noisy_random_steering_summary.csv
results/phase_5a_activation_steering/qwen2.5-3b/noisy_early_layer_steering_summary.csv
```

Phase 5b diagnostics:

```text
results/phase_5b_steering_diagnostics/qwen2.5-3b/noisy_oracle_steering_summary.csv
results/phase_5b_steering_diagnostics/qwen2.5-3b/noisy_layer_sweep_steering_summary.csv
results/phase_5b_steering_diagnostics/qwen2.5-3b/noisy_helped_hurt_report.md
results/phase_5b_steering_diagnostics/qwen2.5-3b/noisy_steering_diagnostics_report.md
```

### Clean Phase 5

Interactive Phase 5 presets detect existing Phase 5a/5b outputs and offer:

```text
[1] Resume / skip existing
[2] Delete Phase 5a/5b outputs and rerun
[3] Choose manually
[q] quit
```

Deletion is limited to:

```text
results/phase_5a_activation_steering/qwen2.5-3b/
figures/phase_5a_activation_steering/qwen2.5-3b/
results/phase_5b_steering_diagnostics/qwen2.5-3b/
figures/phase_5b_steering_diagnostics/qwen2.5-3b/
```

Interactive deletion requires typing:

```text
DELETE PHASE 5 QWEN
```

## Overlay

After both model pipelines are complete:

```powershell
python scripts/run_model_pipeline.py --preset qwen-noisy-recovery --stages overlay
```

or append overlay to a run:

```powershell
python scripts/run_model_pipeline.py --preset qwen-full-spread --run-overlay
```

## Clean Rerun

Interactive clean rerun deletes only generated outputs for the selected model slug, and only after typing:

```text
DELETE
```

Overlay cleanup is asked separately.

The runner never deletes source code, scripts, docs, configs, README files, thesis text, `.gitignore`, git history, or outputs for another active model.
