# Model Pipeline Runner

`scripts/run_model_pipeline.py` is an orchestration helper. It is not a new experiment phase. It calls the existing runnable scripts, streams their output, writes logs, and checks expected outputs.

The runner does not rewrite or duplicate experiment logic. Dataset construction, behavioural evaluation, containment auditing, patching, logit lens analysis, attention visualisation, and overlay plotting remain in their existing scripts.

## Interactive Usage

Run without arguments:

```powershell
python scripts/run_model_pipeline.py
```

The first menu is model-first:

```text
Select model / workflow:

[1] Pythia-2.8B
[2] Qwen2.5-3B
[q] quit
```

Pythia then offers full resume, clean full rerun, Phase 1-4 core, Phase 1-2 behaviour, Phase 3-4 mechanistic, Phase 5 steering, and status-only choices:

```text
[a] Full end-to-end run, Phase 1 -> Phase 5, resume/skip existing
[b] Clean full end-to-end rerun, Phase 1 -> Phase 5, delete Pythia generated outputs first
[c] Core pipeline only, Phase 1 -> Phase 4
[d] Behaviour only, Phase 1 -> Phase 2
[e] Mechanistic only, Phase 3 -> Phase 4
[f] Steering only, Phase 5
[g] Status / dry run
[q] back
```

Qwen exposes the noisy-recovery core, full-spread Phase 2-4 contrasts, Phase 5 steering slices, and status-only choice:

```text
[a] Full end-to-end run, Phase 1 -> Phase 5, resume/skip existing
[b] Clean full end-to-end rerun, Phase 1 -> Phase 5, delete Qwen generated outputs first
[c] Core noisy-recovery pipeline only, Phase 1 -> Phase 4
[d] Full-spread behavioural/mechanistic contrasts
[e] Steering calibration only, Phase 5a
[f] Final steering only, Phase 5b
[g] Steering controls only
[h] Steering analysis only, Phase 5c
[i] Full steering only, Phase 5a -> 5c
[j] Status / dry run
[q] back
```

Full resume choices skip existing outputs where safe and run missing stages. Clean full reruns print model-specific generated folders, require typed confirmation, and then rebuild Phase 1 through Phase 5.

## Presets

### Full End-To-End Presets

`pythia-full-end-to-end` runs the Pythia A->C workflow from Phase 1 through Phase 5:

```text
Phase 1 dataset construction
Phase 2 behavioural evaluation with answer-containment audit
Phase 3a layer activation patching
Phase 3b component patching
Phase 4a logit lens
Phase 4b attention visualisation
Phase 5b final activation steering at known layer 31
Phase 5b random matched-norm and early-layer controls
Phase 5c helped/hurt analysis
```

Pythia does not run the Qwen calibration sweep. Its runner-side Phase 5 steering config uses `layer=31`, `hook=resid_post`, `alphas=0.0 0.5 1.0 2.0`, `train_frac=0.7`, and `seed=42`.

`qwen-full-end-to-end` runs the Qwen B->D noisy-recovery workflow from Phase 1 through Phase 5:

```text
Phase 1 dataset construction
Phase 2 behavioural evaluation with answer-containment audit
Phase 3a layer activation patching
Phase 3b component patching
Phase 4a logit lens
Phase 4b attention visualisation
Phase 5a steering calibration
Phase 5b final steering from the calibration config
Phase 5b controls
Phase 5c steering analysis
```

Qwen final steering does not hardcode the final layer in the full run. Phase 5a writes `noisy_recommended_steering_config.json`, and Phase 5b reads the selected layer, hook, and alpha range from that config.

Full end-to-end output folders stay model-specific under:

```text
dataset/processed/<model_slug>/
results/phase_2_behaviour/<model_slug>/
results/model_agnostic_evaluation/<model_slug>/
results/phase_3a_layer_patching/<model_slug>/
results/phase_3b_component_patching/<model_slug>/
results/phase_4a_logit_lens/<model_slug>/
results/phase_4b_attention/<model_slug>/
results/phase_5a_steering_calibration/<model_slug>/   # Qwen calibration
results/phase_5b_activation_steering/<model_slug>/
results/phase_5c_steering_analysis/<model_slug>/
```

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
python scripts/run_model_pipeline.py --preset pythia-full-end-to-end --skip-existing
python scripts/run_model_pipeline.py --preset pythia-clean
python scripts/run_model_pipeline.py --preset qwen-full-end-to-end --skip-existing
python scripts/run_model_pipeline.py --preset qwen-noisy-recovery
python scripts/run_model_pipeline.py --preset qwen-clean-degradation
python scripts/run_model_pipeline.py --preset qwen-direct-noise
python scripts/run_model_pipeline.py --preset qwen-structured-noise
python scripts/run_model_pipeline.py --preset qwen-full-spread
python scripts/run_model_pipeline.py --preset qwen-steering-calibration
python scripts/run_model_pipeline.py --preset qwen-steering-final
python scripts/run_model_pipeline.py --preset qwen-steering-controls
python scripts/run_model_pipeline.py --preset qwen-steering-analysis
python scripts/run_model_pipeline.py --preset qwen-steering-full
```

Clean full reruns are available as explicit presets:

```powershell
python scripts/run_model_pipeline.py --preset pythia-full-clean --yes
python scripts/run_model_pipeline.py --preset qwen-full-clean --yes
```

The equivalent flag form is also supported:

```powershell
python scripts/run_model_pipeline.py --preset pythia-full-end-to-end --clean-model --yes
python scripts/run_model_pipeline.py --preset qwen-full-end-to-end --clean-model --yes
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
steering-calibration-oracle
steering-calibration-layer-sweep
steering-final
steering-controls
steering-analysis
```

## Dry Runs

```powershell
python scripts/run_model_pipeline.py --preset pythia-full-end-to-end --dry-run
python scripts/run_model_pipeline.py --preset qwen-full-end-to-end --dry-run
python scripts/run_model_pipeline.py --preset qwen-full-spread --dry-run
python scripts/run_model_pipeline.py --preset qwen-clean-degradation --dry-run
python scripts/run_model_pipeline.py --preset qwen-direct-noise --dry-run
python scripts/run_model_pipeline.py --preset qwen-structured-noise --dry-run
python scripts/run_model_pipeline.py --preset pythia-clean --dry-run
python scripts/run_model_pipeline.py --preset qwen-steering-full --dry-run
python scripts/run_model_pipeline.py --preset pythia-full-clean --dry-run
python scripts/run_model_pipeline.py --preset qwen-full-clean --dry-run
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
results/phase_2_behaviour/<model_slug>/<model_slug>_evaluation_results.csv
results/phase_2_behaviour/<model_slug>/<model_slug>_accuracy_summary.csv
dataset/processed/<model_slug>/contrast_examples.json
dataset/processed/<model_slug>/noisy_contrast_examples.json
dataset/processed/<model_slug>/direct_noise_contrast_examples.json
dataset/processed/<model_slug>/structured_noise_contrast_examples.json
dataset/processed/<model_slug>/clean_degradation_contrast_examples.json
```

Examples for Qwen full-spread Phase 3a:

```text
results/phase_3a_layer_patching/qwen2.5-3b/qwen2.5-3b_clean_layer_patch_summary.csv
results/phase_3a_layer_patching/qwen2.5-3b/qwen2.5-3b_noisy_layer_patch_summary.csv
results/phase_3a_layer_patching/qwen2.5-3b/qwen2.5-3b_direct_noise_layer_patch_summary.csv
results/phase_3a_layer_patching/qwen2.5-3b/qwen2.5-3b_structured_noise_layer_patch_summary.csv
results/phase_3a_layer_patching/qwen2.5-3b/qwen2.5-3b_clean_degradation_layer_patch_summary.csv
```

The same model-plus-contrast prefix pattern applies to component patching, logit lens, attention, steering, and figures.

## Phase 5 Steering

Phase 5 now uses a calibration-first workflow:

```text
Phase 5a: steering calibration diagnostics
Phase 5b: final average activation steering intervention
Phase 5c: post-steering helped/hurt analysis
```

The pipeline order is:

```text
calibration -> final steering -> controls -> post-steering analysis
```

Qwen B->D remains the primary calibration-first Phase 5 target. Steering-only presets can skip earlier phases only when the required dataset and contrast files already exist. Full end-to-end presets include the earlier phases so missing steering prerequisites can be produced before Phase 5 starts.

Pythia full end-to-end uses the existing final steering script directly for A->C at known layer 31. It does not invent a Pythia calibration sweep.

### Steering Presets

`qwen-steering-calibration` runs oracle steering and the late-layer sweep:

```powershell
python scripts/run_model_pipeline.py --preset qwen-steering-calibration
```

`qwen-steering-final` runs learned average steering using the Phase 5a recommended config:

```powershell
python scripts/run_model_pipeline.py --preset qwen-steering-final
```

`qwen-steering-controls` runs random matched-norm and early-layer controls using the same recommended layer and alpha range:

```powershell
python scripts/run_model_pipeline.py --preset qwen-steering-controls
```

`qwen-steering-analysis` runs helped/hurt analysis after final steering:

```powershell
python scripts/run_model_pipeline.py --preset qwen-steering-analysis
```

`qwen-steering-full` runs calibration, final steering, controls, and analysis in order:

```powershell
python scripts/run_model_pipeline.py --preset qwen-steering-full
```

To regenerate clean Qwen Phase 5 outputs from existing prerequisites:

```powershell
python scripts/run_model_pipeline.py --preset qwen-steering-full --clean-phase5 --yes
```

### Prerequisites

Calibration requires:

```text
dataset/processed/qwen2.5-3b/dataset.json
dataset/processed/qwen2.5-3b/noisy_contrast_examples.json
```

Final steering and controls require:

```text
results/phase_5a_steering_calibration/qwen2.5-3b/qwen2.5-3b_noisy_recommended_steering_config.json
```

If the recommended config is missing, the runner stops with:

```text
Run qwen-steering-calibration first.
```

Post-steering analysis requires:

```text
results/phase_5b_activation_steering/qwen2.5-3b/qwen2.5-3b_noisy_steering_results.csv
```

If final steering results are missing, the runner stops with:

```text
Run qwen-steering-final first.
```

### Phase 5 Outputs

Phase 5a calibration:

```text
results/phase_5a_steering_calibration/qwen2.5-3b/qwen2.5-3b_noisy_oracle_steering_summary.csv
results/phase_5a_steering_calibration/qwen2.5-3b/qwen2.5-3b_noisy_layer_sweep_steering_summary.csv
results/phase_5a_steering_calibration/qwen2.5-3b/qwen2.5-3b_noisy_recommended_steering_config.json
results/phase_5a_steering_calibration/qwen2.5-3b/qwen2.5-3b_noisy_steering_calibration_report.md
```

Phase 5b final steering:

```text
results/phase_5b_activation_steering/qwen2.5-3b/qwen2.5-3b_noisy_steering_results.csv
results/phase_5b_activation_steering/qwen2.5-3b/qwen2.5-3b_noisy_steering_summary.csv
results/phase_5b_activation_steering/qwen2.5-3b/qwen2.5-3b_noisy_steering_alpha_sweep.csv
results/phase_5b_activation_steering/qwen2.5-3b/qwen2.5-3b_noisy_steering_report.md
```

Phase 5b controls:

```text
results/phase_5b_activation_steering/qwen2.5-3b/qwen2.5-3b_noisy_random_steering_summary.csv
results/phase_5b_activation_steering/qwen2.5-3b/qwen2.5-3b_noisy_early_layer_steering_summary.csv
```

Phase 5c analysis:

```text
results/phase_5c_steering_analysis/qwen2.5-3b/qwen2.5-3b_noisy_helped_hurt_analysis.csv
results/phase_5c_steering_analysis/qwen2.5-3b/qwen2.5-3b_noisy_helped_hurt_report.md
results/phase_5c_steering_analysis/qwen2.5-3b/qwen2.5-3b_noisy_final_steering_interpretation.md
```

### Clean Phase 5

Interactive Phase 5 presets detect existing Phase 5a/5b/5c outputs and offer:

```text
[1] Resume / skip existing
[2] Delete Phase 5a/5b/5c outputs and rerun
[3] Choose manually
[q] quit
```

Deletion is limited to:

```text
results/phase_5a_steering_calibration/qwen2.5-3b/
figures/phase_5a_steering_calibration/qwen2.5-3b/
results/phase_5b_activation_steering/qwen2.5-3b/
figures/phase_5b_activation_steering/qwen2.5-3b/
results/phase_5c_steering_analysis/qwen2.5-3b/
figures/phase_5c_steering_analysis/qwen2.5-3b/
```

The clean option also removes legacy generated Phase 5 output folders if they exist. It does not delete scripts, docs, datasets, or Phase 1-4 outputs.

Interactive deletion requires typing:

```text
DELETE PHASE 5 QWEN
```

## Overlay

The cross-model layer patch overlay is an analysis utility, not a Phase 5 experiment. The runner calls:

```text
scripts/analysis/layer_patch_overlay.py
```

After both model pipelines are complete:

```powershell
python scripts/run_model_pipeline.py --preset qwen-noisy-recovery --stages overlay
```

or append overlay to a run:

```powershell
python scripts/run_model_pipeline.py --preset qwen-full-spread --run-overlay
```

Overlay outputs are written under:

```text
results/analysis/layer_patch_overlay/
figures/analysis/layer_patch_overlay/
```

Overlay files use the combined prefix `pythia-2.8b_qwen2.5-3b_`.

## Clean Rerun

Model clean reruns delete only generated folders for the selected model slug:

```text
dataset/processed/<model_slug>/
results/*/<model_slug>/
figures/*/<model_slug>/
```

Interactive Pythia clean reruns require:

```text
DELETE PYTHIA GENERATED OUTPUTS
```

Interactive Qwen clean reruns require:

```text
DELETE QWEN GENERATED OUTPUTS
```

Non-interactive clean model commands require `--yes`. Qwen steering-only cleanup remains separate: `--clean-phase5 --yes` deletes only Phase 5 generated outputs and leaves Phase 1-4 outputs in place.

The runner never deletes `dataset/raw/`, source code, scripts, docs, configs, README files, thesis text, `.gitignore`, git history, or outputs for another active model.
