# Model Pipeline Runner

`scripts/run_model_pipeline.py` is an orchestration helper for the thesis repo. It is not a new experiment phase. It calls the existing runnable scripts in order, streams their output, records a run log, and checks that expected outputs were produced.

The runner does not rewrite or duplicate experiment logic. The underlying scripts remain responsible for dataset construction, behavioural evaluation, containment auditing, patching, logit lens analysis, attention visualisation, and cross-model overlay.

## Interactive Usage

Run the script without arguments for the normal guided workflow:

```powershell
python scripts/run_model_pipeline.py
```

The runner shows the available presets, prints the resolved model configuration, inspects expected outputs, and asks how to proceed. This mode is designed for day-to-day thesis runs where some outputs may already exist.

## Preset Selection

Available presets:

```text
pythia-clean
qwen-noisy
```

### `pythia-clean`

```text
model: EleutherAI/pythia-2.8b
source cell: A
donor cell: C
contrast file: contrast_examples.json
component layers: 24 25 29 30 31
attention layers: 20 30 31
output naming: base filenames
```

### `qwen-noisy`

```text
model: Qwen/Qwen2.5-3B
source cell: B
donor cell: D
contrast file: noisy_contrast_examples.json
component layers: 31 32 33 34 35
attention layers: 20 31 33 34 35
output naming: flat noisy_ filenames
```

## First Run Behaviour

If no expected outputs exist for the selected preset, the interactive runner reports that this appears to be a first run and asks whether to run the full pipeline.

Default preset stages:

```text
dataset,evaluation,containment,layer-patching,component-patching,logit-lens,attention
```

## Existing-Output Behaviour

If existing outputs are found, the interactive runner offers:

```text
Resume: skip stages with existing outputs, run only missing stages
Clean rerun: delete existing outputs for this preset, then run full pipeline
Choose stages manually
Dry run only
Quit
```

## Resume Mode

Resume mode keeps existing generated outputs and only runs stages whose expected outputs are missing:

```text
dataset
evaluation
containment
layer-patching
component-patching
logit-lens
attention
```

In non-interactive mode, use:

```powershell
python scripts/run_model_pipeline.py --preset qwen-noisy --skip-existing
```

## Clean Rerun Mode

Interactive clean rerun prints the exact generated output folders that will be deleted and requires:

```text
DELETE
```

Only selected model outputs are deleted. For the selected slug, the runner may delete:

```text
dataset/processed/<slug>/
results/phase_2_behaviour/<slug>/
results/model_agnostic_evaluation/<slug>/
results/phase_3a_layer_patching/<slug>/
results/phase_3b_component_patching/<slug>/
results/phase_4a_logit_lens/<slug>/
results/phase_4b_attention/<slug>/
figures/phase_3a_layer_patching/<slug>/
figures/phase_3b_component_patching/<slug>/
figures/phase_4a_logit_lens/<slug>/
figures/phase_4b_attention/<slug>/
```

Overlay cleanup is asked separately. If confirmed, the runner may also delete:

```text
results/phase_5_cross_model/
figures/phase_5_cross_model/
```

## Manual Stage Selection

Manual selection asks about each stage. If a stage already has expected outputs, the default is skip. If outputs are missing, the default is run. Overlay is optional and is asked separately.

The runner prints the final plan and asks for confirmation before executing it.

## Non-Interactive Usage

Automation and reproducible command logs still use explicit flags:

```powershell
python scripts/run_model_pipeline.py --preset qwen-noisy
python scripts/run_model_pipeline.py --preset pythia-clean
```

Select stages with comma-separated names:

```powershell
python scripts/run_model_pipeline.py --preset qwen-noisy --stages dataset,evaluation,containment
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
```

Advanced overrides:

```text
--model
--source-cell
--donor-cell
--component-layers
--attention-layers
--stages
--skip-existing
--dry-run
--run-overlay
```

Example:

```powershell
python scripts/run_model_pipeline.py --model Qwen/Qwen2.5-3B --source-cell B --donor-cell D --component-layers "31 32 33 34 35" --attention-layers "20 31 33 34 35"
```

## Dry-Run Usage

Use `--dry-run` to print commands without running them:

```powershell
python scripts/run_model_pipeline.py --preset qwen-noisy --dry-run
python scripts/run_model_pipeline.py --preset pythia-clean --dry-run
```

Interactive mode also offers a dry-run option when existing outputs are detected.

## Logs and Timers

Every run creates a log under:

```text
logs/pipeline_runs/
```

The log includes timestamp, model, preset, stages, source and donor cells, commands, child-process output, success or failure, and runtime per stage.

During real runs, child script output streams live to the terminal and to the log file.

Each streamed line is prefixed with a compact location snapshot:

```text
[2/7 evaluation | scripts/phase_2_behaviour/run_evaluation.py | stage 00:03:12 | total 00:04:01] ...
```

This keeps the current stage, script, stage runtime, and overall runtime visible while the underlying experiment script prints its own output.

## Expected Outputs

The runner checks representative outputs after each stage.

Dataset:

```text
dataset/processed/<model_slug>/dataset.json
```

Evaluation:

```text
results/phase_2_behaviour/<model_slug>/evaluation_results.csv
results/phase_2_behaviour/<model_slug>/accuracy_summary.csv
dataset/processed/<model_slug>/contrast_examples.json
dataset/processed/<model_slug>/noisy_contrast_examples.json   # qwen-noisy only
```

Containment:

```text
results/model_agnostic_evaluation/<model_slug>/answer_containment_summary.csv
results/model_agnostic_evaluation/<model_slug>/answer_containment_audit.md
```

Layer patching:

```text
results/phase_3a_layer_patching/<model_slug>/layer_patch_summary.csv
figures/phase_3a_layer_patching/<model_slug>/layer_patch_curve.png
results/phase_3a_layer_patching/<model_slug>/noisy_layer_patch_summary.csv      # qwen-noisy
figures/phase_3a_layer_patching/<model_slug>/noisy_layer_patch_curve.png        # qwen-noisy
```

Component patching:

```text
results/phase_3b_component_patching/<model_slug>/component_patch_summary.csv
figures/phase_3b_component_patching/<model_slug>/component_patch_heatmap.png
results/phase_3b_component_patching/<model_slug>/noisy_component_patch_summary.csv   # qwen-noisy
figures/phase_3b_component_patching/<model_slug>/noisy_component_patch_heatmap.png   # qwen-noisy
```

Logit lens:

```text
results/phase_4a_logit_lens/<model_slug>/logit_lens_summary.csv
results/phase_4a_logit_lens/<model_slug>/noisy_logit_lens_summary.csv   # qwen-noisy
```

Attention:

```text
results/phase_4b_attention/<model_slug>/attention_manifest.json
results/phase_4b_attention/<model_slug>/noisy_attention_manifest.json   # qwen-noisy
```

Overlay:

```text
results/phase_5_cross_model/layer_patch_overlay_summary.csv
```

## Overlay

After both model pipelines are complete, run the cross-model overlay:

```powershell
python scripts/run_model_pipeline.py --preset qwen-noisy --stages overlay
```

or append it to a normal run:

```powershell
python scripts/run_model_pipeline.py --preset qwen-noisy --run-overlay
```

The overlay script is:

```text
scripts/phase_5_cross_model/layer_patch_overlay.py
```

## What Gets Deleted

Only interactive clean rerun deletes files, and only after typing `DELETE`. It is limited to generated outputs for the selected model slug, plus optional overlay outputs if explicitly confirmed.

## What Never Gets Deleted

The runner never deletes source code, scripts, docs, configs, README files, thesis text, `.gitignore`, git history, or outputs for the other active model preset.
