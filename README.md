# Enhancing Reasoning Abilities of Large Language Models

Experimental code and generated artifacts for the honours thesis:

**Enhancing Reasoning Abilities of Large Language Models**  
Bachelor of Engineering (Software Engineering)  
Macquarie University

## Project Overview

This project tests whether structured prompting changes both the behavioural success and internal causal mechanisms used by transformer language models on synthetic two-hop reasoning tasks.

The core task uses aligned fact-composition chains:

```text
A -> B -> C
r2(r1(A)) = C
```

The experiments compare direct and structured few-shot prompting under clean and noisy contexts, then inspect the internal activations with mechanistic interpretability tools.

Primary methods:

- Behavioural exact-match evaluation, with answer-containment auditing as a secondary diagnostic.
- Activation patching with TransformerLens:
  - layer-level mediation curves
  - attention-vs-MLP component decomposition
  - attention-head attribution for the original Pythia run
- Diagnostics:
  - logit lens emergence analysis
  - final-token attention heatmaps
- Phase 5 activation steering:
  - Qwen calibration diagnostics
  - learned average-vector steering
  - random and early-layer controls
  - helped/hurt post-steering analysis

## Models

Current workflows support:

```text
EleutherAI/pythia-2.8b
Qwen/Qwen2.5-3B
```

Outputs are namespaced by model slug, for example:

```text
dataset/processed/pythia-2.8b/
dataset/processed/qwen2.5-3b/
results/phase_3a_layer_patching/qwen2.5-3b/
figures/phase_4b_attention/pythia-2.8b/
```

Generated files inside `results/` and `figures/` also start with the model slug, for example `qwen2.5-3b_noisy_layer_patch_summary.csv`.

Processed datasets and the concise thesis results summary are versioned.
Generated results, figures, pipeline logs, caches, model weights, and local
tool configuration are ignored by Git.

All generation is deterministic (`do_sample=False`, `temperature=0`). The Pythia workflow uses TransformerLens directly. Qwen support is routed through the same phase scripts with model-specific output folders and contrast files.

## Quick Start

From the repository root:

```powershell
.\setup-env\setup.ps1
```

Run the interactive pipeline menu:

```powershell
python scripts/run_model_pipeline.py
```

Common non-interactive runs:

```powershell
python scripts/run_model_pipeline.py --preset pythia-full-end-to-end --skip-existing
python scripts/run_model_pipeline.py --preset qwen-full-end-to-end --skip-existing
python scripts/run_model_pipeline.py --preset qwen-full-spread
python scripts/run_model_pipeline.py --preset qwen-steering-full
```

Dry-run a workflow without GPU work:

```powershell
python scripts/run_model_pipeline.py --preset qwen-full-end-to-end --dry-run
```

## Current Status

Implemented:

- Phase 0: model sanity check
- Phase 1: dataset construction
- Phase 2: behavioural evaluation
- Answer-containment audit for exact-match sensitivity
- Phase 3a: layer-level activation patching
- Phase 3b: component-level patching
- Phase 3b head patching for the original Pythia workflow
- Phase 3c: legacy Pythia clean-vs-noisy comparison
- Phase 4a: logit lens analysis
- Phase 4b: attention visualisation
- Qwen full-spread contrast routing
- Pipeline orchestration with logs, dry runs, status checks, skip-existing, and clean reruns
- Phase 5a: Qwen steering calibration
- Phase 5b: final activation steering and controls
- Phase 5c: helped/hurt steering analysis

## Experiment Design

The dataset contains 200 synthetic two-hop reasoning examples across geography, science, biology, and culture.

Prompt cells:

| Cell | Prompt type | Context |
| --- | --- | --- |
| A | Direct few-shot answers | Clean |
| B | Direct few-shot answers | Noisy, with 3 distractors |
| C | Structured Step 1 / Step 2 | Clean |
| D | Structured Step 1 / Step 2 | Noisy, with 3 distractors |
| E | Filler control | Clean |

Prompt variants are token-aligned for patching. The filler control tests whether length or neutral padding alone explains structured-prompt gains.

## Contrast Definitions

Phase 2 writes contrast files from the same `evaluation_results.csv`.

| Contrast | Criterion | File | Interpretation |
| --- | --- | --- | --- |
| A->C | A wrong and C correct | `contrast_examples.json` | Clean structured improvement |
| B->D | B wrong and D correct | `noisy_contrast_examples.json` | Noisy structured recovery |
| B->A | B wrong and A correct | `direct_noise_contrast_examples.json` | Direct noise damage |
| C->D | C wrong and D correct | `structured_noise_contrast_examples.json` | Structured noise stability |
| C->A | C wrong and A correct | `clean_degradation_contrast_examples.json` | Clean degradation / direct recovery |

The runner skips expensive Qwen full-spread Phase 3/4 work by default for contrasts with fewer than 20 examples, while still saving the low-n contrast files.

## Main Results Snapshot

See `docs/results_analysis/full_results_synthesis.md` for the full analysis. Current high-level findings:

- Pythia shows a clean structured-prompt benefit: Cell C exact-match 52.5% vs Cell A 39.5%.
- Pythia also improves under noisy structured prompting: Cell D 67.0% vs Cell B 42.0%.
- Qwen is strong under Direct/Clean exact-match: Cell A 85.5%.
- Qwen is weak under Direct/Noisy exact-match: Cell B 18.5%.
- Qwen Structured/Noisy recovers strongly: Cell D exact-match 66.5%, containment-aware 98.5%.
- Qwen Structured/Clean underperforms Direct/Clean under exact-match, but containment scoring narrows the gap.
- Filler Cell E remains 0% for both models, supporting the claim that gains are not due to length alone.
- Available causal patching evidence generally points to mid-to-late or late-layer mediation.
- Logit lens and attention heatmaps are diagnostic, not causal evidence.

Canonical final-thesis mechanistic contrasts:

| Model | Contrast | n | Peak layer |
| --- | --- | ---: | --- |
| Pythia-2.8B | A->C clean structured improvement | 38 | L31 |
| Qwen2.5-3B | B->D noisy recovery | 104 | L34 |

The additional Qwen contrast files are retained as behavioural context and
supported pipeline inputs, but they are not presented as canonical Phase 3/4
evidence in the final thesis.

## Pipeline Phases

| Phase | Script | Purpose |
| --- | --- | --- |
| 0 | `scripts/phase_0_sanity/prompt_inference_check.py` | Verify model loading, generation, and activation caching |
| 1 | `scripts/phase_1_dataset/build_dataset.py` | Build aligned synthetic dataset |
| 2 | `scripts/phase_2_behaviour/run_evaluation.py` | Run all prompt cells and write contrast files |
| Audit | `scripts/analysis/answer_containment_audit.py` | Score whether outputs contain the gold answer |
| 3a | `scripts/phase_3a_layer_patching/activation_patching.py` | Layer-level causal mediation |
| 3b | `scripts/phase_3b_component_patching/component_patching.py` | Attention/MLP decomposition |
| 3b | `scripts/phase_3b_component_patching/head_patching.py` | Head attribution for selected Pythia layers |
| 3c | `scripts/phase_3c_cross_condition/cross_condition_patching.py` | Legacy Pythia clean/noisy comparison |
| 4a | `scripts/phase_4a_logit_lens/logit_lens_analysis.py` | Residual-stream answer decodability |
| 4b | `scripts/phase_4b_attention/attention_heatmaps.py` | Final-token attention visualisation |
| 5a | `scripts/phase_5a_steering_calibration/steering_diagnostics.py` | Qwen steering calibration |
| 5b | `scripts/phase_5b_activation_steering/activation_steering.py` | Learned average-vector steering and controls |
| 5c | `scripts/phase_5a_steering_calibration/steering_diagnostics.py` | Helped/hurt analysis mode |

## Pipeline Runner

`scripts/run_model_pipeline.py` is the preferred orchestration entry point. It calls the existing phase scripts, streams child output, writes logs under `logs/pipeline_runs/`, checks expected outputs, and supports both interactive and preset-based usage.

Important presets:

```text
pythia-clean
pythia-full-end-to-end
pythia-full-clean
qwen-noisy-recovery
qwen-clean-degradation
qwen-direct-noise
qwen-structured-noise
qwen-full-spread
qwen-full-end-to-end
qwen-full-clean
qwen-steering-calibration
qwen-steering-final
qwen-steering-controls
qwen-steering-analysis
qwen-steering-full
```

Select stages with:

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

## Phase 5 Steering

Phase 5 tests whether donor-source activation differences can be reused as an intervention:

```text
steering_vector = mean(donor_activation - source_activation)
```

The main Qwen steering target is B->D noisy recovery. The calibrated workflow is:

1. Phase 5a oracle steering and late-layer sweep.
2. Write `<model_slug>_noisy_recommended_steering_config.json`.
3. Phase 5b final average steering using the recommended layer, hook, and alpha range.
4. Phase 5b random matched-norm and early-layer controls.
5. Phase 5c helped/hurt analysis.

Run the whole Qwen steering workflow:

```powershell
python scripts/run_model_pipeline.py --preset qwen-steering-full
```

Regenerate clean Qwen Phase 5 outputs only:

```powershell
python scripts/run_model_pipeline.py --preset qwen-steering-full --clean-phase5 --yes
```

## Repository Structure

```text
dataset/
  raw/
  processed/<model_slug>/
docs/
  results_analysis/full_results_synthesis.md
figures/
  <phase>/<model_slug>/
logs/
  pipeline_runs/
results/
  model_agnostic_evaluation/<model_slug>/
  phase_1_dataset/<model_slug>/
  phase_2_behaviour/<model_slug>/
  phase_3a_layer_patching/<model_slug>/
  phase_3b_component_patching/<model_slug>/
  phase_3c_cross_condition/
  phase_4a_logit_lens/<model_slug>/
  phase_4b_attention/<model_slug>/
  phase_5a_steering_calibration/<model_slug>/
  phase_5b_activation_steering/<model_slug>/
  phase_5c_steering_analysis/<model_slug>/
scripts/
  analysis/
  phase_0_sanity/
  phase_1_dataset/
  phase_2_behaviour/
  phase_3a_layer_patching/
  phase_3b_component_patching/
  phase_3c_cross_condition/
  phase_4a_logit_lens/
  phase_4b_attention/
  phase_5a_steering_calibration/
  phase_5b_activation_steering/
  utils/
setup-env/
  environment.yml
  setup.ps1
```

## Environment Setup

This project uses Conda and a PowerShell setup script.

```powershell
.\setup-env\setup.ps1
```

The setup script creates or updates the `enhancing-reasoning-mi` environment and runs:

```powershell
python scripts/utils/verify_env.py
```

Manual activation:

```powershell
conda activate enhancing-reasoning-mi
```

## Running Individual Scripts

All scripts assume the current working directory is the repository root.

Examples:

```powershell
python scripts/phase_1_dataset/build_dataset.py --model EleutherAI/pythia-2.8b
python scripts/phase_2_behaviour/run_evaluation.py --model Qwen/Qwen2.5-3B --run-containment-audit
python scripts/phase_3a_layer_patching/activation_patching.py --model Qwen/Qwen2.5-3B --source-cell B --donor-cell D
python scripts/phase_3b_component_patching/component_patching.py --model Qwen/Qwen2.5-3B --source-cell B --donor-cell D --layers 31 32 33 34 35
python scripts/phase_4a_logit_lens/logit_lens_analysis.py --model Qwen/Qwen2.5-3B --source-cell B --donor-cell D
python scripts/phase_4b_attention/attention_heatmaps.py --model Qwen/Qwen2.5-3B --source-cell B --donor-cell D --layers 20 31 33 34 35
```

Prefer the pipeline runner for long runs because it handles logs, output checks, and resume behaviour.

## Reproducibility Notes

- Exact-match remains the primary behavioural metric.
- Answer-containment is a secondary diagnostic for cases where the gold answer appears inside a longer response.
- Prompt variants are token-aligned before patching.
- Activation patching injects donor-condition activations into source-condition runs at aligned token positions.
- Layer patching reports gold-answer logit changes after activation injection.
- Component patching is run on selected layers, so it is a focused decomposition rather than a full-network component census.
- Logit lens and attention visualisation should be interpreted as diagnostics, not causal proof.
- Cross-model comparisons should be cautious because model architectures, tokenisation, and contrast sample sizes differ.
- Low-n contrasts are saved but should not be treated as equally reliable patching evidence.

## Thesis Context

This repository supports the empirical component of a thesis examining whether reasoning in large language models emerges from the interaction of:

- memory and contextual reasoning, operationalised through clean vs noisy evidence contexts
- structured reasoning processes, manipulated through direct vs step-structured prompting
- internal representations supporting generalisation, measured through causal patching, diagnostics, and steering

The strongest current thesis-safe claim is:

Structured prompting has model- and context-dependent effects; when it produces a useful behavioural effect, causal mediation is concentrated primarily in later transformer layers.

Methodological precedents include Wang et al. 2022 (IOI circuit), Meng et al. 2022 (ROME/causal tracing), and Elhage et al. 2021 (transformer circuits framework).

## License

For academic research use.
