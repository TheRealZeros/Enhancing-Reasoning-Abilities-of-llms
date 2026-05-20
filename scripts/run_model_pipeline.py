#!/usr/bin/env python3
"""Orchestrate existing thesis experiment scripts for a selected model preset."""

import argparse
import json
import os
import shutil
import stat
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TextIO

try:
    from utils.contrast_config import CONTRAST_CONFIGS, contrast_path_for, get_contrast_config, output_prefix_for
except ModuleNotFoundError:
    from scripts.utils.contrast_config import CONTRAST_CONFIGS, contrast_path_for, get_contrast_config, output_prefix_for


ALL_STAGES = [
    "dataset",
    "evaluation",
    "containment",
    "layer-patching",
    "component-patching",
    "logit-lens",
    "attention",
    "overlay",
    "steering-5a",
    "steering-5a-controls",
    "steering-5b-oracle",
    "steering-5b-layer-sweep",
    "steering-5b-helped-hurt",
    "steering-5b-all",
]

DEFAULT_STAGES = [
    "dataset",
    "evaluation",
    "containment",
    "layer-patching",
    "component-patching",
    "logit-lens",
    "attention",
]

SETUP_STAGES = ["dataset", "evaluation", "containment"]
ANALYSIS_STAGES = ["layer-patching", "component-patching", "logit-lens", "attention"]
STEERING_5A_STAGES = ["steering-5a"]
STEERING_5A_CONTROL_STAGES = ["steering-5a-controls"]
STEERING_5B_STAGES = ["steering-5b-oracle", "steering-5b-layer-sweep", "steering-5b-helped-hurt"]
STEERING_STAGES = STEERING_5A_STAGES + STEERING_5A_CONTROL_STAGES + STEERING_5B_STAGES + ["steering-5b-all"]
QWEN_MODEL = "Qwen/Qwen2.5-3B"
QWEN_LATE_COMPONENT_LAYERS = [31, 32, 33, 34, 35]
QWEN_LATE_ATTENTION_LAYERS = [20, 31, 33, 34, 35]
QWEN_FULL_SPREAD_CONTRASTS = [("A", "C"), ("B", "D"), ("B", "A"), ("C", "D"), ("C", "A")]
OUTPUT_PREFIX_OVERRIDES = {
    ("qwen-full-spread", "A", "C"): "clean_",
}
LOW_N_THRESHOLD = 20

STAGE_CONTEXT = {
    "dataset": "Phase 1 dataset construction",
    "evaluation": "Phase 2 behavioural evaluation",
    "containment": "Answer-containment post-processing",
    "layer-patching": "Phase 3a layer-level activation patching",
    "component-patching": "Phase 3b component-level patching",
    "logit-lens": "Phase 4a logit lens",
    "attention": "Phase 4b attention visualisation",
    "overlay": "Optional cross-model overlay",
    "steering-5a": "Phase 5a activation steering first iteration",
    "steering-5a-controls": "Phase 5a random and early-layer steering controls",
    "steering-5b-oracle": "Phase 5b oracle per-example steering diagnostic",
    "steering-5b-layer-sweep": "Phase 5b late-layer steering sweep",
    "steering-5b-helped-hurt": "Phase 5b helped-vs-hurt analysis",
    "steering-5b-all": "Phase 5b diagnostics bundle",
}


@dataclass
class Preset:
    model: str
    source_cell: str
    donor_cell: str
    component_layers: list[int]
    attention_layers: list[int]
    output_prefix: str | None = None
    full_spread: bool = False
    default_stages: list[str] | None = None
    phase5_steering: bool = False


@dataclass
class PipelineConfig:
    preset_label: str
    model: str
    source_cell: str
    donor_cell: str
    component_layers: list[int]
    attention_layers: list[int]
    stages: list[str]
    skip_existing: bool = False
    dry_run: bool = False
    output_prefix: str | None = None
    full_spread: bool = False
    spread_mode: str = "recommended"
    spread_contrasts: list[tuple[str, str]] | None = None
    phase5_steering: bool = False
    clean_phase5: bool = False
    yes: bool = False


@dataclass
class StageRecord:
    stage: str
    status: str
    runtime: float | None = None
    note: str = ""


PRESETS = {
    "pythia-clean": Preset(
        model="EleutherAI/pythia-2.8b",
        source_cell="A",
        donor_cell="C",
        component_layers=[24, 25, 29, 30, 31],
        attention_layers=[20, 30, 31],
    ),
    "qwen-noisy": Preset(
        model=QWEN_MODEL,
        source_cell="B",
        donor_cell="D",
        component_layers=QWEN_LATE_COMPONENT_LAYERS,
        attention_layers=QWEN_LATE_ATTENTION_LAYERS,
        output_prefix="noisy_",
    ),
    "qwen-noisy-recovery": Preset(
        model=QWEN_MODEL,
        source_cell="B",
        donor_cell="D",
        component_layers=QWEN_LATE_COMPONENT_LAYERS,
        attention_layers=QWEN_LATE_ATTENTION_LAYERS,
        output_prefix="noisy_",
    ),
    "qwen-clean-degradation": Preset(
        model=QWEN_MODEL,
        source_cell="C",
        donor_cell="A",
        component_layers=QWEN_LATE_COMPONENT_LAYERS,
        attention_layers=QWEN_LATE_ATTENTION_LAYERS,
        output_prefix="clean_degradation_",
    ),
    "qwen-direct-noise": Preset(
        model=QWEN_MODEL,
        source_cell="B",
        donor_cell="A",
        component_layers=QWEN_LATE_COMPONENT_LAYERS,
        attention_layers=QWEN_LATE_ATTENTION_LAYERS,
        output_prefix="direct_noise_",
    ),
    "qwen-structured-noise": Preset(
        model=QWEN_MODEL,
        source_cell="C",
        donor_cell="D",
        component_layers=QWEN_LATE_COMPONENT_LAYERS,
        attention_layers=QWEN_LATE_ATTENTION_LAYERS,
        output_prefix="structured_noise_",
    ),
    "qwen-full-spread": Preset(
        model=QWEN_MODEL,
        source_cell="B",
        donor_cell="D",
        component_layers=QWEN_LATE_COMPONENT_LAYERS,
        attention_layers=QWEN_LATE_ATTENTION_LAYERS,
        full_spread=True,
    ),
    "qwen-steering-5a": Preset(
        model=QWEN_MODEL,
        source_cell="B",
        donor_cell="D",
        component_layers=QWEN_LATE_COMPONENT_LAYERS,
        attention_layers=QWEN_LATE_ATTENTION_LAYERS,
        output_prefix="noisy_",
        default_stages=["steering-5a"],
        phase5_steering=True,
    ),
    "qwen-steering-5a-controls": Preset(
        model=QWEN_MODEL,
        source_cell="B",
        donor_cell="D",
        component_layers=QWEN_LATE_COMPONENT_LAYERS,
        attention_layers=QWEN_LATE_ATTENTION_LAYERS,
        output_prefix="noisy_",
        default_stages=["steering-5a-controls"],
        phase5_steering=True,
    ),
    "qwen-steering-5b": Preset(
        model=QWEN_MODEL,
        source_cell="B",
        donor_cell="D",
        component_layers=QWEN_LATE_COMPONENT_LAYERS,
        attention_layers=QWEN_LATE_ATTENTION_LAYERS,
        output_prefix="noisy_",
        default_stages=["steering-5b-all"],
        phase5_steering=True,
    ),
    "qwen-steering-full": Preset(
        model=QWEN_MODEL,
        source_cell="B",
        donor_cell="D",
        component_layers=QWEN_LATE_COMPONENT_LAYERS,
        attention_layers=QWEN_LATE_ATTENTION_LAYERS,
        output_prefix="noisy_",
        default_stages=["steering-5a", "steering-5a-controls", "steering-5b-all"],
        phase5_steering=True,
    ),
}


def model_slug(model_name: str) -> str:
    return model_name.split("/")[-1].lower()


def is_noisy(source_cell: str, donor_cell: str) -> bool:
    return source_cell.upper() == "B" and donor_cell.upper() == "D"


def parse_layers(value: str | None, fallback: list[int]) -> list[int]:
    if not value:
        return fallback
    return [int(part) for part in value.replace(",", " ").split()]


def parse_stages(value: str | None, run_overlay: bool) -> list[str]:
    stages = list(DEFAULT_STAGES) if not value else [part.strip() for part in value.split(",") if part.strip()]
    expanded: list[str] = []
    for stage in stages:
        if stage == "steering-5b-all":
            expanded.extend(STEERING_5B_STAGES)
        else:
            expanded.append(stage)
    stages = expanded
    unknown = [stage for stage in stages if stage not in ALL_STAGES]
    if unknown:
        raise ValueError(f"Unknown stage(s): {', '.join(unknown)}")
    if run_overlay and "overlay" not in stages:
        stages.append("overlay")
    return stages


def command_to_text(command: list[str]) -> str:
    return " ".join(command)


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "included"
    seconds = int(round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def script_path_from_command(command: list[str]) -> str:
    if len(command) > 1 and command[1].endswith(".py"):
        return command[1]
    return command[0]


def configure_console_output() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")


def expected_outputs(
    stage: str,
    slug: str,
    source_cell: str,
    donor_cell: str,
    output_prefix: str | None = None,
) -> list[Path]:
    prefix = output_prefix_for(source_cell, donor_cell, output_prefix)

    checks = {
        "dataset": [
            Path(f"dataset/processed/{slug}/dataset.json"),
        ],
        "evaluation": [
            Path(f"results/phase_2_behaviour/{slug}/evaluation_results.csv"),
            Path(f"results/phase_2_behaviour/{slug}/accuracy_summary.csv"),
            Path(f"dataset/processed/{slug}/contrast_examples.json"),
            Path(f"dataset/processed/{slug}/noisy_contrast_examples.json"),
            Path(f"dataset/processed/{slug}/direct_noise_contrast_examples.json"),
            Path(f"dataset/processed/{slug}/structured_noise_contrast_examples.json"),
            Path(f"dataset/processed/{slug}/clean_degradation_contrast_examples.json"),
        ],
        "containment": [
            Path(f"results/model_agnostic_evaluation/{slug}/answer_containment_summary.csv"),
            Path(f"results/model_agnostic_evaluation/{slug}/answer_containment_audit.md"),
        ],
        "layer-patching": [
            Path(f"results/phase_3a_layer_patching/{slug}/{prefix}layer_patch_summary.csv"),
            Path(f"figures/phase_3a_layer_patching/{slug}/{prefix}layer_patch_curve.png"),
        ],
        "component-patching": [
            Path(f"results/phase_3b_component_patching/{slug}/{prefix}component_patch_summary.csv"),
            Path(f"figures/phase_3b_component_patching/{slug}/{prefix}component_patch_heatmap.png"),
        ],
        "logit-lens": [
            Path(f"results/phase_4a_logit_lens/{slug}/{prefix}logit_lens_summary.csv"),
        ],
        "attention": [
            Path(f"results/phase_4b_attention/{slug}/{prefix}attention_manifest.json"),
        ],
        "overlay": [
            Path("results/phase_5_cross_model/layer_patch_overlay_summary.csv"),
        ],
        "steering-5a": [
            Path(f"results/phase_5a_activation_steering/{slug}/{prefix}steering_results.csv"),
            Path(f"results/phase_5a_activation_steering/{slug}/{prefix}steering_summary.csv"),
            Path(f"results/phase_5a_activation_steering/{slug}/{prefix}steering_alpha_sweep.csv"),
            Path(f"results/phase_5a_activation_steering/{slug}/{prefix}steering_report.md"),
        ],
        "steering-5a-controls": [
            Path(f"results/phase_5a_activation_steering/{slug}/{prefix}random_steering_summary.csv"),
            Path(f"results/phase_5a_activation_steering/{slug}/{prefix}early_layer_steering_summary.csv"),
        ],
        "steering-5b-oracle": [
            Path(f"results/phase_5b_steering_diagnostics/{slug}/{prefix}oracle_steering_summary.csv"),
        ],
        "steering-5b-layer-sweep": [
            Path(f"results/phase_5b_steering_diagnostics/{slug}/{prefix}layer_sweep_steering_summary.csv"),
        ],
        "steering-5b-helped-hurt": [
            Path(f"results/phase_5b_steering_diagnostics/{slug}/{prefix}helped_hurt_report.md"),
            Path(f"results/phase_5b_steering_diagnostics/{slug}/{prefix}steering_diagnostics_report.md"),
        ],
        "steering-5b-all": [
            Path(f"results/phase_5b_steering_diagnostics/{slug}/{prefix}oracle_steering_summary.csv"),
            Path(f"results/phase_5b_steering_diagnostics/{slug}/{prefix}layer_sweep_steering_summary.csv"),
            Path(f"results/phase_5b_steering_diagnostics/{slug}/{prefix}helped_hurt_report.md"),
            Path(f"results/phase_5b_steering_diagnostics/{slug}/{prefix}steering_diagnostics_report.md"),
        ],
    }
    return checks.get(stage, [])


def build_command(
    stage: str,
    model: str,
    source_cell: str,
    donor_cell: str,
    component_layers: list[int],
    attention_layers: list[int],
    run_containment_with_evaluation: bool = False,
    output_prefix: str | None = None,
) -> list[str]:
    if stage == "dataset":
        return [sys.executable, "scripts/phase_1_dataset/build_dataset.py", "--model", model]
    if stage == "evaluation":
        command = [
            sys.executable,
            "scripts/phase_2_behaviour/run_evaluation.py",
            "--model",
            model,
        ]
        if run_containment_with_evaluation:
            command.append("--run-containment-audit")
        return command
    if stage == "containment":
        return [sys.executable, "scripts/analysis/answer_containment_audit.py", "--model", model]
    if stage == "layer-patching":
        command = [
            sys.executable,
            "scripts/phase_3a_layer_patching/activation_patching.py",
            "--model",
            model,
            "--source-cell",
            source_cell,
            "--donor-cell",
            donor_cell,
        ]
        if output_prefix is not None:
            command.extend(["--output-prefix", output_prefix])
        return command
    if stage == "component-patching":
        command = [
            sys.executable,
            "scripts/phase_3b_component_patching/component_patching.py",
            "--model",
            model,
            "--source-cell",
            source_cell,
            "--donor-cell",
            donor_cell,
            "--layers",
            *[str(layer) for layer in component_layers],
        ]
        if output_prefix is not None:
            command.extend(["--output-prefix", output_prefix])
        return command
    if stage == "logit-lens":
        command = [
            sys.executable,
            "scripts/phase_4a_logit_lens/logit_lens_analysis.py",
            "--model",
            model,
            "--source-cell",
            source_cell,
            "--donor-cell",
            donor_cell,
        ]
        if output_prefix is not None:
            command.extend(["--output-prefix", output_prefix])
        return command
    if stage == "attention":
        command = [
            sys.executable,
            "scripts/phase_4b_attention/attention_heatmaps.py",
            "--model",
            model,
            "--source-cell",
            source_cell,
            "--donor-cell",
            donor_cell,
            "--layers",
            *[str(layer) for layer in attention_layers],
        ]
        if output_prefix is not None:
            command.extend(["--output-prefix", output_prefix])
        return command
    if stage == "overlay":
        return [sys.executable, "scripts/phase_5_cross_model/layer_patch_overlay.py"]
    if stage == "steering-5a":
        return [
            sys.executable,
            "scripts/phase_5a_activation_steering/activation_steering.py",
            "--model", model,
            "--source-cell", source_cell,
            "--donor-cell", donor_cell,
            "--layer", "34",
            "--hook", "resid_post",
            "--alphas", "0.0", "0.5", "1.0", "2.0",
            "--train-frac", "0.7",
            "--seed", "42",
        ]
    if stage == "steering-5b-oracle":
        return [
            sys.executable,
            "scripts/phase_5b_steering_diagnostics/steering_diagnostics.py",
            "--model", model,
            "--source-cell", source_cell,
            "--donor-cell", donor_cell,
            "--layer", "34",
            "--hook", "resid_post",
            "--diagnostic", "oracle",
            "--alphas", "0.25", "0.5", "0.75", "1.0",
            "--seed", "42",
        ]
    if stage == "steering-5b-layer-sweep":
        return [
            sys.executable,
            "scripts/phase_5b_steering_diagnostics/steering_diagnostics.py",
            "--model", model,
            "--source-cell", source_cell,
            "--donor-cell", donor_cell,
            "--layers", "31", "32", "33", "34", "35",
            "--hook", "resid_post",
            "--diagnostic", "layer_sweep",
            "--alphas", "0.25", "0.5", "0.75", "1.0",
            "--train-frac", "0.7",
            "--seed", "42",
        ]
    if stage == "steering-5b-helped-hurt":
        slug = model_slug(model)
        prefix = output_prefix_for(source_cell, donor_cell, output_prefix)
        return [
            sys.executable,
            "scripts/phase_5b_steering_diagnostics/steering_diagnostics.py",
            "--model", model,
            "--source-cell", source_cell,
            "--donor-cell", donor_cell,
            "--diagnostic", "helped_hurt",
            "--steering-results",
            f"results/phase_5a_activation_steering/{slug}/{prefix}steering_results.csv",
        ]
    raise ValueError(f"Unsupported stage: {stage}")


def build_stage_commands(
    stage: str,
    model: str,
    source_cell: str,
    donor_cell: str,
    component_layers: list[int],
    attention_layers: list[int],
    run_containment_with_evaluation: bool = False,
    output_prefix: str | None = None,
) -> list[list[str]]:
    if stage == "steering-5a-controls":
        base = [
            sys.executable,
            "scripts/phase_5a_activation_steering/activation_steering.py",
            "--model", model,
            "--source-cell", source_cell,
            "--donor-cell", donor_cell,
            "--layer", "34",
            "--hook", "resid_post",
            "--alphas", "0.0", "0.5", "1.0", "2.0",
            "--train-frac", "0.7",
            "--seed", "42",
        ]
        return [
            [*base, "--control", "random", "--random-seeds", "3"],
            [*base, "--control", "early_layer", "--early-layer", "8"],
        ]
    if stage == "steering-5b-all":
        return [
            build_command("steering-5b-oracle", model, source_cell, donor_cell, component_layers, attention_layers, output_prefix=output_prefix),
            build_command("steering-5b-layer-sweep", model, source_cell, donor_cell, component_layers, attention_layers, output_prefix=output_prefix),
            build_command("steering-5b-helped-hurt", model, source_cell, donor_cell, component_layers, attention_layers, output_prefix=output_prefix),
        ]
    return [
        build_command(
            stage,
            model,
            source_cell,
            donor_cell,
            component_layers,
            attention_layers,
            run_containment_with_evaluation=run_containment_with_evaluation,
            output_prefix=output_prefix,
        )
    ]


def output_status(
    stage: str,
    slug: str,
    source_cell: str,
    donor_cell: str,
    output_prefix: str | None = None,
) -> tuple[str, list[str]]:
    expected = expected_outputs(stage, slug, source_cell, donor_cell, output_prefix)
    missing = [str(path) for path in expected if not path.exists()]
    return ("exists" if expected and not missing else "missing"), missing


def check_outputs(
    stage: str,
    slug: str,
    source_cell: str,
    donor_cell: str,
    output_prefix: str | None = None,
) -> tuple[bool, list[str]]:
    status, missing = output_status(stage, slug, source_cell, donor_cell, output_prefix)
    return status == "exists", missing


def check_stage_outputs(
    stage: str,
    slug: str,
    source_cell: str,
    donor_cell: str,
    output_prefix: str | None = None,
    include_containment: bool = False,
) -> tuple[bool, list[str]]:
    stages = [stage]
    if include_containment:
        stages.append("containment")
    missing: list[str] = []
    for stage_name in stages:
        missing.extend(
            str(path)
            for path in expected_outputs(stage_name, slug, source_cell, donor_cell, output_prefix)
            if not path.exists()
        )
    return not missing, missing


def phase5_required_paths(config: PipelineConfig) -> tuple[list[Path], list[Path]]:
    slug = model_slug(config.model)
    prefix = output_prefix_for(config.source_cell, config.donor_cell, config.output_prefix)
    needs_5a = any(stage in {"steering-5a", "steering-5a-controls"} for stage in config.stages)
    needs_5b = any(stage in {"steering-5b-oracle", "steering-5b-layer-sweep", "steering-5b-helped-hurt", "steering-5b-all"} for stage in config.stages)
    produces_5a = "steering-5a" in config.stages

    required: list[Path] = []
    recommended: list[Path] = []
    if needs_5a or needs_5b:
        required.extend([
            Path(f"dataset/processed/{slug}/dataset.json"),
            Path(f"dataset/processed/{slug}/{get_contrast_config(config.source_cell, config.donor_cell).contrast_file}"),
        ])
    if needs_5a:
        required.append(Path(f"results/phase_2_behaviour/{slug}/evaluation_results.csv"))
        recommended.append(Path(f"results/phase_3a_layer_patching/{slug}/{prefix}layer_patch_summary.csv"))
    if needs_5b and not produces_5a:
        required.append(Path(f"results/phase_5a_activation_steering/{slug}/{prefix}steering_results.csv"))
    return required, recommended


def print_phase5_prerequisites(config: PipelineConfig) -> tuple[list[Path], list[Path], list[Path], list[Path]]:
    required, recommended = phase5_required_paths(config)
    missing_required = [path for path in required if not path.exists()]
    missing_recommended = [path for path in recommended if not path.exists()]
    if required or recommended:
        print()
        print("Phase 5 prerequisite check:")
        for path in required:
            status = "exists" if path.exists() else "MISSING"
            print(f"  required    {status:<7} {path}")
        for path in recommended:
            status = "exists" if path.exists() else "missing"
            print(f"  recommended {status:<7} {path}")
    return required, recommended, missing_required, missing_recommended


def check_phase5_prerequisites(config: PipelineConfig, interactive: bool = False) -> bool:
    if not any(stage in STEERING_STAGES for stage in config.stages):
        return True
    _required, _recommended, missing_required, missing_recommended = print_phase5_prerequisites(config)
    if missing_required:
        print()
        for path in missing_required:
            print(f"[missing] {path}")
        if any("phase_5a_activation_steering" in str(path) for path in missing_required):
            print("Run qwen-steering-5a before Phase 5b diagnostics.")
        else:
            print("Required Phase 1/2 prerequisite outputs are missing. Run the Qwen setup/evaluation pipeline first.")
        if config.dry_run:
            print("[dry-run] Missing required prerequisites reported; commands will be printed but not executed.")
            return True
        return False
    if missing_recommended:
        print()
        print("[warning] Recommended Phase 3a layer-patching output is missing.")
        print("[warning] Layer 34 is based on prior known Qwen B->D results.")
        if interactive and not ask_yes_no("Continue with Phase 5 steering anyway?", default=False):
            return False
    return True


def build_config_from_args(args: argparse.Namespace) -> PipelineConfig:
    preset = PRESETS.get(args.preset) if args.preset else None
    if not preset and not args.model:
        raise ValueError("Either --preset or --model is required.")

    model = args.model or preset.model
    source_cell = (args.source_cell or preset.source_cell).upper()
    donor_cell = (args.donor_cell or preset.donor_cell).upper()
    component_layers = parse_layers(args.component_layers, preset.component_layers if preset else [])
    attention_layers = parse_layers(args.attention_layers, preset.attention_layers if preset else [])
    if args.stages:
        stages = parse_stages(args.stages, args.run_overlay)
    elif preset and preset.default_stages is not None:
        stages = parse_stages(",".join(preset.default_stages), args.run_overlay)
    else:
        stages = parse_stages(args.stages, args.run_overlay)
    return PipelineConfig(
        preset_label=args.preset or "custom",
        model=model,
        source_cell=source_cell,
        donor_cell=donor_cell,
        component_layers=component_layers,
        attention_layers=attention_layers,
        stages=stages,
        skip_existing=args.skip_existing,
        dry_run=args.dry_run,
        output_prefix=args.output_prefix if args.output_prefix is not None else (preset.output_prefix if preset else None),
        full_spread=bool(preset.full_spread) if preset else False,
        phase5_steering=bool(preset.phase5_steering) if preset else False,
        clean_phase5=args.clean_phase5,
        yes=args.yes,
    )


def emit(message: str, log_file: TextIO | None = None) -> None:
    print(message)
    if log_file:
        log_file.write(message + "\n")
        log_file.flush()


def stream_command(
    command: list[str],
    log_file: TextIO,
    stage: str,
    index: int,
    total_stages: int,
    stage_start: float,
    overall_start: float,
) -> int:
    script_path = script_path_from_command(command)
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=env,
    )
    assert process.stdout is not None
    for line in process.stdout:
        prefix = (
            f"[{index}/{total_stages} {stage} | {script_path} | "
            f"stage {format_duration(time.time() - stage_start)} | "
            f"total {format_duration(time.time() - overall_start)}] "
        )
        print(prefix + line, end="")
        log_file.write(prefix + line)
        log_file.flush()
    return process.wait()


def print_run_header(config: PipelineConfig, slug: str, log_path: Path, log_file: TextIO) -> None:
    started = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "=" * 70,
        "RUN START",
        f"Preset: {config.preset_label}",
        f"Model: {config.model}",
        f"Slug: {slug}",
        f"Source -> Donor: {config.source_cell} -> {config.donor_cell}",
        f"Stages: {', '.join(config.stages)}",
        f"Started: {started}",
        f"Log file: {log_path}",
        "=" * 70,
    ]
    for line in lines:
        emit(line, log_file)


def print_stage_summary(records: list[StageRecord], total_runtime: float, log_path: Path, log_file: TextIO) -> None:
    emit("", log_file)
    emit("Stage                 Status     Runtime", log_file)
    for record in records:
        emit(f"{record.stage:<21} {record.status:<10} {format_duration(record.runtime)}", log_file)
    emit("", log_file)
    emit(f"Total runtime: {format_duration(total_runtime)}", log_file)
    emit(f"Log file: {log_path}", log_file)


def contrast_count(model: str, source_cell: str, donor_cell: str) -> int | None:
    path = Path(contrast_path_for(model_slug(model), source_cell, donor_cell))
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return len(data) if isinstance(data, list) else None


def contrast_output_prefix(preset_label: str, source_cell: str, donor_cell: str) -> str | None:
    override = OUTPUT_PREFIX_OVERRIDES.get((preset_label, source_cell, donor_cell))
    if override is not None:
        return override
    return output_prefix_for(source_cell, donor_cell)


def print_contrast_counts(model: str, contrast_keys: list[tuple[str, str]]) -> None:
    print()
    print(f"{'Contrast':<12} {'Count':<8} Description")
    for source_cell, donor_cell in contrast_keys:
        cfg = get_contrast_config(source_cell, donor_cell)
        count = contrast_count(model, source_cell, donor_cell)
        count_text = "missing" if count is None else str(count)
        low_n = count is not None and count < LOW_N_THRESHOLD
        suffix = " [low-n]" if low_n else ""
        print(f"{source_cell}->{donor_cell:<8} {count_text:<8} {cfg.description}{suffix}")


def choose_spread_contrasts(config: PipelineConfig) -> list[tuple[str, str]]:
    if config.spread_contrasts is not None:
        return config.spread_contrasts
    if config.spread_mode == "all":
        return list(QWEN_FULL_SPREAD_CONTRASTS)
    selected: list[tuple[str, str]] = []
    for key in QWEN_FULL_SPREAD_CONTRASTS:
        count = contrast_count(config.model, *key)
        if count is None:
            if config.dry_run:
                selected.append(key)
            continue
        if count >= LOW_N_THRESHOLD:
            selected.append(key)
        else:
            cfg = get_contrast_config(*key)
            print(
                f"[low-n] {key[0]}->{key[1]} has {count} examples "
                f"({cfg.description}); skipping by default."
            )
    return selected


def analysis_config_for_contrast(base: PipelineConfig, source_cell: str, donor_cell: str) -> PipelineConfig:
    prefix = contrast_output_prefix(base.preset_label, source_cell, donor_cell)
    return PipelineConfig(
        preset_label=f"{base.preset_label}_{source_cell.lower()}{donor_cell.lower()}",
        model=base.model,
        source_cell=source_cell,
        donor_cell=donor_cell,
        component_layers=list(base.component_layers),
        attention_layers=list(base.attention_layers),
        stages=[stage for stage in base.stages if stage in ANALYSIS_STAGES],
        skip_existing=base.skip_existing,
        dry_run=base.dry_run,
        output_prefix=prefix,
    )


def setup_config_for_full_spread(base: PipelineConfig) -> PipelineConfig:
    return PipelineConfig(
        preset_label=f"{base.preset_label}_setup",
        model=base.model,
        source_cell=base.source_cell,
        donor_cell=base.donor_cell,
        component_layers=list(base.component_layers),
        attention_layers=list(base.attention_layers),
        stages=[stage for stage in base.stages if stage in SETUP_STAGES],
        skip_existing=base.skip_existing,
        dry_run=base.dry_run,
        output_prefix=base.output_prefix,
    )


def run_full_spread_pipeline(config: PipelineConfig) -> int:
    print("[spread] Qwen full spread runs dataset/evaluation once, then repeats Phase 3/4 per contrast.")
    setup = setup_config_for_full_spread(config)
    if setup.stages:
        exit_code = run_pipeline(setup)
        if exit_code != 0:
            return exit_code

    print_contrast_counts(config.model, QWEN_FULL_SPREAD_CONTRASTS)
    selected = choose_spread_contrasts(config)
    if not selected:
        print("[spread] No contrasts selected for Phase 3/4 analysis.")
        return 0

    print()
    print("[spread] Contrasts selected for Phase 3/4:")
    for source_cell, donor_cell in selected:
        count = contrast_count(config.model, source_cell, donor_cell)
        count_text = "unknown" if count is None else str(count)
        low_note = " LOW-N WARNING" if count is not None and count < LOW_N_THRESHOLD else ""
        print(f"  - {source_cell}->{donor_cell}: {count_text} examples{low_note}")

    for source_cell, donor_cell in selected:
        sub_config = analysis_config_for_contrast(config, source_cell, donor_cell)
        if not sub_config.stages:
            continue
        count = contrast_count(config.model, source_cell, donor_cell)
        if count is not None and count < LOW_N_THRESHOLD:
            print(
                f"[low-n] Running expensive stages for {source_cell}->{donor_cell} "
                f"with only {count} examples because it was explicitly selected."
            )
        exit_code = run_pipeline(sub_config)
        if exit_code != 0:
            return exit_code
    if "overlay" in config.stages:
        overlay_config = PipelineConfig(
            preset_label=f"{config.preset_label}_overlay",
            model=config.model,
            source_cell=config.source_cell,
            donor_cell=config.donor_cell,
            component_layers=list(config.component_layers),
            attention_layers=list(config.attention_layers),
            stages=["overlay"],
            skip_existing=config.skip_existing,
            dry_run=config.dry_run,
        )
        return run_pipeline(overlay_config)
    return 0


def run_pipeline(config: PipelineConfig) -> int:
    if config.full_spread:
        return run_full_spread_pipeline(config)

    slug = model_slug(config.model)
    if config.clean_phase5:
        if not config.yes:
            print("[error] --clean-phase5 requires --yes in non-interactive mode.")
            return 1
        if config.dry_run:
            targets = phase5_existing_targets(slug)
            print("[dry-run] Phase 5 generated outputs that would be deleted:")
            if targets:
                for path in targets:
                    print(f"  - {path}")
            else:
                print("  - none")
        else:
            deleted = delete_phase5_outputs(slug)
            print("Deleted Phase 5 generated outputs:")
            if deleted:
                for path in deleted:
                    print(f"  - {path}")
            else:
                print("  - none")

    if not check_phase5_prerequisites(config, interactive=False):
        return 1

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = Path("logs/pipeline_runs") / f"{timestamp}_{config.preset_label}_{slug}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    records: list[StageRecord] = []
    containment_handled_by_evaluation = False
    overall_start = time.time()

    with log_path.open("w", encoding="utf-8") as log_file:
        print_run_header(config, slug, log_path, log_file)
        log_file.write(f"dry_run: {config.dry_run}\n")
        log_file.write(f"skip_existing: {config.skip_existing}\n\n")
        log_file.flush()

        total_stages = len(config.stages)
        for index, stage in enumerate(config.stages, start=1):
            if stage == "containment" and containment_handled_by_evaluation:
                emit(f"[skip] containment already handled by evaluation --run-containment-audit", log_file)
                records.append(StageRecord(stage="containment", status="OK", runtime=None, note="included in evaluation"))
                continue

            run_containment_with_evaluation = stage == "evaluation" and "containment" in config.stages
            commands = build_stage_commands(
                stage,
                config.model,
                config.source_cell,
                config.donor_cell,
                config.component_layers,
                config.attention_layers,
                run_containment_with_evaluation=run_containment_with_evaluation,
                output_prefix=config.output_prefix,
            )
            command_text = "\n".join(command_to_text(command) for command in commands)

            if config.skip_existing:
                ok, _ = check_outputs(stage, slug, config.source_cell, config.donor_cell, config.output_prefix)
                if ok:
                    emit(f"[skip] {stage}: expected outputs already exist", log_file)
                    records.append(StageRecord(stage=stage, status="SKIPPED", runtime=0.0))
                    continue

            emit("-" * 70, log_file)
            emit(f"[{index}/{total_stages}] START {stage}", log_file)
            emit(f"Pipeline context: {STAGE_CONTEXT.get(stage, stage)}", log_file)
            emit(f"Script: {script_path_from_command(commands[0])}", log_file)
            emit(f"Preset/model: {config.preset_label} / {config.model}", log_file)
            emit(f"Source -> Donor: {config.source_cell} -> {config.donor_cell}", log_file)
            emit(f"Command: {command_text}", log_file)
            emit(f"Started: {datetime.now().strftime('%H:%M:%S')}", log_file)
            emit("-" * 70, log_file)

            if config.dry_run:
                if run_containment_with_evaluation:
                    containment_handled_by_evaluation = True
                emit(f"[dry-run] {stage}: command not executed", log_file)
                records.append(StageRecord(stage=stage, status="DRY-RUN", runtime=0.0))
                continue

            stage_start = time.time()
            exit_code = 0
            for command_index, command in enumerate(commands, start=1):
                if len(commands) > 1:
                    emit(f"[subcommand {command_index}/{len(commands)}] {command_to_text(command)}", log_file)
                exit_code = stream_command(
                    command,
                    log_file,
                    stage=stage,
                    index=index,
                    total_stages=total_stages,
                    stage_start=stage_start,
                    overall_start=overall_start,
                )
                if exit_code != 0:
                    break
            runtime = time.time() - stage_start
            elapsed = time.time() - overall_start

            if exit_code != 0:
                records.append(StageRecord(stage=stage, status="FAILED", runtime=runtime))
                emit("-" * 70, log_file)
                emit(f"[{index}/{total_stages}] FAILED {stage}", log_file)
                emit(f"Exit code: {exit_code}", log_file)
                emit(f"Runtime: {format_duration(runtime)}", log_file)
                emit(f"Overall elapsed: {format_duration(elapsed)}", log_file)
                emit(f"Log file: {log_path}", log_file)
                emit("-" * 70, log_file)
                print_stage_summary(records, elapsed, log_path, log_file)
                return exit_code

            ok, missing = check_stage_outputs(
                stage,
                slug,
                config.source_cell,
                config.donor_cell,
                output_prefix=config.output_prefix,
                include_containment=run_containment_with_evaluation,
            )
            if not ok:
                records.append(StageRecord(stage=stage, status="FAILED", runtime=runtime))
                emit("-" * 70, log_file)
                emit(f"[{index}/{total_stages}] FAILED {stage}", log_file)
                emit("Missing expected output(s):", log_file)
                for path in missing:
                    emit(f"  - {path}", log_file)
                emit(f"Runtime: {format_duration(runtime)}", log_file)
                emit(f"Overall elapsed: {format_duration(elapsed)}", log_file)
                emit(f"Log file: {log_path}", log_file)
                emit("-" * 70, log_file)
                print_stage_summary(records, elapsed, log_path, log_file)
                return 1

            records.append(StageRecord(stage=stage, status="OK", runtime=runtime))
            if run_containment_with_evaluation:
                containment_handled_by_evaluation = True

            emit("-" * 70, log_file)
            emit(f"[{index}/{total_stages}] DONE {stage}", log_file)
            emit(f"Runtime: {format_duration(runtime)}", log_file)
            emit(f"Overall elapsed: {format_duration(elapsed)}", log_file)
            emit("-" * 70, log_file)

        total_runtime = time.time() - overall_start
        print_stage_summary(records, total_runtime, log_path, log_file)
    return 0


def config_for_preset(preset_name: str, stages: list[str] | None = None) -> PipelineConfig:
    preset = PRESETS[preset_name]
    selected_stages = stages or preset.default_stages or DEFAULT_STAGES
    return PipelineConfig(
        preset_label=preset_name,
        model=preset.model,
        source_cell=preset.source_cell,
        donor_cell=preset.donor_cell,
        component_layers=list(preset.component_layers),
        attention_layers=list(preset.attention_layers),
        stages=parse_stages(",".join(selected_stages), run_overlay=False),
        output_prefix=preset.output_prefix,
        full_spread=preset.full_spread,
        phase5_steering=preset.phase5_steering,
    )


def print_preset_config(config: PipelineConfig) -> None:
    print()
    print(f"Preset: {config.preset_label}")
    print(f"Model: {config.model}")
    print(f"Slug: {model_slug(config.model)}")
    print(f"Source cell: {config.source_cell}")
    print(f"Donor cell: {config.donor_cell}")
    print(f"Component layers: {' '.join(str(layer) for layer in config.component_layers)}")
    print(f"Attention layers: {' '.join(str(layer) for layer in config.attention_layers)}")
    if config.output_prefix is not None:
        print(f"Output prefix: {config.output_prefix or '(base filenames)'}")


def phase5_status_table(config: PipelineConfig) -> dict[str, str]:
    slug = model_slug(config.model)
    statuses: dict[str, str] = {}
    print()
    print(f"{'Phase 5 stage':<28} Status")
    for stage in config.stages:
        status, _ = output_status(stage, slug, config.source_cell, config.donor_cell, config.output_prefix)
        statuses[stage] = status
        print(f"{stage:<28} {status}")
    return statuses


def status_table(config: PipelineConfig, include_overlay: bool = False) -> dict[str, str]:
    slug = model_slug(config.model)
    stages = list(DEFAULT_STAGES)
    if include_overlay:
        stages.append("overlay")
    statuses: dict[str, str] = {}
    print()
    print(f"{'Stage':<22} Status")
    for stage in stages:
        status, _ = output_status(stage, slug, config.source_cell, config.donor_cell, config.output_prefix)
        statuses[stage] = status
        display_stage = "overlay (optional)" if stage == "overlay" else stage
        print(f"{display_stage:<22} {status}")
    return statuses


def ask_yes_no(prompt: str, default: bool | None = None) -> bool:
    if default is True:
        suffix = " [Y/n] "
    elif default is False:
        suffix = " [y/N] "
    else:
        suffix = " [y/n] "

    while True:
        answer = input(prompt + suffix).strip().lower()
        if not answer and default is not None:
            return default
        if answer in {"y", "yes"}:
            return True
        if answer in {"n", "no"}:
            return False
        print("Please enter y or n.")


def select_preset_interactively() -> str | None:
    while True:
        print("Select pipeline preset:")
        print("[1] pythia-clean")
        print("[2] qwen-noisy-recovery")
        print("[3] qwen-clean-degradation")
        print("[4] qwen-direct-noise")
        print("[5] qwen-structured-noise")
        print("[6] qwen-full-spread")
        print("[7] qwen-steering-5a")
        print("[8] qwen-steering-5a-controls")
        print("[9] qwen-steering-5b")
        print("[10] qwen-steering-full")
        print("[q] quit")
        answer = input("> ").strip().lower()
        if answer == "1":
            return "pythia-clean"
        if answer == "2":
            return "qwen-noisy-recovery"
        if answer == "3":
            return "qwen-clean-degradation"
        if answer == "4":
            return "qwen-direct-noise"
        if answer == "5":
            return "qwen-structured-noise"
        if answer == "6":
            return "qwen-full-spread"
        if answer == "7":
            return "qwen-steering-5a"
        if answer == "8":
            return "qwen-steering-5a-controls"
        if answer == "9":
            return "qwen-steering-5b"
        if answer == "10":
            return "qwen-steering-full"
        if answer == "q":
            return None
        print("Please choose 1-10, or q.")


def clean_targets(slug: str, include_overlay: bool) -> list[Path]:
    targets = [
        Path(f"dataset/processed/{slug}"),
        Path(f"results/phase_2_behaviour/{slug}"),
        Path(f"results/model_agnostic_evaluation/{slug}"),
        Path(f"results/phase_3a_layer_patching/{slug}"),
        Path(f"results/phase_3b_component_patching/{slug}"),
        Path(f"results/phase_4a_logit_lens/{slug}"),
        Path(f"results/phase_4b_attention/{slug}"),
        Path(f"figures/phase_3a_layer_patching/{slug}"),
        Path(f"figures/phase_3b_component_patching/{slug}"),
        Path(f"figures/phase_4a_logit_lens/{slug}"),
        Path(f"figures/phase_4b_attention/{slug}"),
    ]
    if include_overlay:
        targets.extend([
            Path("results/phase_5_cross_model"),
            Path("figures/phase_5_cross_model"),
        ])
    return targets


def phase5_clean_targets(slug: str) -> list[Path]:
    return [
        Path(f"results/phase_5a_activation_steering/{slug}"),
        Path(f"figures/phase_5a_activation_steering/{slug}"),
        Path(f"results/phase_5b_steering_diagnostics/{slug}"),
        Path(f"figures/phase_5b_steering_diagnostics/{slug}"),
    ]


def phase5_existing_targets(slug: str) -> list[Path]:
    return [path for path in phase5_clean_targets(slug) if path.exists()]


def delete_phase5_outputs(slug: str) -> list[Path]:
    targets = phase5_existing_targets(slug)
    delete_targets(targets)
    return targets


def ensure_under_repo(path: Path) -> None:
    repo_root = Path.cwd().resolve()
    path.resolve().relative_to(repo_root)


def handle_remove_readonly(func, path: str, exc_info) -> None:
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except PermissionError:
        raise exc_info[1]


def remove_path(path: Path) -> None:
    if path.is_dir():
        try:
            shutil.rmtree(path, onerror=handle_remove_readonly)
        except PermissionError:
            if path.exists() and not any(path.iterdir()):
                print(f"Warning: left empty directory in place because Windows denied removal: {path}")
                return
            raise
    else:
        try:
            path.unlink()
        except PermissionError:
            os.chmod(path, stat.S_IWRITE)
            path.unlink()


def delete_targets(paths: list[Path]) -> None:
    for path in paths:
        ensure_under_repo(path)
        if not path.exists():
            continue
        remove_path(path)


def clean_rerun_interactive(config: PipelineConfig, statuses: dict[str, str]) -> int:
    slug = model_slug(config.model)
    include_overlay = ask_yes_no("Also delete cross-model overlay outputs?", default=False)
    targets = clean_targets(slug, include_overlay=include_overlay)
    existing_targets = [path for path in targets if path.exists()]

    print()
    print("Clean rerun will delete these generated outputs:")
    if existing_targets:
        for path in existing_targets:
            print(f"  - {path}")
    else:
        print("  - no existing generated outputs for this preset were found")

    confirmation = input("Type DELETE to confirm clean rerun: ").strip()
    if confirmation != "DELETE":
        print("Clean rerun cancelled.")
        return 0

    delete_targets(existing_targets)
    print("Selected generated outputs deleted.")
    if include_overlay and "overlay" not in config.stages:
        config.stages.append("overlay")
    config.skip_existing = False
    config.dry_run = False
    return run_pipeline(config)


def manual_stage_selection(config: PipelineConfig, statuses: dict[str, str]) -> int:
    selected: list[str] = []
    for stage in DEFAULT_STAGES:
        exists = statuses.get(stage) == "exists"
        default = not exists
        status_text = "existing outputs found" if exists else "outputs missing"
        if ask_yes_no(f"Run {stage}? {status_text}", default=default):
            selected.append(stage)

    if ask_yes_no("Run overlay after model pipeline?", default=False):
        selected.append("overlay")

    if not selected:
        print("No stages selected.")
        return 0

    print()
    print("Final plan:")
    print(f"Preset: {config.preset_label}")
    print(f"Stages: {', '.join(selected)}")
    if not ask_yes_no("Execute this plan?", default=False):
        print("Run cancelled.")
        return 0

    config.stages = selected
    config.skip_existing = False
    config.dry_run = False
    return run_pipeline(config)


def configure_full_spread_interactive(config: PipelineConfig) -> int:
    print_contrast_counts(config.model, QWEN_FULL_SPREAD_CONTRASTS)
    print()
    print("Qwen full-spread contrast choice:")
    print("[1] run only recommended contrasts (n >= 20)")
    print("[2] run all contrasts anyway")
    print("[3] choose contrasts manually")
    print("[q] quit")
    while True:
        answer = input("> ").strip().lower()
        if answer == "1":
            config.spread_mode = "recommended"
            return run_pipeline(config)
        if answer == "2":
            config.spread_mode = "all"
            return run_pipeline(config)
        if answer == "3":
            selected: list[tuple[str, str]] = []
            for key in QWEN_FULL_SPREAD_CONTRASTS:
                cfg = get_contrast_config(*key)
                count = contrast_count(config.model, *key)
                default = count is not None and count >= LOW_N_THRESHOLD
                count_text = "missing" if count is None else str(count)
                if ask_yes_no(f"Run {key[0]}->{key[1]} ({cfg.description}, n={count_text})?", default=default):
                    selected.append(key)
            config.spread_mode = "manual"
            config.spread_contrasts = selected
            return run_pipeline(config)
        if answer == "q":
            return 0
        print("Please choose 1, 2, 3, or q.")


def manual_phase5_stage_selection(config: PipelineConfig) -> int:
    choices = [
        "steering-5a",
        "steering-5a-controls",
        "steering-5b-oracle",
        "steering-5b-layer-sweep",
        "steering-5b-helped-hurt",
    ]
    selected: list[str] = []
    for stage in choices:
        if ask_yes_no(f"Run {stage}?", default=stage in config.stages):
            selected.append(stage)
    if not selected:
        print("No Phase 5 stages selected.")
        return 0
    config.stages = selected
    config.skip_existing = False
    config.dry_run = False
    return run_pipeline(config)


def clean_phase5_interactive(config: PipelineConfig) -> int:
    slug = model_slug(config.model)
    targets = phase5_existing_targets(slug)
    print()
    print(f"Existing Phase 5a/5b outputs found for {slug}.")
    print("Delete targets:")
    if targets:
        for path in targets:
            print(f"  - {path}")
    else:
        print("  - none")
    confirmation = input("Type DELETE PHASE 5 QWEN to confirm: ").strip()
    if confirmation != "DELETE PHASE 5 QWEN":
        print("Phase 5 clean cancelled.")
        return 0
    deleted = delete_phase5_outputs(slug)
    print("Deleted Phase 5 generated outputs:")
    if deleted:
        for path in deleted:
            print(f"  - {path}")
    else:
        print("  - none")
    config.skip_existing = False
    config.dry_run = False
    return run_pipeline(config)


def configure_phase5_interactive(config: PipelineConfig) -> int:
    if not check_phase5_prerequisites(config, interactive=True):
        return 1
    statuses = phase5_status_table(config)
    any_existing = bool(phase5_existing_targets(model_slug(config.model)))
    if not any_existing:
        print()
        print("No existing Phase 5a/5b outputs found.")
        if ask_yes_no("Run selected Phase 5 stages now?", default=False):
            return run_pipeline(config)
        return 0

    print()
    print(f"Existing Phase 5a/5b outputs found for {model_slug(config.model)}.")
    print("What do you want to do?")
    print("[1] Resume / skip existing")
    print("[2] Delete Phase 5a/5b outputs and rerun")
    print("[3] Choose manually")
    print("[q] quit")
    while True:
        answer = input("> ").strip().lower()
        if answer == "1":
            config.skip_existing = True
            return run_pipeline(config)
        if answer == "2":
            return clean_phase5_interactive(config)
        if answer == "3":
            return manual_phase5_stage_selection(config)
        if answer == "q":
            return 0
        print("Please choose 1, 2, 3, or q.")


def interactive_main() -> int:
    preset_name = select_preset_interactively()
    if preset_name is None:
        return 0

    config = config_for_preset(preset_name)
    print_preset_config(config)
    if config.phase5_steering:
        return configure_phase5_interactive(config)
    if config.full_spread:
        return configure_full_spread_interactive(config)

    statuses = status_table(config, include_overlay=True)
    any_existing = any(statuses.get(stage) == "exists" for stage in DEFAULT_STAGES)

    if not any_existing:
        print()
        print("No existing outputs found for this preset.")
        if ask_yes_no("Run full pipeline now?", default=False):
            return run_pipeline(config)
        return 0

    print()
    print("Existing outputs found. What do you want to do?")
    print("[1] Resume: skip stages with existing outputs, run only missing stages")
    print("[2] Clean rerun: delete existing outputs for this preset, then run full pipeline")
    print("[3] Choose stages manually")
    print("[4] Dry run only")
    print("[q] quit")
    while True:
        answer = input("> ").strip().lower()
        if answer == "1":
            config.skip_existing = True
            return run_pipeline(config)
        if answer == "2":
            return clean_rerun_interactive(config, statuses)
        if answer == "3":
            return manual_stage_selection(config, statuses)
        if answer == "4":
            config.dry_run = True
            return run_pipeline(config)
        if answer == "q":
            return 0
        print("Please choose 1, 2, 3, 4, or q.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run existing thesis pipeline scripts for one model preset.")
    parser.add_argument("--preset", choices=sorted(PRESETS), default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--source-cell", default=None)
    parser.add_argument("--donor-cell", default=None)
    parser.add_argument("--component-layers", default=None, help="Space- or comma-separated layer list.")
    parser.add_argument("--attention-layers", default=None, help="Space- or comma-separated layer list.")
    parser.add_argument("--output-prefix", default=None, help="Advanced filename prefix override for Phase 3/4 outputs.")
    parser.add_argument("--stages", default=None, help="Comma-separated stage list.")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run-overlay", action="store_true")
    parser.add_argument("--clean-phase5", action="store_true",
                        help="Delete Phase 5a/5b generated outputs for the selected model before running.")
    parser.add_argument("--yes", action="store_true",
                        help="Required with --clean-phase5 in non-interactive mode.")
    return parser


def main() -> int:
    configure_console_output()

    if len(sys.argv) == 1:
        return interactive_main()

    parser = build_parser()
    args = parser.parse_args()

    try:
        config = build_config_from_args(args)
        return run_pipeline(config)
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
