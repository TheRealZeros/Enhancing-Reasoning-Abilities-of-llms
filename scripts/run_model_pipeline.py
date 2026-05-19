#!/usr/bin/env python3
"""Orchestrate existing thesis experiment scripts for a selected model preset."""

import argparse
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


ALL_STAGES = [
    "dataset",
    "evaluation",
    "containment",
    "layer-patching",
    "component-patching",
    "logit-lens",
    "attention",
    "overlay",
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

STAGE_CONTEXT = {
    "dataset": "Phase 1 dataset construction",
    "evaluation": "Phase 2 behavioural evaluation",
    "containment": "Answer-containment post-processing",
    "layer-patching": "Phase 3a layer-level activation patching",
    "component-patching": "Phase 3b component-level patching",
    "logit-lens": "Phase 4a logit lens",
    "attention": "Phase 4b attention visualisation",
    "overlay": "Optional cross-model overlay",
}


@dataclass
class Preset:
    model: str
    source_cell: str
    donor_cell: str
    component_layers: list[int]
    attention_layers: list[int]


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
        model="Qwen/Qwen2.5-3B",
        source_cell="B",
        donor_cell="D",
        component_layers=[31, 32, 33, 34, 35],
        attention_layers=[20, 31, 33, 34, 35],
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


def expected_outputs(stage: str, slug: str, source_cell: str, donor_cell: str) -> list[Path]:
    noisy = is_noisy(source_cell, donor_cell)
    prefix = "noisy_" if noisy else ""

    checks = {
        "dataset": [
            Path(f"dataset/processed/{slug}/dataset.json"),
        ],
        "evaluation": [
            Path(f"results/phase_2_behaviour/{slug}/evaluation_results.csv"),
            Path(f"results/phase_2_behaviour/{slug}/accuracy_summary.csv"),
            Path(f"dataset/processed/{slug}/contrast_examples.json"),
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
    }
    if stage == "evaluation" and noisy:
        checks["evaluation"].append(Path(f"dataset/processed/{slug}/noisy_contrast_examples.json"))
    return checks.get(stage, [])


def build_command(
    stage: str,
    model: str,
    source_cell: str,
    donor_cell: str,
    component_layers: list[int],
    attention_layers: list[int],
    run_containment_with_evaluation: bool = False,
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
        return [
            sys.executable,
            "scripts/phase_3a_layer_patching/activation_patching.py",
            "--model",
            model,
            "--source-cell",
            source_cell,
            "--donor-cell",
            donor_cell,
        ]
    if stage == "component-patching":
        return [
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
    if stage == "logit-lens":
        return [
            sys.executable,
            "scripts/phase_4a_logit_lens/logit_lens_analysis.py",
            "--model",
            model,
            "--source-cell",
            source_cell,
            "--donor-cell",
            donor_cell,
        ]
    if stage == "attention":
        return [
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
    if stage == "overlay":
        return [sys.executable, "scripts/phase_5_cross_model/layer_patch_overlay.py"]
    raise ValueError(f"Unsupported stage: {stage}")


def output_status(stage: str, slug: str, source_cell: str, donor_cell: str) -> tuple[str, list[str]]:
    expected = expected_outputs(stage, slug, source_cell, donor_cell)
    missing = [str(path) for path in expected if not path.exists()]
    return ("exists" if expected and not missing else "missing"), missing


def check_outputs(stage: str, slug: str, source_cell: str, donor_cell: str) -> tuple[bool, list[str]]:
    status, missing = output_status(stage, slug, source_cell, donor_cell)
    return status == "exists", missing


def check_stage_outputs(
    stage: str,
    slug: str,
    source_cell: str,
    donor_cell: str,
    include_containment: bool = False,
) -> tuple[bool, list[str]]:
    stages = [stage]
    if include_containment:
        stages.append("containment")
    missing: list[str] = []
    for stage_name in stages:
        missing.extend(
            str(path)
            for path in expected_outputs(stage_name, slug, source_cell, donor_cell)
            if not path.exists()
        )
    return not missing, missing


def build_config_from_args(args: argparse.Namespace) -> PipelineConfig:
    preset = PRESETS.get(args.preset) if args.preset else None
    if not preset and not args.model:
        raise ValueError("Either --preset or --model is required.")

    model = args.model or preset.model
    source_cell = (args.source_cell or preset.source_cell).upper()
    donor_cell = (args.donor_cell or preset.donor_cell).upper()
    component_layers = parse_layers(args.component_layers, preset.component_layers if preset else [])
    attention_layers = parse_layers(args.attention_layers, preset.attention_layers if preset else [])
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
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
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


def run_pipeline(config: PipelineConfig) -> int:
    slug = model_slug(config.model)
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
            command = build_command(
                stage,
                config.model,
                config.source_cell,
                config.donor_cell,
                config.component_layers,
                config.attention_layers,
                run_containment_with_evaluation=run_containment_with_evaluation,
            )
            command_text = command_to_text(command)

            if config.skip_existing:
                ok, _ = check_outputs(stage, slug, config.source_cell, config.donor_cell)
                if ok:
                    emit(f"[skip] {stage}: expected outputs already exist", log_file)
                    records.append(StageRecord(stage=stage, status="SKIPPED", runtime=0.0))
                    continue

            emit("-" * 70, log_file)
            emit(f"[{index}/{total_stages}] START {stage}", log_file)
            emit(f"Pipeline context: {STAGE_CONTEXT.get(stage, stage)}", log_file)
            emit(f"Script: {script_path_from_command(command)}", log_file)
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
            exit_code = stream_command(
                command,
                log_file,
                stage=stage,
                index=index,
                total_stages=total_stages,
                stage_start=stage_start,
                overall_start=overall_start,
            )
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
    return PipelineConfig(
        preset_label=preset_name,
        model=preset.model,
        source_cell=preset.source_cell,
        donor_cell=preset.donor_cell,
        component_layers=list(preset.component_layers),
        attention_layers=list(preset.attention_layers),
        stages=list(stages or DEFAULT_STAGES),
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


def status_table(config: PipelineConfig, include_overlay: bool = False) -> dict[str, str]:
    slug = model_slug(config.model)
    stages = list(DEFAULT_STAGES)
    if include_overlay:
        stages.append("overlay")
    statuses: dict[str, str] = {}
    print()
    print(f"{'Stage':<22} Status")
    for stage in stages:
        status, _ = output_status(stage, slug, config.source_cell, config.donor_cell)
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
        print("[1] pythia-clean  - Pythia A->C full pipeline")
        print("[2] qwen-noisy    - Qwen B->D noisy full pipeline")
        print("[q] quit")
        answer = input("> ").strip().lower()
        if answer == "1":
            return "pythia-clean"
        if answer == "2":
            return "qwen-noisy"
        if answer == "q":
            return None
        print("Please choose 1, 2, or q.")


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


def interactive_main() -> int:
    preset_name = select_preset_interactively()
    if preset_name is None:
        return 0

    config = config_for_preset(preset_name)
    print_preset_config(config)
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
    parser.add_argument("--stages", default=None, help="Comma-separated stage list.")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run-overlay", action="store_true")
    return parser


def main() -> int:
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
