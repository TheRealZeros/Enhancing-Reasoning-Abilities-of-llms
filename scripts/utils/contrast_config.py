"""Shared contrast routing for behavioural and interpretability scripts."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ContrastConfig:
    name: str
    source_cell: str
    donor_cell: str
    contrast_file: str
    output_prefix: str
    description: str
    criterion: str


CONTRAST_CONFIGS: dict[tuple[str, str], ContrastConfig] = {
    ("A", "C"): ContrastConfig(
        name="clean_ac",
        source_cell="A",
        donor_cell="C",
        contrast_file="contrast_examples.json",
        output_prefix="",
        description="Direct/Clean to Structured/Clean",
        criterion="A wrong and C correct",
    ),
    ("B", "D"): ContrastConfig(
        name="noisy_bd",
        source_cell="B",
        donor_cell="D",
        contrast_file="noisy_contrast_examples.json",
        output_prefix="noisy_",
        description="Direct/Noisy to Structured/Noisy",
        criterion="B wrong and D correct",
    ),
    ("B", "A"): ContrastConfig(
        name="direct_noise_ba",
        source_cell="B",
        donor_cell="A",
        contrast_file="direct_noise_contrast_examples.json",
        output_prefix="direct_noise_",
        description="Direct/Noisy to Direct/Clean",
        criterion="B wrong and A correct",
    ),
    ("C", "D"): ContrastConfig(
        name="structured_noise_cd",
        source_cell="C",
        donor_cell="D",
        contrast_file="structured_noise_contrast_examples.json",
        output_prefix="structured_noise_",
        description="Structured/Clean to Structured/Noisy",
        criterion="C wrong and D correct",
    ),
    ("C", "A"): ContrastConfig(
        name="clean_degradation_ca",
        source_cell="C",
        donor_cell="A",
        contrast_file="clean_degradation_contrast_examples.json",
        output_prefix="clean_degradation_",
        description="Structured/Clean to Direct/Clean",
        criterion="C wrong and A correct",
    ),
}


def normalise_cell(cell: str) -> str:
    return cell.strip().upper()


def get_contrast_config(source_cell: str, donor_cell: str) -> ContrastConfig:
    key = (normalise_cell(source_cell), normalise_cell(donor_cell))
    try:
        return CONTRAST_CONFIGS[key]
    except KeyError as exc:
        supported = ", ".join(f"{src}->{dst}" for src, dst in CONTRAST_CONFIGS)
        raise ValueError(f"Unsupported contrast {key[0]}->{key[1]}. Supported: {supported}") from exc


def contrast_path_for(model_slug: str, source_cell: str, donor_cell: str) -> str:
    config = get_contrast_config(source_cell, donor_cell)
    return f"dataset/processed/{model_slug}/{config.contrast_file}"


def output_prefix_for(source_cell: str, donor_cell: str, override: str | None = None) -> str:
    if override is not None:
        return override
    return get_contrast_config(source_cell, donor_cell).output_prefix


def model_file_prefix(model_slug: str, output_prefix: str | None = "") -> str:
    """Prefix generated result/figure filenames with the model slug."""
    return f"{model_slug}_{output_prefix or ''}"
