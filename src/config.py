"""Validated baseline settings; shared source of truth for training."""
from dataclasses import dataclass, fields
from pathlib import Path

import yaml


@dataclass(frozen=True)
class TrainingConfig:
    dataset: str = "physionet_motor_imagery"
    class_names: tuple[str, ...] = ("feet", "hands", "left_hand", "rest", "right_hand")
    excluded_subjects: tuple[int, ...] = (88, 89, 92, 100, 104)
    n_outputs: int = 5
    n_chans: int = 64
    n_times: int = 400
    sfreq: float = 160.0
    test_subjects: int = 15
    val_subjects: int = 14
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    epochs: int = 30
    patience: int = 5
    seed: int = 42
    checkpoint_path: str = "checkpoints/labram_head_finetuned.pt"
    model_init: str = "scratch"
    pretrained_repo_id: str | None = None


def load_training_config(path: str | Path = "configs/baseline.yaml") -> TrainingConfig:
    with Path(path).open(encoding="utf-8") as handle:
        values = yaml.safe_load(handle)
    if not isinstance(values, dict):
        raise ValueError("Configuration must be a YAML mapping.")
    unknown = set(values) - {field.name for field in fields(TrainingConfig)}
    if unknown:
        raise ValueError(f"Unknown training configuration keys: {sorted(unknown)}")
    for key in ("class_names", "excluded_subjects"):
        if key in values:
            if not isinstance(values[key], list):
                raise ValueError(f"{key} must be a YAML list.")
            values[key] = tuple(values[key])
    config = TrainingConfig(**values)
    if config.dataset != "physionet_motor_imagery":
        raise ValueError("Unsupported dataset.")
    if len(config.class_names) != config.n_outputs or len(set(config.class_names)) != config.n_outputs:
        raise ValueError("Class names and n_outputs do not match.")
    if (config.n_chans, config.n_times, config.sfreq) != (64, 400, 160.0):
        raise ValueError("The baseline expects 64 channels, 400 samples at 160 Hz.")
    if min(config.epochs, config.batch_size, config.patience, config.test_subjects, config.val_subjects) < 1:
        raise ValueError("Training, batch and split sizes must be positive.")
    if config.learning_rate <= 0 or config.weight_decay < 0:
        raise ValueError("Invalid optimizer hyperparameters.")
    if config.model_init not in {"scratch", "pretrained"}:
        raise ValueError("model_init must be 'scratch' or 'pretrained'.")
    if config.model_init == "pretrained" and not config.pretrained_repo_id:
        raise ValueError("pretrained_repo_id is required for pretrained mode.")
    if config.model_init == "scratch" and config.pretrained_repo_id:
        raise ValueError("pretrained_repo_id requires model_init=pretrained.")
    return config
