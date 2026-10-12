"""Regression tests for scientifically distinct initialization/training modes."""
from pathlib import Path

import pytest
nn = pytest.importorskip("torch").nn

from src.config import load_training_config
from src.training_modes import configure_trainable_parameters


class ToyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(8, 8), nn.ReLU())
        self.final_layer = nn.Linear(8, 5)


def test_scratch_full_finetune_updates_every_parameter():
    model = ToyBackbone()
    audit = configure_trainable_parameters(model, strategy="full_finetune", n_outputs=5)
    assert all(p.requires_grad for p in model.parameters())
    assert audit["trainable_parameters"] == audit["total_parameters"]
    assert audit["frozen_parameters"] == 0


def test_pretrained_linear_probe_freezes_only_backbone():
    model = ToyBackbone()
    audit = configure_trainable_parameters(model, strategy="linear_probe", n_outputs=5)
    assert all(not p.requires_grad for p in model.backbone.parameters())
    assert all(p.requires_grad for p in model.final_layer.parameters())
    assert audit["frozen_parameters"] > 0


def test_missing_explicit_head_rejected():
    model = nn.Sequential(nn.Linear(8, 5))
    with pytest.raises(RuntimeError, match="final_layer"):
        configure_trainable_parameters(model, strategy="linear_probe", n_outputs=5)


def test_wrong_head_output_rejected():
    model = ToyBackbone()
    with pytest.raises(RuntimeError, match="n_outputs"):
        configure_trainable_parameters(model, strategy="linear_probe", n_outputs=4)


def test_frozen_scratch_configuration_rejected(tmp_path):
    path = Path(__file__).resolve().parents[1] / "configs" / "baseline.yaml"
    modified = tmp_path / "invalid.yaml"
    modified.write_text(
        path.read_text().replace("training_strategy: full_finetune",
                                 "training_strategy: linear_probe"), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="Scratch requires"):
        load_training_config(modified)
