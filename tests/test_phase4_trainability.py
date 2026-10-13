"""Offline tests: optimizer isolation and trainability for all three regimes."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
from torch import nn

from src.phase4_trainability import (
    configure_labram_finetuning,
    optimizer_groups,
    set_finetuning_train_mode,
)


class SmallLabram(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(8, 8)
        self.blocks = nn.ModuleList([nn.Sequential(nn.Linear(8, 8), nn.Dropout(.4)) for _ in range(4)])
        self.norm = nn.LayerNorm(8)
        self.fc_norm = nn.Identity()
        self.final_layer = nn.Linear(8, 5)

    def forward(self, x):
        x = self.encoder(x)
        for layer in self.blocks:
            x = layer(x)
        return self.final_layer(self.norm(x))


@pytest.mark.parametrize("strategy,expected_blocks", [("probe", []), ("partial", [2, 3]), ("full", [0, 1, 2, 3])])
def test_trainability_and_real_optimizer_step(strategy, expected_blocks):
    torch.manual_seed(1)
    model = SmallLabram()
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    audit = configure_labram_finetuning(model, strategy=strategy)
    enabled_blocks = [i for i in range(4) if any(p.requires_grad for p in model.blocks[i].parameters())]
    assert enabled_blocks == expected_blocks
    assert audit["trainable_parameters"] + audit["frozen_parameters"] == audit["total_parameters"]
    set_finetuning_train_mode(model, strategy)
    if strategy == "probe":
        assert not model.blocks[-1].training
    if strategy == "partial":
        assert not model.blocks[0].training and model.blocks[-1].training
    opt = torch.optim.SGD(optimizer_groups(model, backbone_lr=0.01, head_lr=0.1))
    x = torch.randn(24, 8)
    loss = nn.functional.cross_entropy(model(x), torch.arange(24) % 5)
    loss.backward()
    assert model.final_layer.weight.grad is not None
    for name, p in model.named_parameters():
        if not p.requires_grad:
            assert p.grad is None
    opt.step()
    assert not torch.equal(before["final_layer.weight"], model.final_layer.weight)
    for name, p in model.named_parameters():
        if not p.requires_grad:
            assert torch.equal(before[name], p)
    if strategy == "full":
        assert not torch.equal(before["encoder.weight"], model.encoder.weight)
    else:
        assert torch.equal(before["encoder.weight"], model.encoder.weight)


def test_bad_config_is_rejected():
    model = SmallLabram()
    with pytest.raises(ValueError):
        configure_labram_finetuning(model, strategy="scratch")
    with pytest.raises(ValueError):
        configure_labram_finetuning(model, strategy="partial", last_n_blocks=5)
    with pytest.raises(ValueError):
        optimizer_groups(model, backbone_lr=0, head_lr=0.1)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    with pytest.raises(RuntimeError):
        optimizer_groups(model, backbone_lr=0.1, head_lr=0.1)


def test_probe_eval_is_deterministic_with_dropout_layers():
    model = SmallLabram()
    configure_labram_finetuning(model, strategy="probe")
    set_finetuning_train_mode(model, strategy="probe")
    x = torch.randn(5, 8)
    assert torch.equal(model(x), model(x))
