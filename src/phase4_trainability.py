"""Explicit pretrained LaBraM optimization regimes for locked-split MI validation.

'partial' is the last two transformer blocks, terminal norms and task head.
Never silently guess module locations or unfreeze spatial interpolation.
"""
from __future__ import annotations


def configure_labram_finetuning(model, *, strategy: str, n_outputs: int = 5,
                               last_n_blocks: int = 2) -> dict:
    from torch import nn

    if strategy not in {"probe", "partial", "full"}:
        raise ValueError(f"Unsupported LaBraM regime: {strategy}")
    if not isinstance(last_n_blocks, int) or isinstance(last_n_blocks, bool) or last_n_blocks < 1:
        raise ValueError("last_n_blocks must be a positive integer")
    head = model.get_submodule("final_layer")
    if not isinstance(head, nn.Linear) or head.out_features != n_outputs:
        raise ValueError("Expected final_layer linear classifier with n_outputs classes")
    blocks = model.get_submodule("blocks")
    if not isinstance(blocks, nn.ModuleList) or not len(blocks):
        raise ValueError("Expected a named LaBraM transformer ModuleList: blocks")
    if last_n_blocks > len(blocks):
        raise ValueError("last_n_blocks exceeds the actual transformer depth")
    for param in model.parameters():
        param.requires_grad_(False)
    if strategy == "full":
        for param in model.parameters():
            param.requires_grad_(True)
    else:
        for param in head.parameters():
            param.requires_grad_(True)
        if strategy == "partial":
            for block in blocks[-last_n_blocks:]:
                for param in block.parameters():
                    param.requires_grad_(True)
            for name in ("norm", "fc_norm"):
                layer = getattr(model, name, None)
                if layer is not None:
                    for param in layer.parameters():
                        param.requires_grad_(True)

    names = [name for name, p in model.named_parameters() if p.requires_grad]
    frozen_names = [name for name, p in model.named_parameters() if not p.requires_grad]
    if not names or not any(name.startswith("final_layer.") for name in names):
        raise RuntimeError("Trainability audit lacks the task classifier")
    if strategy == "probe" and any(not name.startswith("final_layer.") for name in names):
        raise RuntimeError("Probe unexpectedly trains pretrained backbone")
    if strategy == "partial":
        selected = tuple(f"blocks.{idx}." for idx in range(len(blocks) - last_n_blocks, len(blocks)))
        if not any(n.startswith(selected) for n in names):
            raise RuntimeError("Partial fine-tuning has no trainable transformer block")
        if any(n.startswith("blocks.") and not n.startswith(selected) for n in names):
            raise RuntimeError("Partial fine-tuning modifies earlier transformer blocks")
    if strategy == "full" and frozen_names:
        raise RuntimeError("Full fine-tuning leaves frozen parameters")
    if strategy != "full" and not frozen_names:
        raise RuntimeError("Frozen regime unexpectedly leaves no frozen parameters")
    return {
        "strategy": strategy, "total_parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "frozen_parameters": sum(p.numel() for p in model.parameters() if not p.requires_grad),
        "transformer_blocks": len(blocks),
        "last_n_blocks": last_n_blocks if strategy == "partial" else None,
        "trainable_parameter_names": names,
        "frozen_parameter_count": len(frozen_names),
    }


def set_finetuning_train_mode(model, strategy: str, last_n_blocks: int = 2) -> None:
    """Keep all frozen feature-extraction modules in eval mode (incl dropout)."""
    if strategy == "full":
        model.train()
        return
    if strategy not in {"probe", "partial"}:
        raise ValueError(strategy)
    model.eval()
    model.get_submodule("final_layer").train()
    if strategy == "partial":
        for block in model.get_submodule("blocks")[-last_n_blocks:]:
            block.train()
        for name in ("norm", "fc_norm"):
            layer = getattr(model, name, None)
            if layer is not None and any(p.requires_grad for p in layer.parameters()):
                layer.train()


def optimizer_groups(model, *, backbone_lr: float, head_lr: float):
    if backbone_lr <= 0 or head_lr <= 0:
        raise ValueError("Learning rates must be positive")
    head = [p for name, p in model.named_parameters()
            if p.requires_grad and name.startswith("final_layer.")]
    backbone = [p for name, p in model.named_parameters()
                if p.requires_grad and not name.startswith("final_layer.")]
    if not head:
        raise RuntimeError("No trainable classification head")
    groups = [{"params": head, "lr": head_lr, "group_name": "head"}]
    if backbone:
        groups.append({"params": backbone, "lr": backbone_lr, "group_name": "backbone"})
    return groups
