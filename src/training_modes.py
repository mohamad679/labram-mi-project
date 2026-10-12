"""Explicit training strategies; initialization and optimization are distinct."""
import torch.nn as nn


def configure_trainable_parameters(model: nn.Module, *, strategy: str, n_outputs: int) -> dict:
    """Set trainability deterministically and return auditable parameter counts.

    A scratch model must use full_finetune; a frozen randomly initialized
    backbone is NOT scratch training. Head matching is by explicit module name,
    not an ambiguous scan over layers with matching output sizes.
    """
    if strategy not in {"linear_probe", "full_finetune"}:
        raise ValueError(f"Unsupported training strategy: {strategy}")
    for parameter in model.parameters():
        parameter.requires_grad_(strategy == "full_finetune")
    if strategy == "linear_probe":
        try:
            head = model.get_submodule("final_layer")
        except AttributeError as exc:
            raise RuntimeError("Expected an explicit final_layer classification head") from exc
        if not isinstance(head, nn.Linear) or head.out_features != n_outputs:
            raise RuntimeError("final_layer must be nn.Linear with configured n_outputs")
        for parameter in head.parameters():
            parameter.requires_grad_(True)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if not trainable or (strategy == "full_finetune" and trainable != total):
        raise RuntimeError("Invalid trainable parameter configuration")
    return {"training_strategy": strategy, "total_parameters": total,
            "trainable_parameters": trainable, "frozen_parameters": total - trainable}
