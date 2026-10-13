"""CPU gate: actual public EEG must update the intended LaBraM pretrained blocks."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from src.data import load_physionet_mi
from src.model import load_labram_model
from src.phase4_trainability import (
    configure_labram_finetuning,
    set_finetuning_train_mode,
)


REVISION = "0563b6c626e7b40d9a36653b763715db94d945d7"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/phase4_real_gradient_smoke.json"))
    args = parser.parse_args()
    import torch
    from torch import nn

    torch.manual_seed(42)
    X, y, metadata = load_physionet_mi(subjects=[1])
    if X.shape != (174, 64, 400) or len(metadata) != 174:
        raise ValueError("Unexpected real PhysioNet subject 1 source EEG dimensions")
    class_names = ("feet", "hands", "left_hand", "rest", "right_hand")
    xb = torch.as_tensor(np.asarray(X[:2], dtype=np.float32))
    yb = torch.as_tensor([class_names.index(str(label)) for label in y[:2]], dtype=torch.long)
    reports = []
    for strategy in ("probe", "partial", "full"):
        torch.manual_seed(42)
        model = load_labram_model(
            n_outputs=5, n_times=400, n_chans=64, sfreq=160,
            pretrained_repo_id="braindecode/labram-pretrained",
            pretrained_revision=REVISION,
        )
        audit = configure_labram_finetuning(model, strategy=strategy, last_n_blocks=2)
        set_finetuning_train_mode(model, strategy=strategy, last_n_blocks=2)
        before = {key: p.detach().clone() for key, p in model.named_parameters()}
        opt = torch.optim.SGD((p for p in model.parameters() if p.requires_grad), lr=0.1)
        opt.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = nn.functional.cross_entropy(logits, yb)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Nonfinite {strategy} public EEG loss")
        loss.backward()
        gradients = {
            key: float(p.grad.norm().item()) if p.grad is not None else None
            for key, p in model.named_parameters()
        }
        if not gradients.get("final_layer.weight") or not math.isfinite(gradients["final_layer.weight"]):
            raise RuntimeError(f"No valid head gradient in {strategy}")
        if strategy != "probe" and not any(
            value is not None and value > 0 and math.isfinite(value)
            for key, value in gradients.items() if key.startswith("blocks.")
        ):
            raise RuntimeError(f"No real transformer gradient for {strategy}")
        if any(gradients[key] is not None for key, p in model.named_parameters() if not p.requires_grad):
            raise RuntimeError(f"Frozen weights received gradient for {strategy}")
        opt.step()
        changed = [key for key, p in model.named_parameters() if not torch.equal(before[key], p)]
        if not any(name.startswith("final_layer.") for name in changed):
            raise RuntimeError("Task classifier did not update")
        if strategy == "probe" and any(not key.startswith("final_layer.") for key in changed):
            raise RuntimeError("Unexpected pretrained backbone update in probe")
        if strategy == "partial":
            if not any(name.startswith("blocks.") for name in changed):
                raise RuntimeError("No partial backbone update")
            if any(name.startswith("blocks.") and int(name.split(".")[1]) < len(model.blocks) - 2
                   for name in changed):
                raise RuntimeError("Early transformer block changed during partial fine-tuning")
        if strategy == "full" and not any(name.startswith("blocks.0.") for name in changed):
            raise RuntimeError("Full tuning did not update early transformer block")
        reports.append({
            "strategy": strategy, "loss": float(loss.item()),
            "trainable_parameters": audit["trainable_parameters"],
            "transformer_blocks": audit["transformer_blocks"],
            "changed_parameter_tensors": len(changed),
            "head_gradient_norm": gradients["final_layer.weight"],
            "first_block_gradient_norm": gradients.get("blocks.0.attn.qkv.weight"),
            "real_public_eeg_subject": 1,
            "one_sgd_step_not_training_convergence": True,
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"status": "PASS", "experiments": reports}, indent=2) + "\n")
    print(json.dumps({"status": "PASS", "experiments": reports}, indent=2), flush=True)


if __name__ == "__main__":
    main()
