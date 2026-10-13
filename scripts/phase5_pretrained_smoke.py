"""CPU smoke: official CBraMod safetensors and ONE train-subject EEG optimizer step.

Subject 3 belongs to the frozen Train partition (never use held-out Test).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.prepare_physionet_s3 import RUNS, edf_path, fetch_run, unsigned_s3
from src.phase5_cbramod import adapt_cbramod_input, load_pinned_cbramod


def run(mne_root: Path, output: Path) -> dict:
    import torch

    from scripts.run_phase5_cbramod import load_checkpoint_from_hf
    from src.data import load_physionet_mi

    client = unsigned_s3()
    sizes = []
    for run in RUNS:
        size, _ = fetch_run(client, 3, run, mne_root)
        if edf_path(mne_root, 3, run).stat().st_size != size:
            raise RuntimeError("EDF source size differs from fetched bytes")
        sizes.append(size)
    x, y, _metadata = load_physionet_mi(subjects=[3])
    if x.shape[1:] != (64, 400) or not len(y):
        raise RuntimeError(f"Invalid training EEG samples {x.shape}")
    signal = adapt_cbramod_input(np.asarray(x[0]))
    weights = load_checkpoint_from_hf()
    model, provenance = load_pinned_cbramod(weights, device="cpu")
    model.eval()
    for p in model.backbone.parameters():
        p.requires_grad_(False)
    initial = model.head[1].weight.detach().clone()
    optimizer = torch.optim.SGD(model.head.parameters(), lr=0.05)
    prediction = model(torch.from_numpy(signal.copy()).unsqueeze(0))
    if prediction.shape != (1, 5) or not torch.isfinite(prediction).all():
        raise RuntimeError("Pretrained CBraMod forward failed")
    loss = torch.nn.functional.cross_entropy(prediction, torch.tensor([3]))
    loss.backward()
    if any(p.grad is not None for p in model.backbone.parameters()):
        raise RuntimeError("Frozen pretrained model got gradients")
    if not torch.isfinite(model.head[1].weight.grad).all() or not torch.any(
        model.head[1].weight.grad != 0
    ):
        raise RuntimeError("Classification head received no gradient")
    optimizer.step()
    if torch.equal(initial, model.head[1].weight.detach()):
        raise RuntimeError("Classification head not updated")
    result = {"status": "PASS_SINGLE_REAL_TRAIN_EEG_STEP_ONLY",
              "subject_id": 3, "train_partition_only": True,
              "six_official_imagery_runs_staged": len(sizes),
              "signal_shape_after_adapter": list(signal.shape),
              "pretraining": provenance,
              "classification_logits_shape": list(prediction.shape),
              "head_changed_after_optimizer_step": True,
              "frozen_backbone_gradient_none": True,
              "heldout_test_touched": False,
              "not_full_104_subject_training": True}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(result,indent=2), flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mne-root",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    run(args.mne_root,args.output)


if __name__ == "__main__":
    main()
