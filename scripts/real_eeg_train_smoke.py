"""Bounded real EEG optimization smoke test; NOT a scientific benchmark.

One model-head optimization step on public PhysioNet subject 1; a held-out
forward pass from subject 2. Only a sanitized diagnostic JSON is saved.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder

from src.data import load_physionet_mi
from src.model import load_labram_model
from src.provenance import validate_pretrained_reference
from src.training_modes import configure_trainable_parameters


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True, help="Pinned 40-hex HF model commit")
    parser.add_argument("--output", type=Path, default=Path("artifacts/real-eeg-smoke.json"))
    args = parser.parse_args()
    repo_id = "braindecode/labram-pretrained"
    validate_pretrained_reference(repo_id, args.revision)
    torch.set_num_threads(2)
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    X, labels, metadata = load_physionet_mi(subjects=[1, 2])
    subjects = metadata["subject"].to_numpy()
    train_candidates = np.flatnonzero(subjects == 1)
    test_candidates = np.flatnonzero(subjects == 2)
    if len(train_candidates) < 2 or len(test_candidates) < 1:
        raise RuntimeError("Insufficient data from distinct subjects 1 and 2.")
    if X.shape[1:] != (64, 400):
        raise RuntimeError(f"Unexpected EEG shape {X.shape}.")
    classes = ("feet", "hands", "left_hand", "rest", "right_hand")
    encoder = LabelEncoder().fit(np.asarray(labels))
    if tuple(encoder.classes_) != classes:
        raise RuntimeError(f"Unexpected class vocabulary {encoder.classes_}.")
    targets = encoder.transform(labels)

    # Bounded CPU step: 2 real epochs from one participant, 1 held-out epoch.
    x_train = torch.from_numpy(X[train_candidates[:2]].astype(np.float32, copy=True))
    y_train = torch.from_numpy(targets[train_candidates[:2]].astype(np.int64, copy=True))
    x_test = torch.from_numpy(X[test_candidates[:1]].astype(np.float32, copy=True))
    model = load_labram_model(
        n_outputs=5, n_times=400, n_chans=64, sfreq=160.0,
        pretrained_repo_id=repo_id, pretrained_revision=args.revision,
    ).cpu()
    audit = configure_trainable_parameters(model, strategy="linear_probe", n_outputs=5)
    head = model.get_submodule("final_layer")
    if audit["trainable_parameters"] != sum(p.numel() for p in head.parameters()):
        raise RuntimeError("Unexpected probe trainability")
    model.eval()  # Frozen backbone: deterministic inference mode.
    head.train()
    before = [param.detach().clone() for param in head.parameters()]
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3)
    optimizer.zero_grad(set_to_none=True)
    logits = model(x_train)
    loss = torch.nn.functional.cross_entropy(logits, y_train)
    if not torch.isfinite(loss):
        raise RuntimeError("Non-finite training loss on real EEG.")
    loss.backward()
    gradients = [param.grad for param in head.parameters()]
    if not all(grad is not None and torch.isfinite(grad).all() for grad in gradients):
        raise RuntimeError("Head gradients are absent or non-finite.")
    optimizer.step()
    head_updated = any(
        not torch.equal(old, new)
        for old, new in zip(before, head.parameters())
    )
    if not head_updated:
        raise RuntimeError("Head did not update after optimizer step.")
    model.eval()
    with torch.inference_mode():
        heldout = model(x_test)
    if heldout.shape != (1, 5) or not torch.isfinite(heldout).all():
        raise RuntimeError("Held-out real EEG forward pass failed.")

    report = {
        "status": "passed",
        "scope": "One optimizer step on subject 1 and forward on subject 2; NOT a benchmark.",
        "training_subject_count": 1,
        "heldout_subject_count": 1,
        "training_epochs_used": 2,
        "heldout_epochs_used": 1,
        "all_head_gradients_finite": True,
        "head_updated": True,
        "heldout_logit_shape": list(heldout.shape),
        "training_loss_finite": True,
        "checkpoint_revision": args.revision,
        "torch_version": torch.__version__,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
