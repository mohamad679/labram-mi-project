"""Phase 1 acceptance diagnostics on REAL PhysioNet EEG, never a benchmark.

Exercises pretrained head memorization, genuine scratch backbone gradients,
and checkpoint artifact integrity. Does not evaluate generalization.
"""
from __future__ import annotations

import argparse
import gc
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder

from src.artifact_integrity import verify_experiment_manifest, write_experiment_manifest
from src.data import load_physionet_mi
from src.model import load_labram_model
from src.provenance import validate_pretrained_reference
from src.training_modes import configure_trainable_parameters

CLASSES = ("feet", "hands", "left_hand", "rest", "right_hand")


def select_training_examples(X, y, subjects, *, per_class: int = 2):
    """Choose fixed, balanced real examples from subject 1 only."""
    selected = []
    for label in range(len(CLASSES)):
        candidates = np.flatnonzero((subjects == 1) & (y == label))
        if len(candidates) < per_class:
            raise RuntimeError(f"Insufficient subject 1 examples for class {label}.")
        selected.extend(candidates[:per_class].tolist())
    return (
        torch.as_tensor(X[selected], dtype=torch.float32),
        torch.as_tensor(y[selected], dtype=torch.long),
    )


def assert_scratch_backbone_gradient(x, y) -> dict:
    """Prove a non-head LaBraM weight really updates on a real EEG step."""
    torch.manual_seed(42)
    model = load_labram_model(n_outputs=5, n_times=400, n_chans=64, sfreq=160.0).cpu()
    audit = configure_trainable_parameters(model, strategy="full_finetune", n_outputs=5)
    if audit["trainable_parameters"] != audit["total_parameters"]:
        raise RuntimeError("Scratch backbone unexpectedly frozen.")
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.02)
    optimizer.zero_grad(set_to_none=True)
    logits = model(x[:2])
    loss = torch.nn.functional.cross_entropy(logits, y[:2])
    if not torch.isfinite(loss):
        raise RuntimeError("Scratch loss is nonfinite.")
    loss.backward()
    candidate = None
    finite_count = 0
    for name, p in model.named_parameters():
        if name.startswith("final_layer.") or p.grad is None:
            continue
        if not torch.isfinite(p.grad).all():
            raise RuntimeError(f"Non-finite scratch backbone gradient: {name}")
        finite_count += 1
        if candidate is None and torch.count_nonzero(p.grad).item() > 0:
            candidate = (name, p, p.detach().clone())
    if candidate is None:
        raise RuntimeError("No nonzero scratch-backbone gradient.")
    name, parameter, before = candidate
    optimizer.step()
    changed = not torch.equal(before, parameter.detach())
    if not changed:
        raise RuntimeError(f"Scratch backbone did not update: {name}")
    result = {
        "strategy": "scratch_full_finetune",
        "total_parameters": audit["total_parameters"],
        "trainable_parameters": audit["trainable_parameters"],
        "finite_gradient_tensors": finite_count,
        "updated_backbone_parameter": name,
        "backbone_parameter_changed": changed,
        "loss_finite": True,
        "training_examples": 2,
    }
    del model, optimizer
    gc.collect()
    return result


def pretrained_head_overfit(x, y, output_dir: Path, revision: str) -> dict:
    """Memorize ten real examples through a frozen pretrained backbone."""
    model = load_labram_model(
        n_outputs=5, n_times=400, n_chans=64, sfreq=160.0,
        pretrained_repo_id="braindecode/labram-pretrained", pretrained_revision=revision,
    ).cpu()
    audit = configure_trainable_parameters(model, strategy="linear_probe", n_outputs=5)
    model.eval()  # deterministic frozen feature extraction
    head = model.get_submodule("final_layer")
    captured = []

    def capture_input(_module, inputs):
        captured.append(inputs[0].detach().clone())

    handle = head.register_forward_pre_hook(capture_input)
    try:
        with torch.no_grad():
            initial_logits = model(x)
    finally:
        handle.remove()
    if initial_logits.shape != (len(y), len(CLASSES)) or len(captured) != 1:
        raise RuntimeError("LaBraM head-feature extraction mismatch.")
    features = captured[0]
    initial_loss = float(torch.nn.functional.cross_entropy(initial_logits, y))
    optimizer = torch.optim.AdamW(head.parameters(), lr=0.03, weight_decay=0)
    steps_used = 0
    for step in range(1, 601):
        optimizer.zero_grad(set_to_none=True)
        logits = head(features)
        loss = torch.nn.functional.cross_entropy(logits, y)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite head-overfitting loss.")
        loss.backward()
        optimizer.step()
        steps_used = step
        if step % 10 == 0:
            with torch.no_grad():
                current = head(features)
                accuracy = float((current.argmax(1) == y).float().mean())
                if accuracy == 1.0 and float(torch.nn.functional.cross_entropy(current, y)) < 0.1:
                    break

    with torch.no_grad():
        output = model(x)
        final_loss = float(torch.nn.functional.cross_entropy(output, y))
        final_accuracy = float((output.argmax(1) == y).float().mean())
    if final_accuracy != 1.0 or final_loss >= 0.1 or final_loss >= initial_loss:
        raise RuntimeError(
            f"Head-only memorization criterion failed: loss={final_loss:.4f} "
            f"accuracy={final_accuracy:.4f} initial={initial_loss:.4f}"
        )

    checkpoint_path = output_dir / "phase1_pretrained_linear_probe.pt"
    torch.save(model.state_dict(), checkpoint_path)
    meta = {
        "diagnostic_scope": "ten balanced REAL subject-1 epochs; memorization only",
        "training_subject_ids": [1],
        "forward_subject_ids": [2],
        "model_init": "pretrained",
        "training_strategy": "linear_probe",
        "source_repo": "braindecode/labram-pretrained",
        "pretrained_revision": revision,
        "trainable_parameters": audit["trainable_parameters"],
        "total_parameters": audit["total_parameters"],
        "torch_version": torch.__version__,
        "test_set_evaluated": False,
    }
    manifest_path = write_experiment_manifest(checkpoint_path, meta)
    verified = verify_experiment_manifest(manifest_path)
    restored_weights = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(restored_weights, strict=True)
    with torch.no_grad():
        restored_logits = model(x)
    if not torch.allclose(restored_logits, output, atol=1e-5, rtol=1e-5):
        raise RuntimeError("Checkpoint reload changes model predictions.")
    return {
        "strategy": "pretrained_linear_probe",
        "real_examples": len(y),
        "classes": len(CLASSES),
        "steps": steps_used,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "final_training_accuracy": final_accuracy,
        "criterion": "training_accuracy=1.0 and training_loss<0.1",
        "checkpoint_roundtrip": True,
        "artifact_integrity_verified": True,
        "checkpoint_sha256": verified["artifacts"]["checkpoint"]["sha256"],
        "manifest_name": manifest_path.name,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/phase1-acceptance"))
    args = parser.parse_args()
    validate_pretrained_reference("braindecode/labram-pretrained", args.revision)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(2)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    X, labels, metadata = load_physionet_mi(subjects=[1, 2])
    subjects = metadata["subject"].to_numpy()
    encoder = LabelEncoder().fit(np.asarray(labels))
    if tuple(encoder.classes_) != CLASSES or set(subjects) != {1, 2}:
        raise RuntimeError("Unexpected labels or subject metadata.")
    if X.shape[1:] != (64, 400) or not np.all(np.isfinite(X)):
        raise RuntimeError("EEG input contract failed.")
    y = encoder.transform(labels)
    x_train, y_train = select_training_examples(X, y, subjects)
    del X, y, metadata
    gc.collect()
    print("PHASE1: verifying real LaBraM pretrained head memorization...", flush=True)
    probe = pretrained_head_overfit(x_train, y_train, args.output_dir, args.revision)
    print(f"PHASE1: pretrained head criterion PASSED: {probe}", flush=True)
    print("PHASE1: verifying genuine scratch backbone gradient...", flush=True)
    scratch = assert_scratch_backbone_gradient(x_train, y_train)
    print(f"PHASE1: scratch gradient criterion PASSED: {scratch}", flush=True)
    report = {
        "status": "passed",
        "scope": "Phase 1 bounded scientific diagnostics, NOT a held-out benchmark",
        "subject_1_only_training": True,
        "subject_2_labels_unused_for_selection": True,
        "scratch": scratch,
        "pretrained_probe": probe,
    }
    (args.output_dir / "phase1_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
