"""One frozen EEGNet-zscore 67/8 Train fit then ONE historical Validation evaluation.

No validation waveforms are read until the entire inner-Train checkpoint is
frozen and independently reloaded. All original 15 Test subjects remain locked.
"""
from __future__ import annotations

import argparse
import json
import platform
import random
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight

from scripts.run_phase3_validation import SHA256_PHASE2, sha256, validate_frozen_split
from src.phase55_trainonly import epoch_standardize, score_five_class
from src.phase55b_protocol import (
    BATCH_SIZE,
    EPOCHS_MAX,
    INNER_SPLIT_SEED,
    LEARNING_RATE,
    MODEL_SEED,
    PATIENCE,
    SELECTED_MODEL,
    WEIGHT_DECAY,
    PermittedEpochSource,
    make_frozen_final_split,
    verify_train_only_nomination,
)
from src.portable_eeg import LABELS, H5SignalSource, inspect_cache
from src.subject_statistics import subject_cluster_bootstrap


def write_once(path: Path, payload: dict) -> None:
    if path.exists():
        raise FileExistsError(path)
    tmp = path.with_suffix(path.suffix + ".partial")
    if tmp.exists():
        raise FileExistsError(f"Stale incomplete output: {tmp}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def run(args):
    import torch
    from braindecode.models import EEGNet
    from torch import nn
    from torch.utils.data import DataLoader, Dataset

    expected = (args.seed, args.epochs, args.patience, args.batch_size,
                args.learning_rate, args.weight_decay)
    frozen = (MODEL_SEED, EPOCHS_MAX, PATIENCE, BATCH_SIZE,
              LEARNING_RATE, WEIGHT_DECAY)
    if expected != frozen or args.device != "cuda" or args.bootstrap != 2000:
        raise ValueError("Phase 5.5b hyperparameters/device/uncertainty budget are frozen")
    if args.output_dir.exists():
        raise FileExistsError("Existing evidence directory; output must be new")
    if not args.cache.is_file() or sha256(args.cache) != SHA256_PHASE2:
        raise RuntimeError("Original complete immutable HDF5 SHA-256 mismatch")
    evidence = json.loads(args.nomination.read_text(encoding="utf-8"))
    verify_train_only_nomination(evidence)
    labels, subjects, shape = inspect_cache(args.cache, require_complete=True)
    if shape != (18096, 64, 400) or tuple(np.unique(labels)) != LABELS:
        raise ValueError("Wrong immutable EEG cohort schema/labels")
    y = np.searchsorted(np.asarray(LABELS), labels)
    partitions = validate_frozen_split(y, subjects, json.loads(args.split_manifest.read_text()))
    final_split = make_frozen_final_split(partitions, subjects)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required; do not silently substitute CPU")

    # Selecting best epoch is conditioned on eight original TRAIN people ONLY.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    source = H5SignalSource(args.cache, shape)
    train_source = PermittedEpochSource(source, final_split["train"], "train")
    validation_source = PermittedEpochSource(source, final_split["validation"], "validation")

    class EEGDataset(Dataset):
        def __init__(self, permit, indices):
            self.permit = permit
            self.indices = np.asarray(indices, dtype=np.int64)

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, item):
            idx = int(self.indices[item])
            epoch = epoch_standardize(self.permit[idx])
            return torch.from_numpy(np.array(epoch, dtype=np.float32, copy=True)), int(y[idx])

    def probs(loader, model):
        model.eval()
        accumulated = []
        with torch.inference_mode():
            for inputs, _ in loader:
                logits = model(inputs.to(args.device))
                if logits.shape[1:] != (5,) or not torch.isfinite(logits).all():
                    raise RuntimeError("Nonfinite or misshapen EEGNet logits")
                accumulated.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
        out = np.concatenate(accumulated, axis=0)
        if out.shape[1] != 5 or not np.isfinite(out).all() or not np.allclose(
            out.sum(axis=1), 1.0, atol=1e-5
        ):
            raise RuntimeError("Invalid five-class probability output")
        return out

    try:
        fit_ids = final_split["fit"]
        early_ids = final_split["early"]
        val_ids = final_split["validation"]
        fit_loader = DataLoader(EEGDataset(train_source, fit_ids), batch_size=args.batch_size,
                                shuffle=True, num_workers=0)
        early_loader = DataLoader(EEGDataset(train_source, early_ids),
                                  batch_size=args.batch_size, shuffle=False, num_workers=0)
        # Val loader is constructed only after checkpoint is irrevocably selected.
        model = EEGNet(n_chans=64, n_times=400, n_outputs=5, sfreq=160.0).to(args.device)
        classes = np.arange(5)
        weights = compute_class_weight("balanced", classes=classes, y=y[fit_ids])
        loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weights, device=args.device,
                                                          dtype=torch.float32))
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                      weight_decay=args.weight_decay)
        checkpoint = args.output_dir / "best_train_only_weights.pt"
        best_score = -1.0
        best_epoch = None
        best_early_proba = None
        patience_count = 0
        history = []
        began = time.monotonic()
        for epoch in range(1, args.epochs + 1):
            model.train()
            loss_sum = 0.0
            for inputs, truth in fit_loader:
                inputs, truth = inputs.to(args.device), truth.to(args.device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(inputs)
                loss = loss_fn(logits.float(), truth)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Nonfinite Train-only loss at epoch {epoch}")
                loss.backward()
                norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0,
                                                error_if_nonfinite=True)
                if not torch.isfinite(norm):
                    raise RuntimeError("Nonfinite gradient detected; no relaxed fallback")
                optimizer.step()
                loss_sum += float(loss.item()) * len(truth)
            early_probability = probs(early_loader, model)
            early_metric = score_five_class(y[early_ids], early_probability, subjects[early_ids])
            row = {"epoch": epoch, "mean_train_loss": loss_sum / len(fit_ids),
                   "inner_train_only_macro_f1": early_metric["macro_f1"],
                   "inner_train_only_balanced_accuracy": early_metric["balanced_accuracy"]}
            history.append(row)
            print("Train-only epoch", epoch, "inner macro-F1", round(row["inner_train_only_macro_f1"], 5), flush=True)
            if early_metric["macro_f1"] > best_score + 1e-12:
                best_score = early_metric["macro_f1"]
                best_epoch = epoch
                best_early_proba = early_probability.copy()
                patience_count = 0
                temp = checkpoint.with_suffix(".pt.partial")
                torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, temp)
                temp.replace(checkpoint)
            else:
                patience_count += 1
            if patience_count >= args.patience:
                break
        if best_epoch is None or best_early_proba is None or not checkpoint.is_file():
            raise RuntimeError("No best Train-only checkpoint persisted")
        restored = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(restored, strict=True)
        verification = probs(early_loader, model)
        max_delta = float(np.max(np.abs(best_early_proba - verification)))
        if max_delta > 1e-6:
            raise RuntimeError("Selected best checkpoint did not reproduce inner Train predictions")

        # ONCE-ONLY historical 14-person Validation inference; no epoch tuning on it.
        val_loader = DataLoader(EEGDataset(validation_source, val_ids),
                                batch_size=args.batch_size, shuffle=False, num_workers=0)
        val_probs = probs(val_loader, model)  # THE ONE ALLOWED VAL FORWARD PASS.
        metric = score_five_class(y[val_ids], val_probs, subjects[val_ids])
        per_class_f1 = f1_score(y[val_ids], val_probs.argmax(1),
                                labels=classes, average=None, zero_division=0)
        uncertainty = subject_cluster_bootstrap(y[val_ids], val_probs, subjects[val_ids],
                                                n_bootstraps=args.bootstrap, seed=42)
        predicted = np.bincount(val_probs.argmax(1), minlength=5)
        metrics = {"status": "COMPLETED_ONE_TIME_DEVELOPMENT_VALIDATION_NO_TEST",
                   "selected_model": SELECTED_MODEL, "dataset_sha256": SHA256_PHASE2,
                   "split_sha256": sha256(args.split_manifest),
                   "nomination_evidence_sha256": sha256(args.nomination),
                   "code_runner_sha256": sha256(Path(__file__).resolve()),
                   "code_protocol_sha256": sha256(Path(__import__(
                       "src.phase55b_protocol", fromlist=["x"]).__file__)),
                   "code_normalizer_sha256": sha256(Path(__import__(
                       "src.phase55_trainonly", fromlist=["x"]).__file__)),
                   "selection_source": "Frozen prior Train-only 3-fold 75-subject OOF; no historical Validation model choice",
                   "original_train_subjects": 75, "actual_fit_subjects": final_split["fit_subjects"],
                   "actual_early_stop_subjects": final_split["early_subjects"],
                   "fit_subject_count": 67, "early_subject_count": 8,
                   "fit_epoch_count": len(fit_ids), "early_epoch_count": len(early_ids),
                   "heldout_validation_subjects": 14, "validation_epoch_count": len(val_ids),
                   "heldout_test_subjects": 15, "test_waveform_inference": False,
                   "validation_evaluations": 1,
                   "validation_used_for_early_stopping": False,
                   "train_only_early_stop_seed": INNER_SPLIT_SEED,
                   "model_initialization_seed": args.seed,
                   "max_epochs": args.epochs, "patience": args.patience,
                   "batch_size": args.batch_size, "lr": args.learning_rate,
                   "weight_decay": args.weight_decay, "precision": "FP32_no_autocast",
                   "gradient_nonfinite_guard_enabled": True,
                   "weighting": "balanced_cross_entropy_from_fit_subjects_only",
                   "fit_only_class_weights": weights.tolist(),
                   "selected_checkpoint_sha256": sha256(checkpoint),
                   "selected_checkpoint_reloaded": True,
                   "max_abs_inner_probability_reload_difference": max_delta,
                   "selected_epoch": best_epoch,
                   "epochs_executed": len(history), "inner_train_history": history,
                   "validation_metrics": metric,
                   "per_class_f1": per_class_f1.tolist(),
                   "validation_subject_cluster_bootstrap": uncertainty,
                   "single_class_prediction_collapse": int(np.count_nonzero(predicted)) == 1,
                   "reference_only_historical_eegnet_validation_macro_f1": 0.4190536187148316,
                   "elapsed_seconds": round(time.monotonic() - began, 1),
                   "environment": {"python": platform.python_version(),
                                   "numpy": np.__version__, "torch": torch.__version__,
                                   "device": args.device}}
        np.savez_compressed(args.output_dir / "validation_predictions.npz",
                            y_true=y[val_ids], y_prob=val_probs,
                            subject_ids=subjects[val_ids], class_names=np.asarray(LABELS),
                            global_validation_row_indices=val_ids)
        write_once(args.output_dir / "report.json", metrics)
        print("ONE-TIME Validation macro-F1", metric["macro_f1"],
              "balanced accuracy", metric["balanced_accuracy"], flush=True)
        return {"status": metrics["status"], "selected_epoch": best_epoch,
                "validation_macro_f1": metric["macro_f1"], "test_predictions": 0}
    finally:
        source.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path,
                        default=Path("configs/phase3_locked_split.json"))
    parser.add_argument("--nomination", type=Path,
                        default=Path("results/validation/phase55_trainonly_oof_audit_20261013.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=MODEL_SEED)
    parser.add_argument("--epochs", type=int, default=EPOCHS_MAX)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2), flush=True)


if __name__ == "__main__":
    main()
