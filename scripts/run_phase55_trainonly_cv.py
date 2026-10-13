"""Phase 5.5 train-only, subject-grouped 3-fold candidate screening.

No original Validation (14 subjects) or Test (15 subjects) EEG waveform access.
Outputs are exploratory OUT-OF-FOLD predictions for the historical Train only.
A later, independently authorized stage may fit the selected method on all Train
and evaluate the historic Validation exactly once; DO NOT use Test here.
"""
from __future__ import annotations

import argparse
import json
import platform
import random
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from scripts.run_phase3_validation import SHA256_PHASE2, sha256, validate_frozen_split
from src.phase55_trainonly import (
    MODEL_NAMES,
    TrainOnlySignalSource,
    epoch_standardize,
    make_bandpower_features,
    make_trainonly_folds,
    score_five_class,
    select_predeclared,
)
from src.portable_eeg import LABELS, H5SignalSource, inspect_cache


def safe_json(path: Path, payload: dict) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    if tmp.exists():
        raise FileExistsError(f"Interrupted file: {tmp}")
    tmp.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")
    tmp.rename(path)


def bandpower_all_train(source, train_rows, *, batch_size=32):
    """Extract epochs only from original Train, with no fitted parameters."""
    out = []
    for offset in range(0, len(train_rows), batch_size):
        rows = train_rows[offset:offset + batch_size]
        epochs = np.stack([source[int(i)] for i in rows])
        out.append(make_bandpower_features(epochs))
        if (offset // batch_size + 1) % 75 == 0:
            print(f"bandpower features: {offset + len(rows)}/{len(train_rows)} Train rows", flush=True)
    return np.concatenate(out, axis=0)


def bandpower_fold(features, y, train_rows, fold):
    # All feature rows correspond exactly to sorted, frozen original Train indices.
    train_locations = np.searchsorted(train_rows, fold["outer_train"])
    outer_locations = np.searchsorted(train_rows, fold["outer_holdout"])
    if not np.array_equal(train_rows[train_locations], fold["outer_train"]):
        raise ValueError("Train features misaligned")
    if not np.array_equal(train_rows[outer_locations], fold["outer_holdout"]):
        raise ValueError("Outer OOF features misaligned")
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, max_iter=3000, solver="lbfgs",
                           class_weight="balanced", random_state=42),
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=ConvergenceWarning)
        clf.fit(features[train_locations], y[fold["outer_train"]])
    lr = clf.steps[-1][1]
    if not np.array_equal(lr.classes_, np.arange(5)):
        raise ValueError("Bandpower LogisticRegression omitted a class")
    if int(lr.n_iter_[0]) >= lr.max_iter:
        raise RuntimeError("Bandpower LogisticRegression reached iteration limit")
    return clf.predict_proba(features[outer_locations]), {
        "train_fit_subjects": len(set(fold["fit_subjects"] + fold["early_subjects"])),
        "early_stopping": "not_used_for_predeclared_convex_regression",
        "scaler_and_logistic_fit_on": "outer_train_only",
        "solver_iterations": lr.n_iter_.tolist(), "C": 1.0,
    }


def deep_fold(source, y, fold, *, name: str, output_root: Path, args):
    import torch
    from braindecode.models import EEGNet, EEGNeX
    from sklearn.utils.class_weight import compute_class_weight
    from torch import nn
    from torch.utils.data import DataLoader, Dataset

    if name not in ("eegnet_raw", "eegnet_epoch_standardized", "eegnex_epoch_standardized"):
        raise ValueError("Unknown pretrained? Only the three named scratch CNNs are allowed")
    normalize = name.endswith("epoch_standardized")
    fit_rows, early_rows, outer_rows = (fold[k] for k in ("fit", "early", "outer_holdout"))

    class EEGDataset(Dataset):
        def __init__(self, rows):
            self.rows = np.asarray(rows, dtype=np.int64)

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, ix):
            row = int(self.rows[ix])
            eeg = np.asarray(source[row], dtype=np.float32)
            if eeg.shape != (64, 400) or not np.isfinite(eeg).all():
                raise ValueError("Wrong train-only EEG epoch shape")
            if normalize:
                eeg = epoch_standardize(eeg)
            return torch.from_numpy(np.array(eeg, dtype=np.float32, copy=True)), int(y[row])

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if name.startswith("eegnet_"):
        model = EEGNet(n_chans=64, n_times=400, n_outputs=5, sfreq=160.)
    else:
        model = EEGNeX(n_chans=64, n_times=400, n_outputs=5, sfreq=160.,
                        kernel_block_1_2=32, drop_prob=0.5)
    model = model.to(args.device)
    train = DataLoader(EEGDataset(fit_rows), batch_size=args.batch_size, shuffle=True,
                       num_workers=0)
    inner = DataLoader(EEGDataset(early_rows), batch_size=args.batch_size,
                       shuffle=False, num_workers=0)
    outer = DataLoader(EEGDataset(outer_rows), batch_size=args.batch_size,
                       shuffle=False, num_workers=0)
    weights = compute_class_weight(class_weight="balanced", classes=np.arange(5), y=y[fit_rows])
    objective = nn.CrossEntropyLoss(weight=torch.tensor(weights, device=args.device,
                                                       dtype=torch.float32))
    optim = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                              weight_decay=args.weight_decay)

    def eval_prob(loader):
        model.eval()
        predictions = []
        with torch.inference_mode():
            for xb, _ in loader:
                logits = model(xb.to(args.device))
                if logits.ndim != 2 or logits.shape[1] != 5 or not torch.isfinite(logits).all():
                    raise RuntimeError("CNN produced invalid five-class logits")
                predictions.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
        return np.concatenate(predictions, axis=0)

    best_metric = -1.
    best_epoch = None
    stale = 0
    history = []
    out = output_root / name / f"fold{fold['fold']}"
    out.mkdir(parents=True, exist_ok=False)
    checkpoint = out / "best_weights.pt"
    for epoch in range(1, args.epochs + 1):
        model.train()
        sum_loss = 0.
        for xb, yb in train:
            xb = xb.to(args.device)
            yb = yb.to(args.device)
            optim.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = objective(logits.float(), yb)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite {name} fold {fold['fold']} loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0,
                                     error_if_nonfinite=True)
            optim.step()
            sum_loss += float(loss.item()) * len(xb)
        inner_prob = eval_prob(inner)
        inner_metrics = score_five_class(y[early_rows], inner_prob,
                                         args.subjects[early_rows])
        current = inner_metrics["macro_f1"]
        history.append({"epoch": epoch, "train_loss": sum_loss / len(fit_rows),
                        "inner_macro_f1": current,
                        "inner_balanced_accuracy": inner_metrics["balanced_accuracy"]})
        print(f"{name}: fold={fold['fold']} epoch={epoch} inner macroF1={current:.4f}",flush=True)
        if current > best_metric + 1e-12:
            best_metric = current
            best_epoch = epoch
            stale = 0
            tmp = checkpoint.with_suffix(".pt.partial")
            torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, tmp)
            tmp.replace(checkpoint)
        else:
            stale += 1
        if stale >= args.patience:
            break

    if best_epoch is None or not checkpoint.is_file():
        raise RuntimeError("No inner-selected checkpoint")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    probabilities = eval_prob(outer)
    # Reconstruct identical outer predictions after saving and reloading state.
    reload_probs = eval_prob(outer)
    if not np.allclose(probabilities, reload_probs, atol=1e-6, rtol=1e-5):
        raise RuntimeError("Saved CNN checkpoint not reproducible")
    return reload_probs, {
        "best_inner_epoch": best_epoch,
        "inner_history": history,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "checkpoint_sha256": sha256(checkpoint),
        "saved_checkpoint_reloaded": True,
        "fixed_fp32": True,
        "no_original_14_subject_validation_for_early_stopping": True,
        "inner_subjects": fold["early_subjects"],
        "fit_subjects": fold["fit_subjects"],
    }


def main_run(args):
    if args.models != list(MODEL_NAMES) or args.seed != 42:
        raise ValueError("Run all four predeclared candidates without selection shortcuts")
    if not (args.batch_size == 32 and args.epochs == 25 and args.patience == 5
            and args.learning_rate == 0.001 and args.weight_decay == 0.0001):
        raise ValueError("Optimization hyperparameters are frozen for Phase 5.5")
    if args.device not in ("cuda", "cpu"):
        raise ValueError("Unknown explicit execution device")
    if args.output_dir.exists():
        raise FileExistsError("Existing experiment output; choose an empty destination")
    if not args.cache.is_file() or sha256(args.cache) != SHA256_PHASE2:
        raise ValueError("Immutable Phase-2 HDF5 missing or hash differs")
    labels, subjects, shape = inspect_cache(args.cache, require_complete=True)
    if shape != (18096, 64, 400) or tuple(np.unique(labels)) != LABELS:
        raise ValueError("Original HDF5 schema/label vocabulary changed")
    y = np.searchsorted(np.asarray(LABELS), labels)
    manifest = json.loads(args.split_manifest.read_text())
    partitions = validate_frozen_split(y, subjects, manifest)
    folds = make_trainonly_folds(partitions, subjects)
    train_rows = partitions["train"]
    args.subjects = subjects
    source = TrainOnlySignalSource(H5SignalSource(args.cache, shape), train_rows)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    safe_json(args.output_dir / "protocol.json", {
        "phase": "5.5a", "status": "TRAIN_ONLY_CV_SOURCE_LOCKED_NO_BENCHMARK_YET",
        "dataset_sha256": SHA256_PHASE2, "split_sha256": sha256(args.split_manifest),
        "class_names": list(LABELS), "source_runner_sha256": sha256(Path(__file__).resolve()),
        "source_feature_sha256": sha256(Path(__import__("src.phase55_trainonly", fromlist=["x"]).__file__)),
        "candidates": list(MODEL_NAMES), "selection": "pooled_train_only_OOF_macro_f1",
        "original_validation_waveform_read": False, "original_test_waveform_read": False,
        "model_seed": 42, "folds": 3, "inner_early_stop_subjects": 8,
        "epochs": 25, "patience": 5, "batch_size": 32,
        "optimizer_lr": .001, "weight_decay": .0001,
        "torch_precision": "FP32_no_autocast",
        "signal_input": "64 channels x first 400 samples at 160Hz, 8-32Hz MOABB, microvolts",
        "bandpower_bands": [[8, 12], [12, 16], [16, 22], [22, 30]],
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                         "device": args.device},
        "fold_subjects": [
            {"fold": f["fold"], "train_fit": f["fit_subjects"],
             "inner_early_stop": f["early_subjects"],
             "outer_holdout": f["outer_subjects"]} for f in folds],
    })
    try:
        print("TRAIN-ONLY: 75 subjects, 13,050 epochs; NO prior Validation/Test EEG inference",flush=True)
        full_features = bandpower_all_train(source, train_rows)
        reports = {}
        for name in MODEL_NAMES:
            started = time.monotonic()
            all_prob = np.full((len(train_rows), 5), np.nan, dtype=np.float32)
            fold_records = []
            for f in folds:
                if name == "bandpower":
                    probs, extra = bandpower_fold(full_features, y, train_rows, f)
                else:
                    probs, extra = deep_fold(source, y, f, name=name,
                                             output_root=args.output_dir, args=args)
                destination = np.searchsorted(train_rows, f["outer_holdout"])
                if np.isfinite(all_prob[destination]).any():
                    raise RuntimeError("Duplicate outer-fold OOF prediction")
                all_prob[destination] = probs
                fold_score = score_five_class(y[f["outer_holdout"]], probs,
                                               subjects[f["outer_holdout"]])
                fold_records.append({"fold": f["fold"], "outer_subjects": f["outer_subjects"],
                                     "n_outer_epochs": len(destination),
                                     "metrics": fold_score, "training": extra})
                print(f"{name} fold {f['fold']}: OOF Macro-F1={fold_score['macro_f1']:.4f}",flush=True)
            metrics = score_five_class(y[train_rows], all_prob, subjects[train_rows])
            target = args.output_dir / name
            target.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(target / "train_only_oof_predictions.npz",
                                y_true=y[train_rows], y_prob=all_prob,
                                subject_ids=subjects[train_rows],
                                global_train_row_indices=train_rows,
                                class_names=np.asarray(LABELS))
            safe_json(target / "report.json", {
                "mode": name, "status": "TRAIN_ONLY_OOF_COMPLETE", "scores": metrics,
                "fold_records": fold_records,
                "elapsed_seconds": round(time.monotonic()-started, 2),
                "validation_waveform_read": False,
                "test_waveform_read": False,
            })
            reports[name] = metrics
            print(f"{name}: pooled TRAIN OOF Macro-F1={metrics['macro_f1']:.4f}",flush=True)
        selection = select_predeclared(reports)
        summary = {
            "phase": "5.5a", "status": "TRAIN_ONLY_OOF_COMPLETE_NO_HISTORIC_VALIDATION",
            "selection": selection, "scores": reports,
            "training_cohort_subjects": 75, "training_cohort_epochs": 13050,
            "prior_validation_subjects_reserved": 14, "test_subjects_reserved": 15,
            "validation_waveform_inference": False, "test_waveform_inference": False,
            "no_claim_of_validation_or_test_generalization": True,
            "protocol_sha256": sha256(args.output_dir / "protocol.json"),
        }
        safe_json(args.output_dir / "summary.json", summary)
        return summary
    finally:
        source.source.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--split-manifest", type=Path, default=Path("configs/phase3_locked_split.json"))
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--models", nargs="+", default=list(MODEL_NAMES))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--learning-rate", type=float, default=0.001)
    p.add_argument("--weight-decay", type=float, default=0.0001)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = p.parse_args()
    print(json.dumps(main_run(args), indent=2), flush=True)


if __name__ == "__main__":
    main()
