"""Phase 3: frozen-cohort, validation-only EEG baselines.

No held-out test waveforms or test labels are read for model selection.
Model comparisons and early stopping use validation participants only.
CPU: Dummy, Riemannian tangent-space regression. GPU: EEGNet, LaBraM probe.
All EEG signals stay in the local read-only HDF5, never in artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler

from src.data import subject_wise_split
from src.portable_eeg import LABELS, H5SignalSource, inspect_cache
from src.subject_statistics import subject_cluster_bootstrap

SHA256_PHASE2 = "f6d2c7824f6aaa522286ea8ee5253b0220602117fc69d7b49c3af6e6a1324f7a"
DEFAULT_SPLIT = Path("configs/phase3_locked_split.json")
MODELS = ("dummy", "riemann", "eegnet", "labram_probe")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_frozen_split(y: np.ndarray, subject_ids: np.ndarray, manifest: dict):
    if len(y) != 18096 or set(np.unique(subject_ids)) != {
        i for i in range(1, 110) if i not in (88, 89, 92, 100, 104)
    }:
        raise ValueError("Only the 104-subject Phase-2 cohort is allowed.")
    if manifest["dataset_sha256"] != SHA256_PHASE2 or manifest["split_seed"] != 42:
        raise ValueError("Frozen dataset hash or split seed unexpectedly changed.")
    index = subject_wise_split(
        np.arange(len(y)), y, subject_ids,
        test_size_subjects=15, val_size_subjects=14, random_state=42,
    )
    partition = {}
    for key, ids in zip(("train", "validation", "test"), index):
        actual = sorted(map(int, set(subject_ids[ids])))
        expected = manifest["partitions"][key]
        if actual != expected["subject_ids"] or len(ids) != expected["epochs_count"]:
            raise ValueError(f"Frozen {key} split differs from Phase-2 executed manifest.")
        partition[key] = np.asarray(ids, dtype=np.int64)
    all_indices = np.concatenate(list(partition.values()))
    if not np.array_equal(np.sort(all_indices), np.arange(len(y))):
        raise ValueError("Split partitions are not exhaustive or contain overlap.")
    return partition


def _features_from_covariances(source, indices, batch_size=32):
    """Stream only designated participant epochs to OAS covariances."""
    from pyriemann.estimation import Covariances

    estimator = Covariances(estimator="oas")
    cov_list = []
    for offset in range(0, len(indices), batch_size):
        batch = np.stack([source[int(i)] for i in indices[offset:offset + batch_size]])
        if not np.all(np.isfinite(batch)):
            raise ValueError("Non-finite training or validation EEG sample.")
        cov = estimator.fit_transform(batch)
        if not np.all(np.isfinite(cov)):
            raise ValueError("Non-finite OAS covariance matrix.")
        cov_list.append(np.asarray(cov, dtype=np.float64))
    return np.concatenate(cov_list)


def _riemann(source, y, train_idx, val_idx, seed, batch_size):
    from pyriemann.tangentspace import TangentSpace

    t0 = time.monotonic()
    print("Phase 3 Riemannian: train OAS covariances (streaming)", flush=True)
    train_cov = _features_from_covariances(source, train_idx, batch_size)
    print("Phase 3 Riemannian: validation covariances", flush=True)
    val_cov = _features_from_covariances(source, val_idx, batch_size)
    tangent = TangentSpace(metric="riemann")
    X_train = tangent.fit_transform(train_cov)  # training participants ONLY
    X_val = tangent.transform(val_cov)
    scaler = StandardScaler()  # MUST fit on training epochs only
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    classifier = LogisticRegression(
        max_iter=3000, tol=1e-4, class_weight="balanced",
        random_state=seed, solver="lbfgs",
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=ConvergenceWarning)
        classifier.fit(X_train, y[train_idx])
    if int(np.max(classifier.n_iter_)) >= classifier.max_iter:
        raise RuntimeError("Riemannian classifier reached iteration limit")
    if not np.array_equal(classifier.classes_, np.arange(5)):
        raise ValueError("Incomplete Riemannian class vocabulary.")
    probs = classifier.predict_proba(X_val)
    return probs, {"algorithm": "OAS + Riemannian TangentSpace + class-balanced LR",
                   "tangent_fitted_on_train_only": True, "scaler_fitted_on_train_only": True, "solver": "lbfgs",
                   "solver_max_iter": classifier.max_iter, "solver_iterations": classifier.n_iter_.tolist(),
                   "solver_converged": True,
                   "train_cov_shape": list(train_cov.shape),
                   "runtime_seconds": round(time.monotonic() - t0, 2)}


def _dummy(y, train_idx, val_idx):
    model = DummyClassifier(strategy="prior")
    model.fit(np.zeros((len(train_idx), 1)), y[train_idx])
    if not np.array_equal(model.classes_, np.arange(5)):
        raise ValueError("Dummy class vocabulary incomplete.")
    return model.predict_proba(np.zeros((len(val_idx), 1))), {
        "strategy": "prior", "training_class_counts": np.bincount(y[train_idx], minlength=5).tolist(),
        "labels_from_train_only": True,
    }


def _deep(source, y, train_idx, val_idx, seed, *, model_name, epochs, patience,
          batch_size, lr, weight_decay, device, checkpoint_path):
    import torch
    from sklearn.utils.class_weight import compute_class_weight
    from torch import nn
    from torch.utils.data import DataLoader, Dataset

    class EpochData(Dataset):
        def __init__(self, rows):
            self.rows = np.asarray(rows, dtype=np.int64)

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, index):
            row = int(self.rows[index])
            return (torch.from_numpy(np.array(source[row], dtype=np.float32, copy=True)),
                    torch.tensor(int(y[row]), dtype=torch.long))

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    from braindecode.models import EEGNet
    if model_name == "eegnet":
        model = EEGNet(n_chans=64, n_outputs=5, n_times=400, sfreq=160.)
        configuration = {"architecture": "braindecode.EEGNet", "pretrained": False}
    elif model_name == "labram_probe":
        from src.model import load_labram_model
        from src.training_modes import configure_trainable_parameters
        model = load_labram_model(
            n_outputs=5, n_times=400, n_chans=64, sfreq=160.,
            pretrained_repo_id="braindecode/labram-pretrained",
            pretrained_revision="0563b6c626e7b40d9a36653b763715db94d945d7",
        )
        configuration = {
            "architecture": "LaBraM 400-sample pretrained prefix adapter",
            "pretrained": True,
            "pretrained_revision": "0563b6c626e7b40d9a36653b763715db94d945d7",
            "parameter_audit": configure_trainable_parameters(
                model, strategy="linear_probe", n_outputs=5,
            ),
        }
    else:
        raise ValueError(model_name)
    model = model.to(device)
    tr = DataLoader(EpochData(train_idx), batch_size=batch_size, shuffle=True, num_workers=0)
    va = DataLoader(EpochData(val_idx), batch_size=batch_size, shuffle=False, num_workers=0)
    weights = compute_class_weight("balanced", classes=np.arange(5), y=y[train_idx])
    criterion = nn.CrossEntropyLoss(weight=torch.as_tensor(weights, device=device, dtype=torch.float32))
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=lr, weight_decay=weight_decay)
    history = []
    best_f1 = -1.0
    best_probs = None
    best_weights = None
    best_epoch = 0
    stale = 0
    for epoch in range(1, epochs + 1):
        if model_name == "labram_probe":
            model.eval()  # frozen feature extractor must not change dropout/BN state
            model.get_submodule("final_layer").train()
        else:
            model.train()
        total_loss = 0.
        for xb, yb in tr:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite {model_name} training loss")
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu().item()) * len(xb)
        model.eval()
        probabilities = []
        with torch.no_grad():
            for xb, _ in va:
                logits = model(xb.to(device))
                probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
        probs = np.concatenate(probabilities)
        metric = score_validation(y[val_idx], probs)["macro_f1"]
        history.append({"epoch": epoch, "train_loss": total_loss / len(tr.dataset),
                        "validation_macro_f1": metric})
        print(f"{model_name} epoch={epoch} train_loss={history[-1]['train_loss']:.5f} "
              f"validation_macro_f1={metric:.5f}", flush=True)
        if metric > best_f1 + 1e-12:
            best_f1 = metric
            best_epoch = epoch
            best_probs = probs.copy()
            best_weights = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    if best_probs is None or best_weights is None:
        raise RuntimeError(f"No valid {model_name} validation checkpoint.")
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.exists():
        raise FileExistsError("Refusing to overwrite selected model weights")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=False)
    import torch
    torch.save(best_weights, checkpoint_path)
    restored = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(restored, strict=True)
    model.eval()
    verified_probs = []
    with torch.inference_mode():
        for xb, _ in va:
            verified_probs.append(torch.softmax(model(xb.to(device)), dim=1).cpu().numpy())
    verified_probs = np.concatenate(verified_probs)
    if not np.allclose(verified_probs, best_probs, rtol=1e-5, atol=1e-6):
        raise RuntimeError("Persisted best model predictions mismatch selected validation epoch")
    checkpoint_digest = sha256(checkpoint_path)
    # No test predictions, even after validation-only checkpoint selection.
    configuration.update({"device": str(device), "best_epoch": best_epoch,
                          "epochs_completed": len(history), "history": history,
                          "learning_rate": lr, "weight_decay": weight_decay,
                          "class_weights": weights.tolist(),
                          "best_state_dict_saved": True,
                          "selected_checkpoint_sha256": checkpoint_digest,
                          "selected_checkpoint_bytes": checkpoint_path.stat().st_size,
                          "checkpoint_reload_validation_predictions_match": True})
    return best_probs, configuration


def score_validation(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    if y_prob.shape != (len(y_true), 5) or not np.all(np.isfinite(y_prob)):
        raise ValueError("Invalid five-class validation predictions")
    if np.any(y_prob < 0) or not np.allclose(y_prob.sum(1), 1., atol=1e-5):
        raise ValueError("Invalid probability rows")
    if set(y_true) != set(range(5)):
        raise ValueError("All five classes must occur in validation.")
    predictions = y_prob.argmax(1)
    recalls = [(predictions[y_true == i] == i).mean() for i in range(5)]
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "macro_f1": float(f1_score(y_true, predictions, labels=np.arange(5),
                                   average="macro", zero_division=0)),
        "balanced_accuracy": float(np.mean(recalls)),
        "per_class_recall": [float(x) for x in recalls],
        "confusion_matrix": confusion_matrix(y_true, predictions,
                                               labels=np.arange(5)).tolist(),
    }


def _dataset_manifest(path, data_sha, args, partitions, y, subjects) -> dict:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown: source supplied independently"
    return {
        "source_commit": commit,
        "cache_basename": path.name,
        "cache_sha256": data_sha,
        "split_seed": 42,
        "random_seed": args.seed,
        "subjects": {key: sorted(map(int, set(subjects[rows]))) for key, rows in partitions.items()},
        "epoch_counts": {key: len(rows) for key, rows in partitions.items()},
        "class_names": list(LABELS),
        "preprocessing": "PhysioNet EEGMMIDB 1.0.0 / MOABB 1.7.2 / 8-32Hz / 160Hz / uV / first 400 samples",
        "validation_class_counts": np.bincount(y[partitions["validation"]], minlength=5).tolist(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "device": args.device,
        "evaluation_scope": "validation_only_no_heldout_test_labels_used",
    }


def run(args):
    path = Path(args.cache).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Attach the validated Phase 2 HDF5 to Kaggle: {path}")
    if args.seed < 0 or args.batch_size < 1 or args.epochs < 1 or args.patience < 1:
        raise ValueError("Invalid training configuration")
    models = args.models
    if not models or any(m not in MODELS for m in models) or len(set(models)) != len(models):
        raise ValueError("Models must be a unique non-empty subset of known baselines.")
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite Phase 3 evidence: {args.output_dir}")
    data_sha = sha256(path)
    if data_sha != SHA256_PHASE2:
        raise ValueError("Refusing modified or unverifiable Phase 2 EEG HDF5 checksum.")
    labels, subjects, shape = inspect_cache(path, require_complete=True)
    if tuple(shape) != (18096, 64, 400):
        raise ValueError(f"HDF5 schema shape changed: {shape}")
    if set(np.unique(labels)) != set(LABELS):
        raise ValueError("Unexpected label classes or order.")
    vocabulary = {label: index for index, label in enumerate(LABELS)}
    y = np.fromiter((vocabulary[label] for label in labels), dtype=np.int64, count=len(labels))
    manifest = json.loads(Path(args.split_manifest).read_text(encoding="utf-8"))
    partitions = validate_frozen_split(y, subjects, manifest)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    root = _dataset_manifest(path, data_sha, args, partitions, y, subjects)
    root["split_manifest_sha256"] = sha256(Path(args.split_manifest))
    root["models_requested"] = models
    (args.output_dir / "protocol.json").write_text(json.dumps(root, indent=2) + "\n")
    src = H5SignalSource(path, shape)
    try:
        for model in models:
            start = time.monotonic()
            print(f"[Phase 3] RUN {model}; validation-only; test locked", flush=True)
            if model == "dummy":
                probabilities, hyper = _dummy(y, partitions["train"], partitions["validation"])
            elif model == "riemann":
                probabilities, hyper = _riemann(src, y, partitions["train"],
                                                 partitions["validation"], args.seed,
                                                 args.batch_size)
            else:
                probabilities, hyper = _deep(src, y, partitions["train"],
                                            partitions["validation"], args.seed,
                                            model_name=model, epochs=args.epochs,
                                            patience=args.patience, batch_size=args.batch_size,
                                            lr=args.learning_rate, weight_decay=args.weight_decay,
                                            device=args.device,
                                            checkpoint_path=args.output_dir / model / "best_weights.pt")
            metrics = score_validation(y[partitions["validation"]], probabilities)
            stats = subject_cluster_bootstrap(
                y[partitions["validation"]], probabilities,
                subjects[partitions["validation"]], n_bootstraps=args.bootstrap,
                seed=args.seed,
            )
            report = {
                "status": "completed", "model": model,
                "validation_metrics": metrics,
                "validation_subject_cluster_intervals": stats,
                "configuration": hyper,
                "elapsed_seconds": round(time.monotonic() - start, 1),
                "no_test_predictions_computed": True,
                "protocol_sha256": sha256(args.output_dir / "protocol.json"),
            }
            target = args.output_dir / model
            target.mkdir(exist_ok=True)
            (target / "report.json").write_text(json.dumps(report, indent=2) + "\n")
            np.savez_compressed(target / "validation_predictions.npz",
                                y_true=y[partitions["validation"]],
                                y_prob=probabilities,
                                subject_ids=subjects[partitions["validation"]],
                                class_names=np.asarray(LABELS))
            print(f"[Phase 3] {model} val Macro-F1={metrics['macro_f1']:.4f}; "
                  f"BalancedAcc={metrics['balanced_accuracy']:.4f}", flush=True)
    finally:
        src.close()
    existing = sorted(p.name for p in args.output_dir.iterdir() if p.is_dir())
    all_done = sorted(models) == existing
    summary = {"status": "completed" if all_done else "partial",
               "models_completed": existing, "models_requested": models,
               "evaluation_scope": "validation_only", "test_evaluations": 0,
               "dataset_sha256": data_sha}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if not all_done:
        raise RuntimeError("Missing a requested model.")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/phase3_validation"))
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--bootstrap", type=int, default=250)
    parser.add_argument("--device", default="cuda" if __import__("importlib").util.find_spec("torch") and __import__("torch").cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.bootstrap < 10 or args.device not in ("cuda", "cpu"):
        parser.error("Bootstrap must be >=10 and device cpu or cuda")
    result = run(args)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
