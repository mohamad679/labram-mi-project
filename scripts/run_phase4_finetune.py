"""Phase 4: compare pinned pretrained LaBraM training regimes on frozen validation.

No test waveforms are read. Model selection and early stopping use 14 validation
subjects only. Record independently verifiable, data-provenance-locked evidence.
The 15 designated test subjects remain embargoed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight

from scripts.run_phase3_validation import SHA256_PHASE2, sha256, validate_frozen_split
from src.phase4_trainability import (
    configure_labram_finetuning,
    optimizer_groups,
    set_finetuning_train_mode,
)
from src.portable_eeg import LABELS, H5SignalSource, inspect_cache
from src.subject_statistics import subject_cluster_bootstrap

PINNED_PRETRAINED_REVISION = "0563b6c626e7b40d9a36653b763715db94d945d7"
STRATEGIES = ("probe", "partial", "full")


def _fingerprint_parameters(model, *, trainable: bool | None = None) -> str:
    """Hash name/dtype/shape/actual bytes, including negative (frozen) controls."""
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if trainable is not None and parameter.requires_grad != trainable:
            continue
        value = parameter.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.tobytes())
    return digest.hexdigest()


def validation_metrics(truth: np.ndarray, probs: np.ndarray) -> dict:
    labels = np.arange(5)
    probs = np.asarray(probs, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.int64)
    if probs.shape != (len(truth), 5) or len(truth) != 2436:
        raise ValueError("Expected the complete 2,436-row validation partition")
    if not np.isfinite(probs).all() or np.any(probs < 0) or not np.allclose(probs.sum(1), 1, atol=1e-5):
        raise ValueError("Invalid model class probabilities")
    pred = probs.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(truth, pred)),
        "macro_f1": float(f1_score(truth, pred, labels=labels, average="macro", zero_division=0)),
        "balanced_accuracy": float(np.mean([
            np.mean(pred[truth == cls] == cls) if np.any(truth == cls) else np.nan
            for cls in labels
        ])),
        "confusion_matrix": confusion_matrix(truth, pred, labels=labels).tolist(),
        "class_counts": np.bincount(truth, minlength=5).tolist(),
    }


def make_datasets(source, y, train_rows, val_rows):
    import torch
    from torch.utils.data import Dataset

    class LazyEpochData(Dataset):
        def __init__(self, rows):
            self.rows = np.asarray(rows, dtype=np.int64)

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, idx):
            row = int(self.rows[idx])
            return (torch.from_numpy(np.array(source[row], dtype=np.float32, copy=True)),
                    torch.tensor(int(y[row]), dtype=torch.long))

    return LazyEpochData(train_rows), LazyEpochData(val_rows)


def evaluate(model, loader, device):
    import torch

    model.eval()
    outputs = []
    with torch.inference_mode():
        for xb, _ in loader:
            logits = model(xb.to(device))
            if logits.ndim != 2 or logits.shape[1] != 5 or not torch.isfinite(logits).all():
                raise RuntimeError("Non-finite logits or wrong LaBraM classifier dimension")
            outputs.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
    return np.concatenate(outputs, axis=0)


def _safe_atomic_json(path: Path, data: dict) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_suffix(path.suffix + ".partial")
    if staging.exists():
        raise FileExistsError(f"Interrupted output: {staging}")
    staging.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    staging.replace(path)


def run_strategy(*, strategy, source, y, subjects, train_rows, val_rows,
                 args, output_root) -> dict:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    from src.model import load_labram_model

    if strategy not in STRATEGIES:
        raise ValueError(f"Invalid strategy: {strategy}")
    run_dir = output_root / strategy
    if run_dir.exists():
        raise FileExistsError(f"Refusing to overwrite benchmark outputs: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=False)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train_data, val_data = make_datasets(source, y, train_rows, val_rows)
    tr = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=0)
    va = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=0)

    started = time.monotonic()
    model = load_labram_model(
        n_outputs=5, n_times=400, n_chans=64, sfreq=160.0,
        pretrained_repo_id="braindecode/labram-pretrained",
        pretrained_revision=PINNED_PRETRAINED_REVISION,
    )
    trainability = configure_labram_finetuning(
        model, strategy=strategy, n_outputs=5, last_n_blocks=args.last_n_blocks,
    )
    frozen_pre = _fingerprint_parameters(model, trainable=False)
    # Sentinel provides evidence that all modes loaded the same public backbone.
    first = model.get_submodule("blocks")[0]
    first_block_hash = hashlib.sha256(b"".join(
        p.detach().cpu().numpy().tobytes() for p in first.parameters()
    )).hexdigest()
    model.to(args.device)

    class_weight = compute_class_weight(
        class_weight="balanced", classes=np.arange(5), y=y[train_rows],
    )
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(
        class_weight, dtype=torch.float32, device=args.device,
    ))
    backbone_lr = args.partial_lr if strategy == "partial" else args.full_lr
    groups = optimizer_groups(model, head_lr=args.head_lr, backbone_lr=backbone_lr)
    opt = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    best_f1 = -1.0
    best_epoch = None
    no_improve = 0
    history = []
    checkpoint = run_dir / "selected_weights.pt"
    val_probs_best = None
    for epoch in range(1, args.epochs + 1):
        set_finetuning_train_mode(model, strategy, last_n_blocks=args.last_n_blocks)
        optimizer_steps = 0
        train_loss = 0.0
        opt.zero_grad(set_to_none=True)
        n_batches = len(tr)
        for batch_index, (xb, yb) in enumerate(tr):
            xb, yb = xb.to(args.device), yb.to(args.device)
            logits = model(xb)
            loss = criterion(logits, yb)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite train loss in {strategy}, epoch {epoch}")
            group_start = (batch_index // args.accumulate) * args.accumulate
            denom = min(args.accumulate, n_batches - group_start)
            (loss / denom).backward()
            train_loss += loss.detach().item() * len(yb)
            if (batch_index + 1) % args.accumulate == 0 or batch_index + 1 == n_batches:
                nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], max_norm=args.grad_clip,
                    error_if_nonfinite=True,
                )
                opt.step()
                opt.zero_grad(set_to_none=True)
                optimizer_steps += 1
            if (batch_index + 1) % 120 == 0 or batch_index + 1 == n_batches:
                print(f"{strategy}: epoch={epoch} batch={batch_index+1}/{n_batches} steps={optimizer_steps}", flush=True)

        probs = evaluate(model, va, args.device)
        scores = validation_metrics(y[val_rows], probs)
        history.append({"epoch": epoch, "train_loss": round(train_loss / len(tr.dataset), 7),
                        "optimizer_steps": optimizer_steps, "validation_macro_f1": scores["macro_f1"],
                        "validation_balanced_accuracy": scores["balanced_accuracy"]})
        print(f"{strategy}: epoch={epoch} validation_macro_f1={scores['macro_f1']:.5f} "
              f"validation_balanced_accuracy={scores['balanced_accuracy']:.5f}", flush=True)
        if scores["macro_f1"] > best_f1 + 1e-12:
            best_f1 = scores["macro_f1"]
            best_epoch = epoch
            val_probs_best = probs.copy()
            no_improve = 0
            if checkpoint.exists():
                checkpoint.unlink()
            torch.save({key: tensor.detach().cpu() for key, tensor in model.state_dict().items()}, checkpoint)
        else:
            no_improve += 1
        if no_improve >= args.patience:
            break

    if val_probs_best is None or best_epoch is None or not checkpoint.is_file():
        raise RuntimeError("No validation-selected checkpoint")
    saved_sha = sha256(checkpoint)
    weights = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(weights, strict=True)
    reload_probs = evaluate(model, va, args.device)
    difference = float(np.max(np.abs(reload_probs - val_probs_best)))
    if not np.allclose(reload_probs, val_probs_best, rtol=1e-5, atol=1e-6):
        raise RuntimeError(f"Restored validation probabilities changed: {difference}")
    frozen_post = _fingerprint_parameters(model, trainable=False)
    if strategy != "full" and frozen_pre != frozen_post:
        raise RuntimeError("Frozen pretrained backbone changed during optimization")
    preds_path = run_dir / "validation_predictions.npz"
    if preds_path.exists():
        raise FileExistsError("Cannot overwrite validation predictions")
    np.savez_compressed(preds_path, y_true=y[val_rows], y_prob=reload_probs,
                        subject_ids=subjects[val_rows], class_names=np.asarray(LABELS))
    report = {
        "status": "PASSED_VALIDATION_RELOAD_GATE", "strategy": strategy,
        "dataset_sha256": SHA256_PHASE2, "split_seed": 42, "model_seed": args.seed,
        "validation_only": True, "test_predictions_computed": False,
        "train_subjects": 75, "validation_subjects": 14, "test_subjects_embargoed": 15,
        "validation_epochs": len(val_rows), "best_epoch": best_epoch,
        "parameters": trainability, "training_history": history,
        "pretrained_revision": PINNED_PRETRAINED_REVISION,
        "original_first_transformer_block_sha256": first_block_hash,
        "frozen_parameters_sha256_before": frozen_pre,
        "frozen_parameters_sha256_after": frozen_post,
        "frozen_parameters_unchanged": strategy == "full" or frozen_pre == frozen_post,
        "checkpoint_file": str(checkpoint.relative_to(output_root)),
        "checkpoint_sha256": saved_sha,
        "checkpoint_reloaded": True, "max_abs_probability_reload_error": difference,
        "validation_metrics": validation_metrics(y[val_rows], reload_probs),
        "macro_f1_subject_bootstrap": subject_cluster_bootstrap(
            y[val_rows], reload_probs, subjects[val_rows], n_bootstraps=args.bootstrap,
            seed=args.seed,
        )["macro_f1"],
        "optimizer": {
            "head_lr": args.head_lr,
            "backbone_lr": None if strategy == "probe" else backbone_lr,
            "weight_decay": args.weight_decay,
            "accumulation": args.accumulate,
            "microbatch": args.batch_size,
            "epochs_limit": args.epochs,
            "patience": args.patience,
            "grad_clip": args.grad_clip,
        },
        "elapsed_seconds": round(time.monotonic() - started, 2),
    }
    _safe_atomic_json(run_dir / "report.json", report)
    return report


def run(args):
    if args.device not in ("cuda", "cpu"):
        raise ValueError("Device must be cuda or cpu")
    if args.epochs < 1 or args.patience < 1 or args.bootstrap < 10 or args.seed < 0:
        raise ValueError("Invalid epoch/patience/bootstrap/seed")
    if args.batch_size < 1 or args.accumulate < 1 or args.grad_clip <= 0:
        raise ValueError("Invalid batch size/gradient accumulation or clipping")
    if any(lr <= 0 for lr in (args.head_lr, args.partial_lr, args.full_lr)):
        raise ValueError("Learning rates must be positive")
    cache = Path(args.cache).resolve()
    if not cache.is_file() or sha256(cache) != SHA256_PHASE2:
        raise ValueError("Missing or SHA-mismatched immutable Phase-2 HDF5")
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError("Benchmark output already exists; use a new directory")
    output.mkdir(parents=True, exist_ok=False)
    import torch

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA accelerator requested but not available")
    labels, subjects, shape = inspect_cache(cache, require_complete=True)
    if shape != (18096, 64, 400) or tuple(np.unique(labels)) != LABELS:
        raise ValueError("Unexpected EEG data dimensions or label order")
    y = np.searchsorted(np.asarray(LABELS), labels)
    with Path(args.split_manifest).open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    split = validate_frozen_split(y, subjects, manifest)
    if any(len(np.unique(subjects[split[k]])) != n for k, n in (("train",75),("validation",14),("test",15))):
        raise ValueError("Invalid split subject counts")
    print("Validation-only: test subject waveform access prohibited", flush=True)
    source = H5SignalSource(cache, shape)
    reports = []
    try:
        for strategy in args.strategies:
            reports.append(run_strategy(
                strategy=strategy, source=source, y=y, subjects=subjects,
                train_rows=split["train"], val_rows=split["validation"],
                args=args, output_root=output,
            ))
    finally:
        source.close()
    if len({r["original_first_transformer_block_sha256"] for r in reports}) != 1:
        raise RuntimeError("Pretrained initialization differs across training regimes")
    summary = {
        "phase": 4, "status": "VALIDATION_COMPLETE_NOT_A_HELDOUT_RESULT",
        "class_order": list(LABELS), "dataset_sha256": SHA256_PHASE2,
        "pretrained_revision": PINNED_PRETRAINED_REVISION,
        "strategies_completed": list(args.strategies),
        "split_subject_counts": {"train":75,"validation":14,"test_locked":15},
        "split_epoch_counts": {"train":13050,"validation":2436,"test_locked":2610},
        "models": {r["strategy"]: {"macro_f1":r["validation_metrics"]["macro_f1"],
                                 "balanced_accuracy":r["validation_metrics"]["balanced_accuracy"],
                                 "checkpoint_sha256":r["checkpoint_sha256"],
                                 "best_epoch":r["best_epoch"]} for r in reports},
        "source_code_sha256": sha256(Path(__file__).resolve()),
        "environment": {"python":platform.python_version(), "torch":torch.__version__,
                        "numpy":np.__version__, "device":args.device},
        "test_access": "FORBIDDEN (metadata partition assertions only)",
    }
    _safe_atomic_json(output / "phase4_summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--split-manifest", type=Path, default=Path("configs/phase3_locked_split.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/phase4_finetune"))
    parser.add_argument("--strategies", nargs="+", choices=STRATEGIES, default=list(STRATEGIES))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--accumulate", type=int, default=2)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--partial-lr", type=float, default=1e-4)
    parser.add_argument("--full-lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--last-n-blocks", type=int, default=2)
    parser.add_argument("--bootstrap", type=int, default=250)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    summary = run(args)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
