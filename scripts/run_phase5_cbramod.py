"""Phase 5 validation-only pretrained CBraMod baseline; held-out Test is embargoed."""
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
from src.phase5_cbramod import (
    CHECKPOINT_REPO,
    CHECKPOINT_REVISION,
    CHECKPOINT_SHA256,
    adapt_cbramod_input,
    load_pinned_cbramod,
)
from src.portable_eeg import LABELS, H5SignalSource, inspect_cache
from src.subject_statistics import subject_cluster_bootstrap

MODES = ("probe", "full")


def validate_probs(y, probs):
    probs = np.asarray(probs)
    if len(y) != 2436 or probs.shape != (2436, 5):
        raise ValueError("Validation predictions must be exactly 2436 x 5")
    if not np.isfinite(probs).all() or np.min(probs) < 0 or not np.allclose(
        np.sum(probs, axis=1), 1., atol=1e-5,
    ):
        raise ValueError("Invalid five-class probability matrix")
    y = np.asarray(y, dtype=np.int64)
    if not np.array_equal(np.unique(y), np.arange(5)):
        raise ValueError("Validation five-class vocabulary mismatch")
    pred = probs.argmax(axis=1)
    recall = [float(np.mean(pred[y == c] == c)) for c in range(5)]
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, labels=list(range(5)),
                                   average="macro", zero_division=0)),
        "balanced_accuracy": float(np.mean(recall)),
        "per_class_recall": recall,
        "confusion_matrix": confusion_matrix(y, pred, labels=list(range(5))).tolist(),
    }


def parameter_digest(model, *, backbone_only=True):
    h = hashlib.sha256()
    for name, p in model.named_parameters():
        if backbone_only and not name.startswith("backbone."):
            continue
        h.update(name.encode("utf-8"))
        h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def load_checkpoint_from_hf():
    from huggingface_hub import hf_hub_download

    file = Path(hf_hub_download(repo_id=CHECKPOINT_REPO,
                                revision=CHECKPOINT_REVISION,
                                filename="model.safetensors"))
    if sha256(file) != CHECKPOINT_SHA256:
        raise RuntimeError("Downloaded safetensors do not match pinned SHA-256")
    return file


def run_mode(*, mode, source, y, subjects, train_ids, val_ids, args,
             output_root, checkpoint_file):
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset

    if mode not in MODES:
        raise ValueError("Unknown predeclared mode")
    dest = output_root / mode
    dest.mkdir(parents=True, exist_ok=False)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    class H5Epochs(Dataset):
        def __init__(self, ids):
            self.ids = np.asarray(ids, dtype=np.int64)

        def __len__(self):
            return len(self.ids)

        def __getitem__(self, idx):
            ix = int(self.ids[idx])
            x = adapt_cbramod_input(source[ix])
            return torch.from_numpy(x), torch.tensor(int(y[ix]), dtype=torch.long)

    train_loader = DataLoader(H5Epochs(train_ids), batch_size=args.batch_size,
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(H5Epochs(val_ids), batch_size=args.batch_size,
                            shuffle=False, num_workers=0)
    model, checkpoint_info = load_pinned_cbramod(checkpoint_file, device=args.device)
    if mode == "probe":
        for p in model.backbone.parameters():
            p.requires_grad_(False)
    else:
        for p in model.backbone.parameters():
            p.requires_grad_(True)
    frozen_start = parameter_digest(model)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    if mode == "full" and n_trainable != n_total:
        raise RuntimeError("Full fine-tuning unexpectedly froze parameters")
    if mode == "probe" and not (0 < n_trainable < n_total):
        raise RuntimeError("Probe has wrong trainable parameter count")
    groups = [{"params": list(model.head.parameters()), "lr": args.head_lr}]
    if mode == "full":
        groups.append({"params": list(model.backbone.parameters()), "lr": args.full_lr})
    opt = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    weights = compute_class_weight("balanced", classes=np.arange(5), y=y[train_ids])
    objective = nn.CrossEntropyLoss(weight=torch.as_tensor(
        weights, dtype=torch.float32, device=args.device,
    ))
    amp = bool(args.amp and args.device == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    def eval_validation():
        model.eval()
        items = []
        with torch.inference_mode():
            for xb, _ in val_loader:
                xb = xb.to(args.device)
                with torch.autocast(device_type=args.device, enabled=amp):
                    logits = model(xb)
                if logits.shape[1:] != (5,) or not torch.isfinite(logits).all():
                    raise RuntimeError("Invalid five-class logits")
                items.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
        return np.concatenate(items)

    started = time.monotonic()
    selected_score = -1.
    best_epoch = 0
    best_probs = None
    stale = 0
    history = []
    state_path = dest / "selected_weights.pt"
    saw_backbone_gradient = False
    for epoch in range(1, args.epochs + 1):
        if mode == "probe":
            model.eval()  # hold pretrained dropout/normalization fixed
            model.head.train()
        else:
            model.train()
        total_loss = 0.
        opt.zero_grad(set_to_none=True)
        for j, (xb, yb) in enumerate(train_loader):
            xb = xb.to(args.device)
            yb = yb.to(args.device)
            with torch.autocast(device_type=args.device, enabled=amp):
                logits = model(xb)
                loss = objective(logits.float(), yb)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite train loss")
            scaler.scale(loss).backward()
            if j == 0:
                grads = [p.grad for p in model.backbone.parameters() if p.requires_grad]
                if mode == "full":
                    saw_backbone_gradient = any(
                        g is not None and torch.isfinite(g).all() and torch.any(g != 0)
                        for g in grads
                    )
                    if not saw_backbone_gradient:
                        raise RuntimeError("Full fine-tune backbone has no nonzero gradients")
                elif any(p.grad is not None for p in model.backbone.parameters()):
                    raise RuntimeError("Frozen backbone received a gradient")
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                           max_norm=args.grad_clip, error_if_nonfinite=True)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            total_loss += float(loss.detach().item()) * len(yb)
            if (j + 1) % 200 == 0 or j + 1 == len(train_loader):
                print(f"{mode} epoch {epoch}: batch {j+1}/{len(train_loader)}", flush=True)
        probs = eval_validation()
        scores = validate_probs(y[val_ids], probs)
        history.append({"epoch":epoch, "train_loss": total_loss / len(train_ids),
                        "validation_macro_f1":scores["macro_f1"],
                        "validation_balanced_accuracy":scores["balanced_accuracy"]})
        print(f"{mode} epoch {epoch}: val macro-F1={scores['macro_f1']:.5f}",flush=True)
        if scores["macro_f1"] > selected_score + 1e-12:
            selected_score = scores["macro_f1"]
            best_epoch = epoch
            best_probs = probs.copy()
            torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()},
                       state_path.with_suffix(".partial"))
            state_path.with_suffix(".partial").replace(state_path)
            stale = 0
        else:
            stale += 1
        if stale >= args.patience:
            break
    if best_epoch < 1 or best_probs is None:
        raise RuntimeError("No selected model checkpoint")
    selected_sha = sha256(state_path)
    loaded = torch.load(state_path, weights_only=True, map_location="cpu")
    model.load_state_dict(loaded, strict=True)
    loaded_probs = eval_validation()
    max_abs_diff = float(np.max(np.abs(loaded_probs - best_probs)))
    if not np.allclose(loaded_probs, best_probs, rtol=1e-5, atol=1e-6):
        raise RuntimeError("Checkpoint reload did not reproduce selected probabilities")
    frozen_end = parameter_digest(model)
    if mode == "probe" and frozen_start != frozen_end:
        raise RuntimeError("Frozen CBraMod weights changed")
    if mode == "full" and frozen_start == frozen_end:
        raise RuntimeError("Full fine-tuning did not change pretrained backbone weights")
    np.savez_compressed(dest / "validation_predictions.npz",
                        y_true=y[val_ids], y_prob=loaded_probs,
                        subject_ids=subjects[val_ids], class_names=np.asarray(LABELS))
    report={
        "model":"CBraMod", "status":"VALIDATION_ONLY_CHECKPOINT_RELOADED", "mode":mode,
        "seed":args.seed, "dataset_sha256":SHA256_PHASE2, "pretrained":checkpoint_info,
        "adapter":{"input_unit":"uV / 100", "resample":"160->200 Hz scipy polyphase 5:4",
                   "actual_signal_samples":500,"zero_pad_samples":100,"input_shape":[64,600]},
        "train_subjects":75,"validation_subjects":14,"test_subjects_embargoed":15,
        "train_epochs":len(train_ids), "validation_epochs":len(val_ids),
        "trainable_parameters":n_trainable,"total_parameters":n_total,
        "full_backbone_nonzero_gradient_seen":saw_backbone_gradient if mode=="full" else None,
        "backbone_hash_before":frozen_start,"backbone_hash_after":frozen_end,
        "best_epoch":best_epoch,"history":history,
        "checkpoint_sha256":selected_sha,"checkpoint_reloaded":True,
        "max_abs_probability_reload_difference":max_abs_diff,
        "scores":validate_probs(y[val_ids],loaded_probs),
        "macro_f1_subject_bootstrap":subject_cluster_bootstrap(
            y[val_ids], loaded_probs, subjects[val_ids], n_bootstraps=args.bootstrap,
            seed=args.seed,
        )["macro_f1"],
        "wall_seconds":round(time.monotonic()-started,2),
    }
    (dest / "report.json").write_text(json.dumps(report,indent=2)+"\n")
    return report


def run(args):
    if args.seed != 42 or args.device not in ("cpu","cuda"):
        raise ValueError("Only prespecified seed 42 and explicit device allowed")
    if args.epochs < 1 or args.patience < 1 or args.batch_size < 1 or args.bootstrap < 10:
        raise ValueError("Invalid run budget")
    if any(v <= 0 for v in (args.head_lr,args.full_lr,args.grad_clip)):
        raise ValueError("Invalid optimization controls")
    if args.output_dir.exists():
        raise FileExistsError("Refusing overwrite: new output directory required")
    if sha256(args.cache) != SHA256_PHASE2:
        raise RuntimeError("Phase-2 HDF5 SHA mismatch")
    labels, subjects, shape = inspect_cache(args.cache, require_complete=True)
    if shape != (18096,64,400) or tuple(np.unique(labels)) != LABELS:
        raise RuntimeError("Invalid Phase-2 cohort schema")
    y = np.searchsorted(np.asarray(LABELS),labels)
    manifest = json.loads(Path(args.split_manifest).read_text())
    indices=validate_frozen_split(y,subjects,manifest)
    print("TEST LOCKED: no waveform inference on 15 test subjects",flush=True)
    import braindecode
    import torch
    from huggingface_hub import __version__ as hf_version
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU; do not silently fall back to CPU")
    checkpoint=load_checkpoint_from_hf()
    args.output_dir.mkdir(parents=True,exist_ok=False)
    source = H5SignalSource(args.cache,shape)
    reports=[]
    try:
        for mode in args.modes:
            reports.append(run_mode(mode=mode,source=source,y=y,subjects=subjects,
                                    train_ids=indices["train"],val_ids=indices["validation"],
                                    args=args, output_root=args.output_dir,
                                    checkpoint_file=checkpoint))
    finally:
        source.close()
    if len({r["backbone_hash_before"] for r in reports})!=1:
        raise RuntimeError("Pretrained initialization differs between modes")
    summary={"phase":5,"status":"EXPLORATORY_VALIDATION_ONLY_NOT_SUPERVISOR_PASS",
             "dataset_sha256":SHA256_PHASE2,
             "pretrained_checkpoint_sha256":CHECKPOINT_SHA256,
             "runner_sha256":sha256(Path(__file__).resolve()),
             "train_subjects":75,"validation_subjects":14,"test_subjects_locked":15,
             "validation_epochs":2436,
             "models":{r["mode"]:{"macro_f1":r["scores"]["macro_f1"],
                                  "balanced_accuracy":r["scores"]["balanced_accuracy"],
                                  "selected_epoch":r["best_epoch"],
                                  "checkpoint_sha256":r["checkpoint_sha256"]}
                       for r in reports},
             "environment":{"python":platform.python_version(),
                            "torch":torch.__version__,"braindecode":braindecode.__version__,
                            "huggingface_hub":hf_version,"numpy":np.__version__,
                            "device":args.device},
             "test_waveform_inference":False}
    (args.output_dir / "phase5_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache",type=Path,required=True)
    p.add_argument("--split-manifest",type=Path,default=Path("configs/phase3_locked_split.json"))
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--modes",nargs="+",choices=MODES,default=list(MODES))
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--device",choices=("cuda","cpu"),default="cuda")
    p.add_argument("--epochs",type=int,default=10)
    p.add_argument("--patience",type=int,default=3)
    p.add_argument("--batch-size",type=int,default=8)
    p.add_argument("--head-lr",type=float,default=1e-3)
    p.add_argument("--full-lr",type=float,default=3e-5)
    p.add_argument("--weight-decay",type=float,default=1e-4)
    p.add_argument("--grad-clip",type=float,default=1.0)
    p.add_argument("--amp",action="store_true")
    p.add_argument("--bootstrap",type=int,default=500)
    args=p.parse_args()
    print(json.dumps(run(args),indent=2),flush=True)


if __name__=="__main__":
    main()
