"""
Phase 1 — Fine-tune LaBraM's classification head on PhysioNet Motor Imagery.

Loads cached (or freshly fetched) data, applies the subject-wise split,
freezes the LaBraM backbone, and trains only the classification head
(linear-probe baseline). Reports accuracy + macro-F1 on val/test.
"""
import os
import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score, classification_report

from src.config import load_training_config
from src.channels import verify_channel_order, verify_sampling_rate
from src.data import load_physionet_mi, subject_wise_split
from src.model import load_labram_model

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CACHE_PATHS = [
    "physionet_mi_cache.npz",  # local cache only; never commit raw EEG data
    "/kaggle/input/physionet-mi-cache/physionet_mi_cache.npz",
    "/kaggle/working/physionet_mi_cache.npz",
]
EXCLUDED_SUBJECTS = (88, 89, 92, 100, 104)  # known bad sampling-rate subjects

BATCH_SIZE = 32
EPOCHS = 30
LR = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = "checkpoints/labram_head_finetuned.pt"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_dataset(config=None):
    config = config or load_training_config()
    candidates = ([os.environ['PHYSIONET_CACHE_PATH']]
                  if os.getenv('PHYSIONET_CACHE_PATH') else []) + CACHE_PATHS
    for path in candidates:
        if os.path.exists(path):
            print(f"Loading cached data from {path}")
            with np.load(path, allow_pickle=False) as data:
                X, y, subject_ids = data["X"], data["y"], data["subject_ids"]
                required = {"channel_names", "sfreq", "input_unit"}
                missing = required - set(data.files)
                if missing:
                    if os.environ.get("ALLOW_UNVERIFIED_EEG_CACHE") != "1":
                        raise ValueError(
                            "Unverified legacy EEG cache is missing provenance "
                            f"{sorted(missing)} at {path}. Rebuild from MOABB or "
                            "explicitly set ALLOW_UNVERIFIED_EEG_CACHE=1 for "
                            "non-publishable exploratory use."
                        )
                    print("WARNING: using legacy EEG cache without verified channel order.")
                else:
                    verify_channel_order(data["channel_names"].tolist())
                    verify_sampling_rate(float(data["sfreq"]))
                    if str(data["input_unit"]) != "V":
                        raise ValueError("Cache unit must be V (MNE EEG volts).")
            keep = ~np.isin(subject_ids, config.excluded_subjects)
            X, y, subject_ids = X[keep], y[keep], subject_ids[keep]
            if X.shape[1:] != (config.n_chans, config.n_times):
                raise ValueError(f"Cache has incompatible EEG shape: {X.shape}")
            return X, y, subject_ids

    print("No cache found anywhere — downloading fresh via MOABB "
          "(one-time, ~1.5h for all subjects).")
    from moabb.datasets import PhysionetMI
    all_subjects = [s for s in PhysionetMI().subject_list if s not in config.excluded_subjects]
    X, y, metadata = load_physionet_mi(subjects=all_subjects)
    subject_ids = metadata["subject"].values

    save_path = os.environ.get("PHYSIONET_CACHE_PATH", CACHE_PATHS[0])
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        save_path, X=X, y=y, subject_ids=subject_ids,
        channel_names=np.array(metadata.attrs["channel_names"]),
        sfreq=np.float64(metadata.attrs["sfreq"]),
        input_unit=np.array(metadata.attrs["input_unit"]),
    )
    print(f"Saved local EEG cache: {save_path}; do NOT commit subject data to Git.")

    return X, y, subject_ids


class EEGDataset(Dataset):
    def __init__(self, X, y, indices):
        self.X = X[indices]
        self.y = y[indices]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32)
        y = torch.tensor(self.y[idx], dtype=torch.long)
        return x, y


# ---------------------------------------------------------------------------
# Head detection + freezing.
# ---------------------------------------------------------------------------
def freeze_backbone_unfreeze_head(model, n_outputs):
    for p in model.parameters():
        p.requires_grad = False

    head_name, head_module = None, None
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and module.out_features == n_outputs:
            head_name, head_module = name, module

    if head_module is None:
        raise RuntimeError(
            f"Could not auto-detect classification head "
            f"(no nn.Linear with out_features={n_outputs} found)."
        )

    for p in head_module.parameters():
        p.requires_grad = True

    n_head_params = sum(p.numel() for p in head_module.parameters())
    print(f"Fine-tuning head: '{head_name}' ({n_head_params} trainable params)")
    return head_module


# ---------------------------------------------------------------------------
# Train / eval
# ---------------------------------------------------------------------------
def run_epoch(model, loader, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    all_preds, all_targets = [], []

    with torch.set_grad_enabled(is_train):
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            out = model(xb)
            loss = criterion(out, yb)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * xb.size(0)
            all_preds.append(out.argmax(dim=1).cpu().numpy())
            all_targets.append(yb.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average="macro")
    return avg_loss, acc, macro_f1


def main(config_path='configs/baseline.yaml'):
    config = load_training_config(config_path)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    X, y_raw, subject_ids = load_dataset(config)
    n_times, n_chans = X.shape[-1], X.shape[1]

    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    n_outputs = len(le.classes_)
    if tuple(le.classes_) != config.class_names:
        raise ValueError(f"Observed classes {list(le.classes_)} differ from config.")
    print(f"Classes ({n_outputs}): {list(le.classes_)}")

    train_idx, val_idx, test_idx = subject_wise_split(
        X, y, subject_ids,
        test_size_subjects=config.test_subjects,
        val_size_subjects=config.val_subjects,
        random_state=config.seed,
    )

    train_ds = EEGDataset(X, y, train_idx)
    val_ds = EEGDataset(X, y, val_idx)
    test_ds = EEGDataset(X, y, test_idx)

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False)

    class_weights = compute_class_weight(
        "balanced", classes=np.arange(n_outputs), y=y[train_idx]
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float32, device=DEVICE)
    print(f"Class weights (train dist): {dict(zip(le.classes_, class_weights.cpu().numpy()))}")

    model = load_labram_model(
        n_outputs=n_outputs, n_times=n_times, n_chans=n_chans, sfreq=config.sfreq,
        pretrained_repo_id=(config.pretrained_repo_id if config.model_init == "pretrained" else None),
        pretrained_revision=config.pretrained_revision,
    ).to(DEVICE)
    freeze_backbone_unfreeze_head(model, n_outputs)
    print(f"Initialization mode: {config.model_init}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=config.learning_rate, weight_decay=config.weight_decay)

    best_val_f1 = -1.0
    epochs_no_improve = 0

    checkpoint_path = Path(config.checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, config.epochs + 1):
        train_loss, train_acc, train_f1 = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, val_f1 = run_epoch(model, val_loader, criterion, optimizer=None)

        print(f"Epoch {epoch:02d} | "
              f"train_loss={train_loss:.4f} acc={train_acc:.3f} f1={train_f1:.3f} | "
              f"val_loss={val_loss:.4f} acc={val_acc:.3f} f1={val_f1:.3f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            epochs_no_improve = 0
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  ↑ New best val macro-F1: {best_val_f1:.3f} (checkpoint saved)")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= config.patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {config.patience} epochs).")
                break

    print("\nLoading best checkpoint for final test evaluation...")
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE, weights_only=True))
    test_loss, test_acc, test_f1 = run_epoch(model, test_loader, criterion, optimizer=None)
    print(f"\nTEST  | loss={test_loss:.4f}  acc={test_acc:.3f}  macro_f1={test_f1:.3f}")

    # Save the held-out predictions so every reported figure is reproducible.
    # Store labels/probabilities, not raw EEG waveforms.
    model.eval()
    all_probs, all_labels, all_subjects = [], [], []
    with torch.no_grad():
        for batch_indices in range(0, len(test_idx), config.batch_size):
            row_ids = test_idx[batch_indices:batch_indices + config.batch_size]
            xb = torch.as_tensor(X[row_ids], dtype=torch.float32, device=DEVICE)
            logits = model(xb)
            all_probs.append(torch.softmax(logits, dim=1).cpu().numpy())
            all_labels.append(y[row_ids])
            all_subjects.append(subject_ids[row_ids])
    predictions_path = checkpoint_path.with_name("heldout_predictions.npz")
    np.savez_compressed(
        predictions_path,
        y_true=np.concatenate(all_labels),
        y_prob=np.concatenate(all_probs),
        subject_ids=np.concatenate(all_subjects),
        class_names=np.asarray(config.class_names),
    )
    print(f"Saved held-out prediction artifact: {predictions_path}")

    metadata = {
        "configuration": asdict(config), "torch_version": torch.__version__,
        "train_subjects": sorted(map(int, set(subject_ids[train_idx]))),
        "validation_subjects": sorted(map(int, set(subject_ids[val_idx]))),
        "test_subjects": sorted(map(int, set(subject_ids[test_idx]))),
        "best_validation_macro_f1": float(best_val_f1),
        "test_accuracy": float(test_acc), "test_macro_f1": float(test_f1),
        "historical_results_reproduced": False,
        "heldout_predictions_path": str(predictions_path),
    }
    checkpoint_path.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            out = model(xb.to(DEVICE))
            all_preds.append(out.argmax(dim=1).cpu().numpy())
            all_targets.append(yb.numpy())
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    print("\nPer-class report (test set):")
    print(classification_report(all_targets, all_preds, target_names=le.classes_))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/baseline.yaml")
    args = parser.parse_args()
    main(args.config)
