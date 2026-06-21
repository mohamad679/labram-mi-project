"""
Phase 1 — Fine-tune LaBraM's classification head on PhysioNet Motor Imagery.

Loads cached (or freshly fetched) data, applies the subject-wise split,
freezes the LaBraM backbone, and trains only the classification head
(linear-probe baseline). Reports accuracy + macro-F1 on val/test.
"""
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score, classification_report

from src.data import load_physionet_mi, subject_wise_split
from src.model import load_labram_model

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CACHE_PATHS = [
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
CHECKPOINT_PATH = "/kaggle/working/labram_head_finetuned.pt"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_dataset():
    for path in CACHE_PATHS:
        if os.path.exists(path):
            print(f"Loading cached data from {path}")
            data = np.load(path)
            return data["X"], data["y"], data["subject_ids"]

    print("No cache found — loading fresh via MOABB (slow).")
    from moabb.datasets import PhysionetMI
    all_subjects = [s for s in PhysionetMI().subject_list if s not in EXCLUDED_SUBJECTS]
    X, y, metadata = load_physionet_mi(subjects=all_subjects)
    subject_ids = metadata["subject"].values
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
# Finds the Linear layer whose out_features matches n_outputs instead of
# assuming a hardcoded attribute name (robust to internal LaBraM naming).
# ---------------------------------------------------------------------------
def freeze_backbone_unfreeze_head(model, n_outputs):
    for p in model.parameters():
        p.requires_grad = False

    head_name, head_module = None, None
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and module.out_features == n_outputs:
            head_name, head_module = name, module  # keep last match = final layer

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


def main():
    X, y_raw, subject_ids = load_dataset()
    n_times, n_chans = X.shape[-1], X.shape[1]

    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    n_outputs = len(le.classes_)
    print(f"Classes ({n_outputs}): {list(le.classes_)}")

    train_idx, val_idx, test_idx = subject_wise_split(X, y, subject_ids)

    train_ds = EEGDataset(X, y, train_idx)
    val_ds = EEGDataset(X, y, val_idx)
    test_ds = EEGDataset(X, y, test_idx)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    class_weights = compute_class_weight(
        "balanced", classes=np.arange(n_outputs), y=y[train_idx]
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float32, device=DEVICE)
    print(f"Class weights (train dist): {dict(zip(le.classes_, class_weights.cpu().numpy()))}")

    model = load_labram_model(
        n_outputs=n_outputs, n_times=n_times, n_chans=n_chans, sfreq=160.0
    ).to(DEVICE)
    freeze_backbone_unfreeze_head(model, n_outputs)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=LR, weight_decay=WEIGHT_DECAY)

    best_val_f1 = -1.0
    epochs_no_improve = 0

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc, train_f1 = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, val_f1 = run_epoch(model, val_loader, criterion, optimizer=None)

        print(f"Epoch {epoch:02d} | "
              f"train_loss={train_loss:.4f} acc={train_acc:.3f} f1={train_f1:.3f} | "
              f"val_loss={val_loss:.4f} acc={val_acc:.3f} f1={val_f1:.3f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            epochs_no_improve = 0
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            print(f"  ↑ New best val macro-F1: {best_val_f1:.3f} (checkpoint saved)")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"Early stopping at epoch {epoch} (no improvement for {PATIENCE} epochs).")
                break

    print("\nLoading best checkpoint for final test evaluation...")
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    test_loss, test_acc, test_f1 = run_epoch(model, test_loader, criterion, optimizer=None)
    print(f"\nTEST  | loss={test_loss:.4f}  acc={test_acc:.3f}  macro_f1={test_f1:.3f}")

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
    main()
