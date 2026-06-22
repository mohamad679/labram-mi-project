"""
Phase 2 — Uncertainty Quantification via MC Dropout.

Method: MC Dropout (Gal & Ghahramani, 2016).

Injection points (verified from model inspection):
  - model.pos_drop  : Dropout(p=0.0) → replaced with Dropout(p=MC_DROPOUT_P)
    Acts on full token sequence (CLS + patches) before all 12 transformer
    blocks. Forward: x = self.pos_drop(x)
  - model.final_layer : Linear(200, 5) → wrapped as
    Sequential(Dropout(MC_DROPOUT_P), Linear(200, 5))
    Acts on the pooled CLS-token representation. Forward: x = self.final_layer(x)

Both replacements happen AFTER load_state_dict — Dropout has no learnable
parameters so state_dict keys are unaffected. strict=True loads cleanly.

Outputs per sample:
  mean_probs   : (N, C)  mean softmax over n_samples passes
  std_probs    : (N, C)  std across passes
  pred_entropy : (N,)    H[E_w p(y|x,w)]   — total uncertainty
  expected_ent : (N,)    E_w[H[p(y|x,w)]]  — aleatoric component
  bald_score   : (N,)    pred_entropy - expected_ent  — epistemic (BALD)

Scope note: conformal_calibrate stub is kept for reference only;
Conformal Prediction is out of scope for this project.
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader

from src.data import subject_wise_split
from src.model import load_labram_model
from src.train import CHECKPOINT_PATH, DEVICE, EEGDataset, load_dataset

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
MC_DROPOUT_P  = 0.3   # dropout probability at both injection points
MC_N_SAMPLES  = 50    # stochastic forward passes per batch
MC_BATCH_SIZE = 64    # inference batch size (no grad → larger than train)
RESULTS_PATH  = "/kaggle/working/mc_dropout_results.npz"


# ---------------------------------------------------------------------------
# Dropout injection
# ---------------------------------------------------------------------------
def inject_mc_dropout(model: nn.Module, dropout_p: float = MC_DROPOUT_P) -> nn.Module:
    """
    Modify two verified locations in the loaded LaBraM model to add
    stochastic Dropout for MC inference.

    1. Replace model.pos_drop (was Dropout(0.0)) with Dropout(dropout_p).
    2. Wrap model.final_layer with Sequential(Dropout(dropout_p), Linear).

    Call AFTER load_state_dict. No learnable parameters are touched.
    """
    # Site 1: positional dropout (token sequence, pre-transformer)
    model.pos_drop = nn.Dropout(p=dropout_p)

    # Site 2: head dropout (pooled CLS token, pre-linear)
    original_head = model.final_layer          # Linear(200, n_outputs)
    model.final_layer = nn.Sequential(
        nn.Dropout(p=dropout_p),
        original_head,
    )
    return model


def enable_mc_dropout(model: nn.Module) -> nn.Module:
    """
    Set model to eval() (stable LayerNorm statistics) then re-enable
    all nn.Dropout submodules for stochastic inference.

    Call after inject_mc_dropout. Each forward pass draws a fresh
    dropout mask from both injection sites.
    """
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()
    return model


# ---------------------------------------------------------------------------
# Core MC Dropout inference — single batch
# ---------------------------------------------------------------------------
def mc_dropout_predict(
    model: nn.Module,
    x: torch.Tensor,
    n_samples: int = MC_N_SAMPLES,
) -> dict:
    """
    Run n_samples stochastic forward passes on a single batch.

    Parameters
    ----------
    model    : nn.Module — inject_mc_dropout + enable_mc_dropout already applied
    x        : torch.Tensor (N, C, T)
    n_samples: int

    Returns
    -------
    dict:
      mean_probs   ndarray (N, n_classes)   mean softmax probability
      std_probs    ndarray (N, n_classes)   std across samples
      pred_entropy ndarray (N,)             total uncertainty
      expected_ent ndarray (N,)             aleatoric uncertainty
      bald_score   ndarray (N,)             epistemic uncertainty (BALD)
      predictions  ndarray (N,)             argmax of mean_probs
    """
    x = x.to(DEVICE)
    draws = []  # n_samples × (N, C)

    with torch.no_grad():
        for _ in range(n_samples):
            logits = model(x)                         # (N, C)
            probs  = torch.softmax(logits, dim=-1)    # (N, C)
            draws.append(probs.cpu().numpy())

    stack = np.stack(draws, axis=0)   # (n_samples, N, C)
    mean_probs = stack.mean(axis=0)   # (N, C)
    std_probs  = stack.std(axis=0)    # (N, C)

    eps = 1e-8
    # Total predictive entropy: H[mean_probs]
    pred_entropy = -(mean_probs * np.log(mean_probs + eps)).sum(axis=-1)  # (N,)

    # Per-draw entropy, averaged → aleatoric component
    per_draw_ent = -(stack * np.log(stack + eps)).sum(axis=-1)  # (n_samples, N)
    expected_ent = per_draw_ent.mean(axis=0)                     # (N,)

    # BALD = epistemic uncertainty
    bald_score = pred_entropy - expected_ent  # (N,)

    return {
        "mean_probs":   mean_probs,
        "std_probs":    std_probs,
        "pred_entropy": pred_entropy,
        "expected_ent": expected_ent,
        "bald_score":   bald_score,
        "predictions":  mean_probs.argmax(axis=-1),
    }


# ---------------------------------------------------------------------------
# Dataset-level inference
# ---------------------------------------------------------------------------
def run_mc_on_loader(
    model: nn.Module,
    loader: DataLoader,
    n_samples: int = MC_N_SAMPLES,
) -> dict:
    """
    Run mc_dropout_predict over every batch in loader.
    Concatenates results across batches; adds 'labels' key.
    """
    keys = ("mean_probs", "std_probs", "pred_entropy",
            "expected_ent", "bald_score", "predictions")
    accum  = {k: [] for k in keys}
    labels = []

    for xb, yb in loader:
        res = mc_dropout_predict(model, xb, n_samples=n_samples)
        for k in keys:
            accum[k].append(res[k])
        labels.append(yb.numpy())

    for k in keys:
        accum[k] = np.concatenate(accum[k], axis=0)
    accum["labels"] = np.concatenate(labels, axis=0)
    return accum


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
def compute_ece(
    mean_probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Expected Calibration Error (ECE).
    Bins by max confidence; measures |accuracy - confidence| per bin.
    Lower is better; 0.0 = perfect calibration.
    """
    conf    = mean_probs.max(axis=-1)
    preds   = mean_probs.argmax(axis=-1)
    correct = (preds == labels).astype(float)
    edges   = np.linspace(0.0, 1.0, n_bins + 1)
    ece     = 0.0
    n       = len(labels)

    for i in range(n_bins):
        mask = (conf >= edges[i]) & (conf < edges[i + 1])
        if not mask.any():
            continue
        ece += (mask.sum() / n) * abs(correct[mask].mean() - conf[mask].mean())

    return float(ece)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def print_uncertainty_report(
    results: dict,
    n_samples: int,
    class_names,
) -> None:
    labels = results["labels"]
    preds  = results["predictions"]
    acc    = accuracy_score(labels, preds)
    f1     = f1_score(labels, preds, average="macro")
    ece    = compute_ece(results["mean_probs"], labels)

    print("=" * 55)
    print("MC DROPOUT UNCERTAINTY REPORT")
    print(f"  n_samples        : {n_samples}")
    print(f"  dropout_p        : {MC_DROPOUT_P}")
    print("-" * 55)
    print(f"  Accuracy         : {acc:.4f}")
    print(f"  Macro-F1         : {f1:.4f}")
    print(f"  ECE              : {ece:.4f}")
    print("-" * 55)
    print(f"  Mean pred_entropy  (total unc) : {results['pred_entropy'].mean():.4f}")
    print(f"  Mean expected_ent  (aleatoric) : {results['expected_ent'].mean():.4f}")
    print(f"  Mean BALD          (epistemic) : {results['bald_score'].mean():.4f}")
    print(f"  Mean std_probs                 : {results['std_probs'].mean():.4f}")
    print("-" * 55)
    print("Per-class mean predictive entropy:")
    for c, name in enumerate(class_names):
        mask = labels == c
        if mask.any():
            print(f"  {name:<12}: {results['pred_entropy'][mask].mean():.4f}")
    print("=" * 55)


# ---------------------------------------------------------------------------
# Stub — out of scope
# ---------------------------------------------------------------------------
def conformal_calibrate(model, calibration_set):
    """
    Out of scope for this project (one method only — MC Dropout chosen).
    Stub kept to preserve the original API surface.
    """
    raise NotImplementedError(
        "Conformal Prediction is out of scope. Use mc_dropout_predict."
    )


# ---------------------------------------------------------------------------
# Entry point:  !python -m src.uncertainty
# ---------------------------------------------------------------------------
def main():
    X, y_raw, subject_ids = load_dataset()
    le = LabelEncoder()
    y  = le.fit_transform(y_raw)
    n_outputs         = len(le.classes_)
    n_times, n_chans  = X.shape[-1], X.shape[1]

    _, _, test_idx = subject_wise_split(X, y, subject_ids)
    test_ds     = EEGDataset(X, y, test_idx)
    test_loader = DataLoader(test_ds, batch_size=MC_BATCH_SIZE, shuffle=False)

    print(f"[Phase 2] Loading checkpoint: {CHECKPOINT_PATH}")
    model = load_labram_model(
        n_outputs=n_outputs, n_times=n_times, n_chans=n_chans, sfreq=160.0,
    ).to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))

    inject_mc_dropout(model, dropout_p=MC_DROPOUT_P)
    enable_mc_dropout(model)
    print(f"[Phase 2] MC Dropout injected: pos_drop + head (p={MC_DROPOUT_P})")

    print(f"[Phase 2] Running {MC_N_SAMPLES} stochastic passes per batch...")
    results = run_mc_on_loader(model, test_loader, n_samples=MC_N_SAMPLES)

    print_uncertainty_report(results, n_samples=MC_N_SAMPLES, class_names=le.classes_)

    np.savez_compressed(RESULTS_PATH, **results)
    print(f"[Phase 2] Results saved → {RESULTS_PATH}")


if __name__ == "__main__":
    main()
