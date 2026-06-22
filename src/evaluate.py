"""
Evaluation metrics: accuracy, F1, calibration.
Used across Phase 1 (baseline) and Phase 2 (uncertainty calibration).
"""

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def compute_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, average="macro"),
    }


def calibration_curve_metrics(
    y_true,
    y_prob,
    n_bins: int = 10,
) -> dict:
    """
    Compute calibration curve data and Expected Calibration Error (ECE).

    Parameters
    ----------
    y_true : array-like (N,)    integer class labels
    y_prob : array-like (N, C)  softmax probabilities (e.g. from mc_dropout_predict)
    n_bins : int                number of confidence bins

    Returns
    -------
    dict:
      bin_confs  : ndarray (n_bins,)  mean confidence per non-empty bin
      bin_accs   : ndarray (n_bins,)  mean accuracy per non-empty bin
      bin_counts : ndarray (n_bins,)  sample count per bin
      ece        : float              Expected Calibration Error
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    conf    = y_prob.max(axis=-1)
    preds   = y_prob.argmax(axis=-1)
    correct = (preds == y_true).astype(float)

    edges      = np.linspace(0.0, 1.0, n_bins + 1)
    bin_confs  = np.zeros(n_bins)
    bin_accs   = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins, dtype=int)

    for i in range(n_bins):
        mask = (conf >= edges[i]) & (conf < edges[i + 1])
        n_in = mask.sum()
        if n_in == 0:
            continue
        bin_counts[i] = n_in
        bin_confs[i]  = conf[mask].mean()
        bin_accs[i]   = correct[mask].mean()

    ece = float(
        np.sum((bin_counts / len(y_true)) * np.abs(bin_accs - bin_confs))
    )

    return {
        "bin_confs":  bin_confs,
        "bin_accs":   bin_accs,
        "bin_counts": bin_counts,
        "ece":        ece,
    }
