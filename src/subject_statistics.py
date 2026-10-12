"""Subject-cluster bootstrap for a subject-disjoint EEG evaluation.

Trials from the same EEG participant are dependent. Resample held-out
participants, carrying all epochs of each sampled participant together.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def subject_cluster_bootstrap(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    subject_ids: np.ndarray,
    *,
    n_bootstraps: int = 1000,
    seed: int = 42,
    confidence_level: float = 0.95,
) -> dict:
    """Return accuracy and fixed-class macro-F1 with subject-resampled intervals."""
    labels = np.asarray(y_true)
    probabilities = np.asarray(y_prob, dtype=np.float64)
    subjects = np.asarray(subject_ids)
    if labels.ndim != 1 or subjects.ndim != 1 or probabilities.ndim != 2:
        raise ValueError("Expected labels/subjects (N,) and probabilities (N,C).")
    if len(labels) == 0 or len(labels) != len(subjects) or len(labels) != len(probabilities):
        raise ValueError("Epoch labels, subjects and predictions must align and be nonempty.")
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("Labels must be integer-encoded.")
    if probabilities.shape[1] < 2 or np.any(labels < 0) or np.any(labels >= probabilities.shape[1]):
        raise ValueError("Invalid class vocabulary.")
    if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0):
        raise ValueError("Probabilities must be finite and nonnegative.")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("Probability rows must sum to one.")
    if n_bootstraps < 1 or not (0 < confidence_level < 1):
        raise ValueError("n_bootstraps must be positive and confidence_level in (0,1).")
    unique_subjects, inverse = np.unique(subjects, return_inverse=True)
    if len(unique_subjects) < 2:
        raise ValueError("At least two held-out subjects are required for uncertainty intervals.")
    indices = [np.flatnonzero(inverse == i) for i in range(len(unique_subjects))]
    predicted = probabilities.argmax(axis=1)
    class_ids = np.arange(probabilities.shape[1])

    def score(index):
        return (
            float(accuracy_score(labels[index], predicted[index])),
            float(f1_score(
                labels[index], predicted[index], labels=class_ids,
                average="macro", zero_division=0,
            )),
        )

    point = score(np.arange(len(labels)))
    rng = np.random.default_rng(seed)
    scores = np.empty((n_bootstraps, 2), dtype=np.float64)
    for iteration in range(n_bootstraps):
        chosen = rng.integers(0, len(unique_subjects), size=len(unique_subjects))
        selected_indices = np.concatenate([indices[i] for i in chosen])
        scores[iteration] = score(selected_indices)
    alpha = (1 - confidence_level) / 2
    lower, upper = np.quantile(scores, [alpha, 1 - alpha], axis=0)
    return {
        "method": "subject_cluster_percentile_bootstrap",
        "n_heldout_subjects": int(len(unique_subjects)),
        "n_heldout_epochs": int(len(labels)),
        "n_bootstraps": int(n_bootstraps),
        "seed": int(seed),
        "confidence_level": float(confidence_level),
        "accuracy": {"point": point[0], "lower": float(lower[0]), "upper": float(upper[0])},
        "macro_f1": {"point": point[1], "lower": float(lower[1]), "upper": float(upper[1])},
        "note": "Subjects, not epochs, resampled. Percentile CIs do not replace multi-seed experiments.",
    }
