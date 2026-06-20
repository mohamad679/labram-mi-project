"""
Evaluation metrics: accuracy, F1, calibration.
Used across Phase 1 (baseline) and Phase 2 (uncertainty calibration).
"""

from sklearn.metrics import accuracy_score, f1_score


def compute_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, average="macro"),
    }


def calibration_curve_metrics(y_true, y_prob, n_bins: int = 10):
    """TODO (Phase 2): compute calibration curve / ECE for the
    uncertainty quantification step."""
    raise NotImplementedError
