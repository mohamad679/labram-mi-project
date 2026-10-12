"""Build evidence-linked test metrics without inferring anything from screenshots."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from src.evaluate import calibration_curve_metrics


def evaluate_probabilities(
    y_true: np.ndarray, y_prob: np.ndarray, class_names: tuple[str, ...]
) -> tuple[dict, np.ndarray, dict]:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    if y_true.ndim != 1 or y_prob.ndim != 2 or y_prob.shape != (
        len(y_true), len(class_names)
    ):
        raise ValueError("Expected y_true=(N,) and y_prob=(N,number_of_classes).")
    if len(y_true) == 0:
        raise ValueError("Evaluation set must not be empty.")
    if not np.isfinite(y_prob).all() or np.any(y_prob < 0) or np.any(y_prob > 1):
        raise ValueError("Invalid prediction probabilities.")
    if not np.allclose(y_prob.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("Each probability row must sum to 1.")
    if not np.issubdtype(y_true.dtype, np.integer):
        raise ValueError("Class labels must be encoded as integers.")
    if ((y_true < 0) | (y_true >= len(class_names))).any():
        raise ValueError("Unexpected class labels in evaluation set.")

    y_pred = y_prob.argmax(axis=1)
    labels = list(range(len(class_names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    calibration = calibration_curve_metrics(y_true, y_prob, n_bins=10)
    metrics = {
        "n_test_epochs": int(len(y_true)),
        "class_names": list(class_names),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(
            y_true, y_pred, labels=labels, average="macro", zero_division=0
        )),
        "ece_10_bin": float(calibration["ece"]),
        "class_support": cm.sum(axis=1).tolist(),
        "note": (
            "Held-out epoch scores; do not interpret epochs as independent "
            "subjects. Confidence intervals require subject-cluster evaluation."
        ),
    }
    return metrics, cm, calibration


def render_evaluation(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: tuple[str, ...],
    output_dir: str | Path,
) -> dict:
    """Only render plots from supplied evaluation predictions."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metrics, cm, calibration = evaluate_probabilities(y_true, y_prob, class_names)

    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(cm, cmap="Blues")
    fig.colorbar(image, ax=ax, label="Held-out epochs")
    ax.set_xticks(range(len(class_names)), class_names, rotation=35, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground truth")
    ax.set_title("Held-out confusion matrix")
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(output / "confusion_matrix.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    valid = calibration["bin_counts"] > 0
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Ideal calibration")
    ax.plot(calibration["bin_confs"][valid], calibration["bin_accs"][valid],
            marker="o", label="Observed bins")
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean confidence",
           ylabel="Observed accuracy", title="Held-out reliability diagram")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "reliability_diagram.png", dpi=180)
    plt.close(fig)

    (output / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    return metrics
