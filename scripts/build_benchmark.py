"""Build reproducible, publication-oriented evaluation artifacts from held-out predictions.

Input is a .npz containing arrays y_true, y_prob, optional subject_ids and
class_names. No synthetic observations or fabricated benchmark scores.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss

from src.evaluate import calibration_curve_metrics


def evaluate_arrays(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    if y_true.ndim != 1 or y_prob.ndim != 2 or len(y_true) != len(y_prob):
        raise ValueError("Expected labels (N,) and probabilities (N,C) with equal N.")
    if len(y_true) == 0 or y_prob.shape[1] < 2:
        raise ValueError("Empty evaluation or fewer than two classes.")
    if not np.issubdtype(y_true.dtype, np.integer):
        raise ValueError("Labels must be integer encoded 0..C-1.")
    if np.any(y_true < 0) or np.any(y_true >= y_prob.shape[1]):
        raise ValueError("Labels outside probability columns.")
    if not np.all(np.isfinite(y_prob)) or np.any(y_prob < 0):
        raise ValueError("Probabilities must be finite and non-negative.")
    if not np.allclose(y_prob.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("Each probability row must sum to one.")
    y_pred = y_prob.argmax(axis=1)
    labels = np.arange(y_prob.shape[1])
    cal = calibration_curve_metrics(y_true, y_prob)
    return {
        "n_epochs": len(y_true),
        "class_counts": np.bincount(y_true, minlength=len(labels)).tolist(),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "balanced_accuracy": float(np.mean([
            (y_pred[y_true == cls] == cls).mean() if np.any(y_true == cls) else float("nan")
            for cls in labels
        ])),
        "log_loss": float(log_loss(y_true, y_prob, labels=labels)),
        "ece_10_bins": float(cal["ece"]),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "calibration": {k: v.tolist() if isinstance(v, np.ndarray) else v
                        for k, v in cal.items()},
    }


def write_report(source: Path, output_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with np.load(source, allow_pickle=False) as arrays:
        required = {"y_true", "y_prob", "class_names"}
        if not required.issubset(arrays.files):
            raise ValueError(f"Missing fields: {sorted(required - set(arrays.files))}")
        y_true, y_prob = arrays["y_true"], arrays["y_prob"]
        names = [str(v) for v in arrays["class_names"]]
        if len(names) != y_prob.shape[1] or len(set(names)) != len(names):
            raise ValueError("Class names do not match probability columns.")
        metrics = evaluate_arrays(y_true, y_prob)
        metrics["input_file"] = source.name
        if "subject_ids" in arrays.files:
            subject_ids = arrays["subject_ids"]
            if subject_ids.shape != y_true.shape:
                raise ValueError("subject_ids must align with labels.")
            metrics["n_test_subjects"] = int(len(np.unique(subject_ids)))

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    with (output_dir / "metrics.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["metric", "value"])
        for key in ("n_epochs", "accuracy", "macro_f1", "balanced_accuracy", "log_loss", "ece_10_bins"):
            writer.writerow([key, metrics[key]])

    cm = np.asarray(metrics["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(cm, cmap="Blues")
    fig.colorbar(image, ax=ax, label="Epoch count")
    ax.set_xticks(range(len(names)), names, rotation=35, ha="right")
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title("Held-out subject confusion matrix")
    for (i, j), value in np.ndenumerate(cm):
        ax.text(j, i, str(value), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=200)
    plt.close(fig)

    cal = metrics["calibration"]
    counts = np.asarray(cal["bin_counts"])
    occupied = counts > 0
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "--", label="Ideal")
    ax.plot(np.asarray(cal["bin_confs"])[occupied],
            np.asarray(cal["bin_accs"])[occupied], "o-", label="Observed")
    ax.set(xlabel="Mean confidence", ylabel="Fraction correct",
           title="Reliability diagram", xlim=(0, 1), ylim=(0, 1))
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "reliability_diagram.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output-dir", default="artifacts/evaluation", type=Path)
    args = parser.parse_args()
    write_report(args.predictions, args.output_dir)
