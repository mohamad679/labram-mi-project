"""Benchmark metrics are checked against known values; no fabricated training."""
import numpy as np
import pytest

from scripts.build_benchmark import evaluate_arrays


def test_perfect_predictions():
    labels = np.array([0, 1, 2, 0])
    probs = np.eye(3)[labels]
    result = evaluate_arrays(labels, probs)
    assert result["accuracy"] == 1
    assert result["macro_f1"] == 1
    assert result["ece_10_bins"] == pytest.approx(0)
    assert result["confusion_matrix"] == [[2, 0, 0], [0, 1, 0], [0, 0, 1]]


def test_rejects_probabilities_not_summing_to_one():
    with pytest.raises(ValueError, match="sum to one"):
        evaluate_arrays(np.array([0, 1]), np.array([[0.3, 0.3], [0.2, 0.2]]))


def test_rejects_missing_prediction_class():
    with pytest.raises(ValueError, match="outside probability"):
        evaluate_arrays(np.array([0, 5]), np.eye(2))


def test_missing_class_reports_zero_f1_not_undefined():
    result = evaluate_arrays(np.array([0, 0]), np.array([[0.8, 0.2], [0.9, 0.1]]))
    assert result["macro_f1"] == pytest.approx(0.5)
