"""Never generate credible-looking figures from invalid probability arrays."""
import numpy as np
import pytest

from src.reporting import evaluate_probabilities


def test_perfect_results_and_confusion_matrix():
    truth = np.array([0, 1, 2, 1])
    probs = np.eye(3)[truth]
    metrics, cm, calibration = evaluate_probabilities(truth, probs, ("a", "b", "c"))
    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["ece_10_bin"] == pytest.approx(0)
    assert list(np.diag(cm)) == [1, 2, 1]
    assert calibration["bin_counts"].sum() == len(truth)


@pytest.mark.parametrize("probs", [
    [[0.8, 0.8]], [[float("nan"), 0.5]], [[-0.2, 1.2]]
])
def test_invalid_predictions_rejected(probs):
    with pytest.raises(ValueError):
        evaluate_probabilities(np.array([0]), np.array(probs), ("a", "b"))


def test_missing_class_is_reported_not_dropped():
    truth = np.array([0, 0])
    probs = np.array([[0.9, 0.1], [0.9, 0.1]])
    metrics, cm, _ = evaluate_probabilities(truth, probs, ("a", "b"))
    assert cm.shape == (2, 2)
    assert metrics["class_support"] == [2, 0]
