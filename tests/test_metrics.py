"""Boundary tests for accuracy, macro-F1 and calibration."""
import numpy as np
import pytest

from src.evaluate import calibration_curve_metrics, compute_metrics


def test_metrics_perfect_classification():
    result = compute_metrics([0, 1, 2], [0, 1, 2])
    assert result["accuracy"] == 1
    assert result["f1"] == 1


def test_ece_includes_probability_one_correct_predictions():
    p = np.eye(3)
    result = calibration_curve_metrics(np.arange(3), p, n_bins=10)
    assert result["bin_counts"].sum() == 3
    assert result["ece"] == pytest.approx(0)


def test_ece_includes_probability_one_incorrect_predictions():
    p = np.eye(3)
    result = calibration_curve_metrics(np.array([1, 2, 0]), p, n_bins=10)
    assert result["bin_counts"].sum() == 3
    assert result["ece"] == pytest.approx(1)
