"""Offline subject-level uncertainty validation."""
import numpy as np
import pytest

from src.subject_statistics import subject_cluster_bootstrap


def _data():
    labels = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=int)
    subjects = np.repeat([1, 2, 3, 4], 2)
    predictions = np.eye(2)[labels].astype(float)
    return labels, predictions, subjects


def test_perfect_classifier_has_perfect_subject_ci():
    y, p, s = _data()
    summary = subject_cluster_bootstrap(y, p, s, n_bootstraps=25, seed=3)
    assert summary["accuracy"] == {"point": 1.0, "lower": 1.0, "upper": 1.0}
    assert summary["macro_f1"]["lower"] == pytest.approx(1.0)
    assert summary["n_heldout_subjects"] == 4


def test_deterministic_given_seed():
    y, p, s = _data()
    p[0] = [0.2, 0.8]
    a = subject_cluster_bootstrap(y, p, s, seed=49, n_bootstraps=30)
    b = subject_cluster_bootstrap(y, p, s, seed=49, n_bootstraps=30)
    assert a == b
    assert a["accuracy"]["lower"] <= a["accuracy"]["point"]


def test_rejects_one_subject():
    y, p, _ = _data()
    with pytest.raises(ValueError, match="two held-out"):
        subject_cluster_bootstrap(y, p, np.ones(len(y)), n_bootstraps=2)


def test_rejects_nan_probabilities():
    y, p, s = _data()
    p[0] = [np.nan, np.nan]
    with pytest.raises(ValueError, match="finite"):
        subject_cluster_bootstrap(y, p, s, n_bootstraps=2)


def test_rejects_misaligned_epoch_subject_count():
    y, p, s = _data()
    with pytest.raises(ValueError, match="align"):
        subject_cluster_bootstrap(y, p, s[:-1], n_bootstraps=2)
