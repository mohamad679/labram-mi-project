"""Deterministic validation-only and split-lock negative controls."""
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.run_phase3_validation import (
    SHA256_PHASE2,
    _dummy,
    score_validation,
    validate_frozen_split,
)

LOCK = Path(__file__).resolve().parents[1] / "configs/phase3_locked_split.json"


def test_dummy_prior_never_uses_validation_labels():
    y = np.asarray([3] * 10 + [0, 1, 2, 4] + [0, 1, 2, 3, 4])
    train = np.arange(14)
    valid = np.arange(14, 19)
    prob, meta = _dummy(y, train, valid)
    assert prob.shape == (5, 5)
    assert np.allclose(prob[:, 3], 10 / 14)
    assert meta["labels_from_train_only"]
    score = score_validation(y[valid], prob)
    assert score["balanced_accuracy"] == pytest.approx(0.2)


def test_prediction_metrics_reject_nonprobabilities():
    y = np.arange(5)
    prob = np.eye(5)
    assert score_validation(y, prob)["macro_f1"] == 1.0
    prob[0, :] = 0
    with pytest.raises(ValueError, match="probability"):
        score_validation(y, prob)
    with pytest.raises(ValueError, match="five-class"):
        score_validation(y, np.ones((5, 4)) / 4)


def test_frozen_split_exactly_matches_phase2_subject_ids():
    lock = json.loads(LOCK.read_text())
    counts = lock["partitions"]
    assert [len(counts[k]["subject_ids"]) for k in ("train", "validation", "test")] == [75, 14, 15]
    assert [counts[k]["epochs_count"] for k in ("train", "validation", "test")] == [13050, 2436, 2610]
    assert lock["dataset_sha256"] == SHA256_PHASE2
    parts = [set(counts[k]["subject_ids"]) for k in ("train", "validation", "test")]
    assert set.union(*parts) == (set(range(1, 110)) - {88, 89, 92, 100, 104})
    assert not any(parts[i] & parts[j] for i in range(3) for j in range(i + 1, 3))


def test_tampered_split_fails_closed():
    lock = json.loads(LOCK.read_text())
    all_subjects = np.concatenate([
        np.repeat(s, 174) for s in sorted(set(range(1, 110)) - {88, 89, 92, 100, 104})
    ])
    # The real dataset has exactly 174 epochs per retained subject.
    assert len(all_subjects) == 18096
    y = np.zeros(len(all_subjects), dtype=np.int64)
    groups = validate_frozen_split(y, all_subjects, lock)
    assert len(groups["test"]) == 2610
    lock["partitions"]["validation"]["subject_ids"][0] = 1
    with pytest.raises(ValueError, match="Frozen validation split"):
        validate_frozen_split(y, all_subjects, lock)
