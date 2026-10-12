"""Fail-closed Phase 3 split manifest invariants (offline, no EEG downloads)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.data import subject_wise_split
from src.portable_eeg import EXPECTED_SUBJECTS

FROZEN = Path(__file__).resolve().parents[1] / "configs/phase3_locked_split.json"
EXPECTED_DATA_SHA256 = "f6d2c7824f6aaa522286ea8ee5253b0220602117fc69d7b49c3af6e6a1324f7a"


def test_split_manifest_matches_phase2_source_cohort():
    manifest = json.loads(FROZEN.read_text(encoding="utf-8"))
    assert manifest["dataset_sha256"] == EXPECTED_DATA_SHA256
    assert manifest["split_seed"] == 42
    subjects = {
        key: set(manifest["partitions"][key]["subject_ids"])
        for key in ("train", "validation", "test")
    }
    assert {key: len(ids) for key, ids in subjects.items()} == {
        "train": 75, "validation": 14, "test": 15,
    }
    assert not subjects["train"] & subjects["validation"]
    assert not subjects["train"] & subjects["test"]
    assert not subjects["validation"] & subjects["test"]
    assert set.union(*subjects.values()) == set(EXPECTED_SUBJECTS)


def test_real_cohort_split_reproduced_from_synthetic_group_ids():
    """Tests group split ordering, NOT simulated EEG model performance."""
    manifest = json.loads(FROZEN.read_text(encoding="utf-8"))
    # All retained PhysioNet subjects contribute 174 epochs, per Phase 2
    group_ids = np.repeat(np.asarray(EXPECTED_SUBJECTS), 174)
    assert len(group_ids) == 18_096
    train, val, test = subject_wise_split(
        np.arange(len(group_ids)), np.zeros(len(group_ids), dtype=int),
        group_ids, test_size_subjects=15, val_size_subjects=14, random_state=42,
    )
    for key, index in (("train", train), ("validation", val), ("test", test)):
        expected = manifest["partitions"][key]
        assert len(index) == expected["epochs_count"]
        assert sorted(map(int, set(group_ids[index]))) == expected["subject_ids"]
    assert sorted(np.concatenate((train, val, test)).tolist()) == list(
        range(len(group_ids))
    )
