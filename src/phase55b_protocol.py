"""Frozen Phase 5.5b: one candidate, one historical Validation pass; Test forbidden.

Only participant IDs and labels may be inspected before the final evaluation;
waveform IO is separately restricted to the permitted partition.
"""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import GroupShuffleSplit

SELECTED_MODEL = "eegnet_epoch_standardized"
INNER_SPLIT_SEED = 142
INNER_EARLY_SUBJECTS = (9, 25, 56, 61, 70, 81, 83, 90)
HISTORICAL_VALIDATION_SUBJECTS = (2, 6, 13, 15, 22, 26, 36, 39, 50, 52, 58, 62, 75, 98)
HISTORICAL_TEST_SUBJECTS = (1, 11, 19, 31, 32, 46, 48, 54, 65, 66, 78, 91, 95, 97, 99)
EPOCHS_MAX = 25
PATIENCE = 5
BATCH_SIZE = 32
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
MODEL_SEED = 42


def make_frozen_final_split(partitions: dict, subject_ids: np.ndarray) -> dict:
    """Verify precommitted 67/8 inner Train split without seeing Validation EEG."""
    s = np.asarray(subject_ids)
    train = np.asarray(partitions["train"], dtype=np.int64)
    validation = np.asarray(partitions["validation"], dtype=np.int64)
    test = np.asarray(partitions["test"], dtype=np.int64)
    if any(v.ndim != 1 or not len(v) for v in (train, validation, test)):
        raise ValueError("Missing or malformed partition indices")
    if len(train) != 13050 or len(validation) != 2436 or len(test) != 2610:
        raise ValueError("Unexpected immutable partition epoch counts")
    if not np.array_equal(np.unique(train), train):
        raise ValueError("Train global row indices must be sorted and unique")
    id_sets = [set(map(int, s[r])) for r in (train, validation, test)]
    if tuple(map(len, id_sets)) != (75, 14, 15):
        raise ValueError("Frozen participant count mismatch")
    if any(id_sets[i] & id_sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("Participant leakage in original partitions")
    if tuple(sorted(id_sets[1])) != HISTORICAL_VALIDATION_SUBJECTS:
        raise ValueError("Unexpected historical Validation participants")
    if tuple(sorted(id_sets[2])) != HISTORICAL_TEST_SUBJECTS:
        raise ValueError("Unexpected locked Test participants")
    unique_train = np.asarray(sorted(id_sets[0]))
    gss = GroupShuffleSplit(n_splits=1, test_size=8, random_state=INNER_SPLIT_SEED)
    _, early_ix = next(gss.split(unique_train, groups=unique_train))
    early_subject_ids = set(map(int, unique_train[early_ix]))
    if tuple(sorted(early_subject_ids)) != INNER_EARLY_SUBJECTS:
        raise RuntimeError("Pinned inner Train-only split changed")
    fit_mask = ~np.isin(s[train], list(early_subject_ids))
    fit, early = train[fit_mask], train[~fit_mask]
    fit_subject_ids = set(map(int, s[fit]))
    if len(fit_subject_ids) != 67 or len(early) == 0 or len(fit) == 0:
        raise ValueError("Expected 67 fit and 8 inner early-stop participants")
    if fit_subject_ids & early_subject_ids or fit_subject_ids | early_subject_ids != id_sets[0]:
        raise ValueError("Train-only inner split invalid")
    return {"fit": fit, "early": early, "validation": validation,
            "test": test, "train": train,
            "fit_subjects": sorted(fit_subject_ids),
            "early_subjects": sorted(early_subject_ids)}


class PermittedEpochSource:
    """Deny data reads outside an explicitly named frozen partition."""

    def __init__(self, source, allowed_indices, role: str):
        if role not in ("train", "validation"):
            raise ValueError("Only train or historical validation signal access is legal")
        self.source = source
        self.role = role
        self.allowed = frozenset(map(int, allowed_indices))
        expected_count = 13050 if role == "train" else 2436
        if len(self.allowed) != expected_count:
            raise ValueError(f"{role}: expected exactly {expected_count} allowed rows")

    def __getitem__(self, index):
        if not np.isscalar(index) or isinstance(index, (bool, np.bool_)):
            raise PermissionError("Only scalar row indices are permitted")
        value = int(index)
        if value not in self.allowed:
            raise PermissionError(f"Waveform access outside {self.role} is forbidden")
        return self.source[value]


def verify_train_only_nomination(evidence: dict) -> None:
    """Reject any changed candidate list or re-selection from historic Validation."""
    if evidence.get("selection", {}).get("selected") != SELECTED_MODEL:
        raise ValueError("Phase 5.5a frozen nominated candidate does not match")
    if evidence.get("phase55a_gate") != "PASS_FOR_COMPLETED_TRAIN_ONLY_CANDIDATE_SELECTION_WITH_DECLARED_LIMITS":
        raise ValueError("Phase 5.5a Train-only selection evidence is not accepted")
    if evidence.get("dataset", {}).get("train_subjects") != 75:
        raise ValueError("Training-cohort evidence mismatch")
    if evidence.get("dataset", {}).get("validation_or_test_predictions_in_zip") != 0:
        raise ValueError("Selection evidence unexpectedly used historical heldouts")
