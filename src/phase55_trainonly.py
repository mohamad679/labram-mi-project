"""Phase 5.5: train-partition-only subject CV, signal feature and scoring guards.

No Model/Scaler is allowed to see historical Validation or Test waveforms.
The frozen cohort is checked by Phase 3 before constructing these train-only folds.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

N_CLASSES = 5
BANDS = ((8., 12.), (12., 16.), (16., 22.), (22., 30.))
MODEL_NAMES = ("bandpower", "eegnet_raw", "eegnet_epoch_standardized", "eegnex_epoch_standardized")


def make_trainonly_folds(partitions: dict, subjects: np.ndarray, *, n_splits: int = 3,
                         inner_heldout_subjects: int = 8, seed: int = 42) -> list[dict]:
    """Outer subject-disjoint OOF; fixed inner subject holdout for deep early-stop.

    Raises on any leakage from 14 original Validation or 15 untouched Test subjects.
    The same folds are used by all candidate models.
    """
    if n_splits != 3 or inner_heldout_subjects != 8 or seed != 42:
        raise ValueError("Phase 5.5 protocol locks 3 outer folds, 8 inner subjects, seed=42")
    ids = np.asarray(subjects)
    train = np.asarray(partitions["train"], dtype=np.int64)
    val = np.asarray(partitions["validation"], dtype=np.int64)
    test = np.asarray(partitions["test"], dtype=np.int64)
    if any(x.ndim != 1 or x.size == 0 for x in (train, val, test)):
        raise ValueError("Empty or invalid original partitions")
    if not np.array_equal(np.unique(train), np.sort(train)):
        raise ValueError("Train row indices must be unique and sorted")
    s_train, s_val, s_test = [set(map(int, ids[r])) for r in (train, val, test)]
    if (len(s_train), len(s_val), len(s_test)) != (75, 14, 15):
        raise ValueError("Subject counts violate frozen protocol")
    if s_train & s_val or s_train & s_test or s_val & s_test:
        raise ValueError("Original subject partitions overlap")
    results = []
    seen_outer = []
    splitter = GroupKFold(n_splits=n_splits)
    for fold, (tr_loc, outer_loc) in enumerate(splitter.split(train, groups=ids[train]), 1):
        outer_train = train[tr_loc]
        outer_holdout = train[outer_loc]
        inner = GroupShuffleSplit(n_splits=1, test_size=inner_heldout_subjects,
                                  random_state=seed + fold)
        fit_local, early_local = next(inner.split(
            outer_train, groups=ids[outer_train]
        ))
        fit, early = outer_train[fit_local], outer_train[early_local]
        groups = [set(map(int, ids[z])) for z in (fit, early, outer_holdout)]
        if any(groups[i] & groups[j] for i in range(3) for j in range(i + 1, 3)):
            raise ValueError("Inner/outer subject leakage")
        if set.union(*groups) != s_train or any(z & (s_val | s_test) for z in groups):
            raise ValueError("Fold subjects do not partition the original Train cohort")
        if len(groups[1]) != inner_heldout_subjects:
            raise ValueError("Wrong number of inner early-stop subjects")
        seen_outer.extend(map(int, outer_holdout))
        results.append({"fold": fold, "fit": fit, "early": early,
                        "outer_train": outer_train, "outer_holdout": outer_holdout,
                        "fit_subjects": sorted(groups[0]),
                        "early_subjects": sorted(groups[1]),
                        "outer_subjects": sorted(groups[2])})
    if len(seen_outer) != len(train) or not np.array_equal(np.sort(seen_outer), train):
        raise ValueError("OOF predictions must cover Train rows exactly once")
    return results


class TrainOnlySignalSource:
    """Fail-closed guard before the HDF5 array is indexed, for all models."""

    def __init__(self, source, allowed_indices):
        self.source = source
        self.allowed = frozenset(map(int, allowed_indices))
        if len(self.allowed) != 13050:
            raise ValueError("Only the full 13,050-row original Train partition allowed")

    def __getitem__(self, index):
        if not np.isscalar(index) or isinstance(index, (bool, np.bool_)):
            raise ValueError("Only individual indexed train epochs can be loaded")
        idx = int(index)
        if idx not in self.allowed:
            raise PermissionError("Attempt to access original Validation or Test EEG waveform")
        return self.source[idx]


def epoch_standardize(x: np.ndarray) -> np.ndarray:
    """Per-epoch, per-electrode normalization; never fit group-wide statistics."""
    x = np.asarray(x, dtype=np.float32)
    if x.shape != (64, 400) or not np.isfinite(x).all():
        raise ValueError("Expected finite EEG 64x400")
    centered = x - x.mean(axis=-1, keepdims=True)
    std = x.std(axis=-1, keepdims=True)
    result = centered / np.maximum(std, 1e-6)
    if not np.isfinite(result).all():
        raise ValueError("Invalid normalized EEG")
    return result.astype(np.float32)


def make_bandpower_features(x: np.ndarray, *, sfreq: float = 160.) -> np.ndarray:
    """Prefiltered 8-32Hz EEG -> 4 finer frequency bands x 64 electrode log-var.

    Uses only each epoch; fit StandardScaler+classifier exclusively on outer train.
    This is NOT a true FBCSP algorithm and must never be labeled as such.
    """
    from scipy.signal import butter, sosfiltfilt

    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 2:
        x = x[None]
    if x.ndim != 3 or x.shape[1:] != (64, 400) or not np.isfinite(x).all():
        raise ValueError("Bandpower expects finite (N,64,400) EEG")
    if sfreq != 160.:
        raise ValueError("Frozen sampling frequency is 160 Hz")
    bands = []
    for low, high in BANDS:
        sos = butter(4, (low, high), btype="bandpass", fs=sfreq, output="sos")
        filtered = sosfiltfilt(sos, x.astype(np.float64), axis=-1)
        power = np.var(filtered, axis=-1)
        bands.append(np.log(np.maximum(power, 1e-12)))
    features = np.concatenate(bands, axis=1).astype(np.float32)
    if features.shape != (len(x), 256) or not np.isfinite(features).all():
        raise ValueError("Invalid filter-bank bandpower shape")
    return features


def score_five_class(y: np.ndarray, probs: np.ndarray, subject_ids: np.ndarray) -> dict:
    y = np.asarray(y, dtype=np.int64)
    probs = np.asarray(probs, dtype=np.float64)
    subject_ids = np.asarray(subject_ids)
    if y.ndim != 1 or len(y) == 0 or subject_ids.shape != y.shape:
        raise ValueError("Labels and subject IDs must align")
    if probs.shape != (len(y), N_CLASSES):
        raise ValueError("Expected a five-class probability matrix")
    if set(np.unique(y)) != set(range(N_CLASSES)):
        raise ValueError("All five labels required")
    if not np.isfinite(probs).all() or np.any(probs < 0) or not np.allclose(
        probs.sum(axis=1), 1.0, atol=1e-5,
    ):
        raise ValueError("Nonfinite/invalid probability rows")
    pred = probs.argmax(axis=1)
    recalls = [float(np.mean(pred[y == i] == i)) for i in range(N_CLASSES)]
    counts = np.bincount(pred, minlength=N_CLASSES)
    # Report every class and reject a total single-class collapse from model selection.
    fully_collapsed = int(np.count_nonzero(counts)) == 1
    by_subject = {}
    for subj in np.unique(subject_ids):
        rows = subject_ids == subj
        by_subject[str(int(subj))] = float(f1_score(
            y[rows], pred[rows], labels=np.arange(N_CLASSES),
            average="macro", zero_division=0,
        ))
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, labels=np.arange(N_CLASSES),
                                   average="macro", zero_division=0)),
        "balanced_accuracy": float(np.mean(recalls)),
        "class_recall": recalls,
        "true_class_counts": np.bincount(y, minlength=N_CLASSES).tolist(),
        "predicted_class_counts": counts.tolist(),
        "confusion_matrix": confusion_matrix(y, pred, labels=np.arange(N_CLASSES)).tolist(),
        "n_subjects": len(np.unique(subject_ids)),
        "per_subject_macro_f1": by_subject,
        "single_class_collapse": fully_collapsed,
    }


def select_predeclared(candidates: dict) -> dict:
    """Objective fixed before seeing any original 14-subject Validation data."""
    if not candidates or set(candidates) - set(MODEL_NAMES):
        raise ValueError("Unexpected candidate names")
    eligible = {k: v for k, v in candidates.items() if not v["single_class_collapse"]}
    if not eligible:
        return {"selected": None, "status": "NO_NONCOLLAPSED_CANDIDATE"}
    chosen = min(eligible, key=lambda k: (
        -eligible[k]["macro_f1"],
        -eligible[k]["balanced_accuracy"], k,
    ))
    return {"selected": chosen, "status": "TRAIN_ONLY_OOF_SELECTION", "metric": "macro_f1",
            "tie_breaker": "balanced_accuracy_then_lexicographic_name"}
