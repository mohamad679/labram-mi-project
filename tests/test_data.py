"""Offline tests for held-out subject splitting (no EEG download)."""
import numpy as np
import pytest

from src.data import subject_wise_split


def _example():
    groups = np.repeat(np.arange(12), 3)
    return np.zeros((len(groups), 64, 400), dtype=np.float32), groups % 5, groups

def test_no_subject_leakage():
    X, y, groups = _example()
    tr, val, test = subject_wise_split(X, y, groups, 3, 2, random_state=42)
    subjects = [set(groups[part]) for part in (tr, val, test)]
    assert [len(s) for s in subjects] == [7, 2, 3]
    assert all(not subjects[i] & subjects[j] for i in range(3) for j in range(i+1, 3))
    assert set(np.concatenate([tr, val, test])) == set(range(len(groups)))

def test_split_repeatability():
    X, y, groups = _example()
    a = subject_wise_split(X, y, groups, 3, 2, random_state=42)
    b = subject_wise_split(X, y, groups, 3, 2, random_state=42)
    assert all(np.array_equal(x, z) for x, z in zip(a, b))

def test_insufficient_subjects_rejected():
    X, y, groups = _example()
    with pytest.raises(ValueError):
        subject_wise_split(X, y, groups, test_size_subjects=12, val_size_subjects=2)
