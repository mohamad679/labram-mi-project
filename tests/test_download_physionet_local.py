"""Offline tests: downloads can resume without restarting completed subjects."""
from __future__ import annotations

import json

import h5py
import numpy as np

from scripts import download_physionet_local as download
from src.portable_eeg import inspect_cache


def test_local_cache_resumes_subjects_without_refetch(monkeypatch, tmp_path):
    calls = []

    class Meta:
        def __getitem__(self, key):
            assert key == "subject"
            return self

        def to_numpy(self):
            return np.array([calls[-1]], dtype=np.int32)

    def fake_loader(subjects):
        calls.append(subjects[0])
        X = np.ones((2, 64, 400), dtype=np.float32)
        return X, np.array(["feet", "hands"]), Meta()

    monkeypatch.setattr(download, "load_physionet_mi", fake_loader)
    file = tmp_path / "cache.h5"
    first = download.build(file, subjects=(1,))
    assert first["status"] == "partial"
    assert calls == [1]

    again = download.build(file, subjects=(1, 2))
    assert again["status"] == "partial"
    assert calls == [1, 2]
    y, ids, shape = inspect_cache(file, require_complete=False)
    assert shape == (4, 64, 400)
    assert ids.tolist() == [1, 1, 2, 2]
    assert y.tolist() == ["feet", "hands", "feet", "hands"]
    assert download.sha256_file(file) == download.sha256_file(file)
    with h5py.File(file, "r") as f:
        assert json.loads(f.attrs["completed_subjects"]) == [1, 2]


def test_wrong_subject_fails_before_committing(monkeypatch, tmp_path):
    class WrongMeta:
        def __getitem__(self, key):
            return self

        def to_numpy(self):
            return np.array([3])

    monkeypatch.setattr(
        download, "load_physionet_mi",
        lambda subjects: (np.ones((1, 64, 400), dtype=np.float32),
                          np.array(["feet"]), WrongMeta()),
    )
    file = tmp_path / "cache.h5"
    import pytest
    with pytest.raises(ValueError, match="subject mismatch"):
        download.build(file, subjects=(1,))
    with h5py.File(file, "r") as f:
        assert json.loads(f.attrs["completed_subjects"]) == []
