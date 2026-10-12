"""Synthetic-only regression tests for portable, resumable EEG HDF5 caches."""
import h5py
import numpy as np
import pytest

from src.portable_eeg import (
    EXPECTED_SUBJECTS,
    H5SignalSource,
    append_subject,
    initialize_cache,
    inspect_cache,
)


def _epoch(n=2):
    return np.ones((n, 64, 400), dtype=np.float32)


def test_create_append_and_lazy_read(tmp_path):
    path = tmp_path / "one.h5"
    initialize_cache(path)
    assert append_subject(path, 1, _epoch(), ["feet", "hands"])
    assert not append_subject(path, 1, _epoch(), ["feet", "hands"])
    y, ids, shape = inspect_cache(path, require_complete=False)
    assert shape == (2, 64, 400)
    assert y.tolist() == ["feet", "hands"]
    assert ids.tolist() == [1, 1]
    with pytest.raises(ValueError, match="incomplete"):
        inspect_cache(path, require_complete=True)
    reader = H5SignalSource(path, shape)
    assert reader.shape == shape
    assert np.array_equal(reader[1], _epoch()[0])
    assert np.array_equal(reader[[1, 0]], _epoch())
    reader.close()


def test_interrupted_subject_is_rolled_back(tmp_path):
    path = tmp_path / "resume.h5"
    initialize_cache(path)
    append_subject(path, 1, _epoch(1), ["feet"])
    with h5py.File(path, "r+") as h:
        for name in ("X", "y", "subject_ids"):
            h[name].resize((3, *h[name].shape[1:]))
    with pytest.raises(ValueError, match="Uncommitted"):
        inspect_cache(path, require_complete=False)
    initialize_cache(path)
    assert append_subject(path, 2, _epoch(1), ["hands"])
    y, ids, shape = inspect_cache(path, require_complete=False)
    assert y.tolist() == ["feet", "hands"]
    assert ids.tolist() == [1, 2]
    assert shape[0] == 2


def test_rejects_wrong_labels_shape_and_excluded_subject(tmp_path):
    path = tmp_path / "bad.h5"
    initialize_cache(path)
    with pytest.raises(ValueError, match="Excluded"):
        append_subject(path, 88, _epoch(1), ["feet"])
    with pytest.raises(ValueError, match="shape"):
        append_subject(path, 1, np.ones((1, 63, 400), dtype=np.float32), ["feet"])
    with pytest.raises(ValueError, match="Unknown"):
        append_subject(path, 1, _epoch(1), ["unknown"])
    with pytest.raises(ValueError, match="non-finite"):
        append_subject(path, 1, np.full((1, 64, 400), np.nan), ["feet"])


def test_complete_full_cohort_and_manifest_checks(tmp_path):
    path = tmp_path / "full.h5"
    initialize_cache(path)
    labels = ["feet", "hands", "left_hand", "rest", "right_hand"]
    for i, subject in enumerate(EXPECTED_SUBJECTS):
        append_subject(path, subject, _epoch(1), [labels[i % len(labels)]])
    y, ids, shape = inspect_cache(path, require_complete=True)
    assert len(set(ids)) == 104
    assert shape == (104, 64, 400)
    with pytest.raises(ValueError, match="Complete"):
        append_subject(path, 109, _epoch(1), ["feet"])
    assert len(y) == len(ids)


def test_reject_corrupted_provenance(tmp_path):
    path = tmp_path / "corrupt.h5"
    initialize_cache(path)
    append_subject(path, 1, _epoch(1), ["feet"])
    with h5py.File(path, "r+") as h:
        h.attrs["input_unit"] = "V"
    with pytest.raises(ValueError, match="units"):
        inspect_cache(path, require_complete=False)
