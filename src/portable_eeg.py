"""Portable, resume-safe PhysioNet EEG cache with bounded RAM usage.

HDF5 cache schema 3: one file with compressed, append-only 64 x 400 epochs.
Only completed subjects are committed. No untrusted pickle deserialization.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.channels import PHYSIONET_64_CH_NAMES, verify_channel_order, verify_sampling_rate

CACHE_SCHEMA = 3
N_CHANNELS = 64
N_TIMES = 400
LABELS = ("feet", "hands", "left_hand", "rest", "right_hand")
EXCLUDED_SUBJECTS = (88, 89, 92, 100, 104)
EXPECTED_SUBJECTS = tuple(i for i in range(1, 110) if i not in EXCLUDED_SUBJECTS)


def _h5py():
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("HDF5 support requires: pip install h5py") from exc
    return h5py


def _decode(value):
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def initialize_cache(path: str | Path) -> None:
    """Create the cache if needed; repair an interrupted last subject append."""
    h5py = _h5py()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "a") as file:
        if "X" not in file:
            if file.keys():
                raise ValueError("Cannot initialize nonempty unrecognized HDF5 file.")
            file.create_dataset(
                "X", shape=(0, N_CHANNELS, N_TIMES),
                maxshape=(None, N_CHANNELS, N_TIMES),
                chunks=(2, N_CHANNELS, N_TIMES), compression="lzf", dtype="f4",
            )
            file.create_dataset("y", shape=(0,), maxshape=(None,), dtype="S16")
            file.create_dataset("subject_ids", shape=(0,), maxshape=(None,), dtype="i2")
            file.attrs["channel_names"] = np.asarray(PHYSIONET_64_CH_NAMES, dtype="S8")
            file.attrs["sfreq"] = 160.0
            file.attrs["input_unit"] = "uV"
            file.attrs["scaling_factor_from_volts"] = 1e6
            file.attrs["cache_schema_version"] = CACHE_SCHEMA
            file.attrs["completed_subjects"] = json.dumps([])
            file.attrs["committed_rows"] = 0
            file.attrs["complete"] = False
            file.flush()
        verify_cache_metadata(file)
        committed = int(file.attrs["committed_rows"])
        for name in ("X", "y", "subject_ids"):
            dataset = file[name]
            if dataset.shape[0] < committed:
                raise ValueError("Cache is shorter than its durable commit marker.")
            if dataset.shape[0] > committed:
                dataset.resize((committed, *dataset.shape[1:]))
        file.flush()


def verify_cache_metadata(file) -> None:
    if int(file.attrs.get("cache_schema_version", -1)) != CACHE_SCHEMA:
        raise ValueError("Expected portable EEG HDF5 cache schema 3.")
    verify_channel_order([_decode(x) for x in file.attrs["channel_names"]])
    verify_sampling_rate(float(file.attrs["sfreq"]))
    if _decode(file.attrs["input_unit"]) != "uV":
        raise ValueError("Unexpected EEG cache units; expected microvolts.")
    if not np.isclose(float(file.attrs["scaling_factor_from_volts"]), 1e6):
        raise ValueError("Incorrect volts-to-microvolts scaling.")
    if file["X"].shape[1:] != (N_CHANNELS, N_TIMES):
        raise ValueError("Unexpected EEG epoch shape.")
    if file["X"].dtype != np.dtype("float32"):
        raise ValueError("EEG cache must use float32.")
    if int(file.attrs["committed_rows"]) < 0:
        raise ValueError("Negative durable cache position.")
    subjects = json.loads(_decode(file.attrs["completed_subjects"]))
    if len(subjects) != len(set(subjects)) or any(s not in EXPECTED_SUBJECTS for s in subjects):
        raise ValueError("Invalid cached subject manifest.")


def append_subject(path: str | Path, subject: int, X, y) -> bool:
    """Append one verified subject; on retry, skip previously committed subjects."""
    h5py = _h5py()
    subject = int(subject)
    if subject not in EXPECTED_SUBJECTS:
        raise ValueError(f"Excluded or unknown subject: {subject}")
    X = np.asarray(X, dtype=np.float32)
    labels = np.asarray(y).astype(str)
    if X.ndim != 3 or X.shape[1:] != (N_CHANNELS, N_TIMES):
        raise ValueError("Expected EEG shape (epochs, 64, 400).")
    if X.shape[0] == 0 or len(labels) != len(X):
        raise ValueError("Empty or misaligned EEG subject.")
    if not set(labels).issubset(set(LABELS)) or not np.all(np.isfinite(X)):
        raise ValueError("Unknown labels or non-finite EEG samples.")
    with h5py.File(path, "r+") as file:
        verify_cache_metadata(file)
        done = json.loads(_decode(file.attrs["completed_subjects"]))
        if subject in done:
            return False
        if file.attrs["complete"]:
            raise ValueError("Complete cache cannot be extended.")
        start = int(file.attrs["committed_rows"])
        if any(file[name].shape[0] != start for name in ("X", "y", "subject_ids")):
            raise ValueError("Incomplete append: first resume with initialize_cache().")
        end = start + len(X)
        # Subject writes are committed only after all three arrays are flushed.
        for name in ("X", "y", "subject_ids"):
            file[name].resize((end, *file[name].shape[1:]))
        file["X"][start:end] = X
        file["y"][start:end] = labels.astype("S16")
        file["subject_ids"][start:end] = subject
        file.flush()
        file.attrs["committed_rows"] = end
        file.attrs["completed_subjects"] = json.dumps(done + [subject])
        file.attrs["complete"] = len(done) + 1 == len(EXPECTED_SUBJECTS)
        file.flush()
    return True


def inspect_cache(path: str | Path, require_complete: bool = True):
    """Fail closed on truncated, unverified, or incomplete HDF5 files."""
    h5py = _h5py()
    with h5py.File(path, "r") as file:
        verify_cache_metadata(file)
        n = int(file.attrs["committed_rows"])
        if any(file[key].shape[0] != n for key in ("X", "y", "subject_ids")):
            raise ValueError("Uncommitted or truncated HDF5 EEG file.")
        done = json.loads(_decode(file.attrs["completed_subjects"]))
        if require_complete and (not file.attrs["complete"] or set(done) != set(EXPECTED_SUBJECTS)):
            raise ValueError(f"EEG cache incomplete: {len(done)}/{len(EXPECTED_SUBJECTS)} subjects.")
        y = np.asarray([_decode(x) for x in file["y"][:]], dtype="<U16")
        subjects = np.asarray(file["subject_ids"][:], dtype=np.int32)
        if len(y) != n or set(y) != set(LABELS) and require_complete:
            raise ValueError("Missing epochs or expected MI classes.")
        if any(int(np.count_nonzero(subjects == s)) == 0 for s in done):
            raise ValueError("Completed subject has no EEG epochs.")
        if set(map(int, np.unique(subjects))) != set(done):
            raise ValueError("Subject IDs and committed manifest differ.")
        return y, subjects, (n, N_CHANNELS, N_TIMES)


class H5SignalSource:
    """Lazy EEG reader; one HDF5 handle per process, never materializes the cohort."""

    def __init__(self, path: str | Path, shape: tuple[int, int, int]):
        self.path = str(path)
        self.shape = tuple(shape)
        self._file = None
        self._pid = None

    def __len__(self):
        return self.shape[0]

    def _dataset(self):
        import os
        if self._file is None or self._pid != os.getpid():
            self.close()
            self._file = _h5py().File(self.path, "r")
            self._pid = os.getpid()
        return self._file["X"]

    def __getitem__(self, index):
        data = self._dataset()
        if np.isscalar(index):
            return np.asarray(data[int(index)], dtype=np.float32)
        # h5py fancy indexing requires sorted unique IDs; preserve caller ordering.
        return np.stack([np.asarray(data[int(i)], dtype=np.float32) for i in index])

    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None
            self._pid = None

    def __del__(self):
        self.close()
