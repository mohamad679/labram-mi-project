"""Build one resumable, portable PhysioNet EEG HDF5 cache on your computer.

Run from repository root with a compatible Python environment:
    python -m pip install -r requirements.txt h5py
    python -m scripts.download_physionet_local --output ~/Downloads/physionet_mi_104subjects.h5

Downloads one eligible subject at a time. Restart the identical command after
interruption: completed subjects are durable and skipped. Do not use a legacy
unverified NPZ; Kaggle reads the resulting verified HDF5 lazily.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

from src.data import load_physionet_mi
from src.portable_eeg import (
    EXPECTED_SUBJECTS,
    append_subject,
    initialize_cache,
    inspect_cache,
)


def _done(path: Path) -> set[int]:
    import h5py

    with h5py.File(path, "r") as f:
        return set(map(int, json.loads(f.attrs["completed_subjects"])))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build(output: Path, subjects: tuple[int, ...] = EXPECTED_SUBJECTS) -> dict:
    os.environ["MOABB_DOWNLOAD_PROVIDER"] = "upstream"
    os.environ["MPLBACKEND"] = "Agg"
    output = output.expanduser().resolve()
    initialize_cache(output)
    completed = _done(output)
    start = time.monotonic()
    n_total = len(subjects)

    for step, subject in enumerate(subjects, 1):
        if subject in completed:
            print(f"[{step}/{n_total}] Subject {subject:03d}: cached — skipped", flush=True)
            continue
        print(f"[{step}/{n_total}] Subject {subject:03d}: loading from PhysioNet", flush=True)
        t0 = time.monotonic()
        X, y, metadata = load_physionet_mi(subjects=[subject])
        found = set(map(int, metadata["subject"].to_numpy()))
        if found != {subject}:
            raise ValueError(f"Downloaded subject mismatch: wanted {subject}, found {found}")
        if X.shape[1:] != (64, 400):
            raise ValueError(f"Unexpected input shape: {X.shape}")
        append_subject(output, subject, X, y)
        completed.add(subject)
        took = time.monotonic() - t0
        remaining = n_total - step
        eta_minutes = (time.monotonic() - start) / max(1, step) * remaining / 60
        print(
            f"[{step}/{n_total}] Subject {subject:03d}: saved {len(y)} epochs "
            f"in {took:.1f}s | {len(completed)}/104 subjects saved | "
            f"ETA approx {eta_minutes:.0f} min | file {output.stat().st_size / 2**30:.2f} GiB",
            flush=True,
        )

    complete = len(completed) == len(EXPECTED_SUBJECTS)
    if complete:
        y, ids, shape = inspect_cache(output, require_complete=True)
        checksum = sha256_file(output)
        report = {
            "file": str(output), "sha256": checksum, "subjects": len(set(ids)),
            "epochs": len(y), "shape": list(shape), "status": "complete",
            "cache_schema": 3,
        }
        report_file = output.with_suffix(".sha256.json")
        report_file.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print("COMPLETE VERIFIED CACHE:", json.dumps(report, indent=2), flush=True)
        print("Save a backup on your computer; upload H5 as a PRIVATE Kaggle Dataset.")
    else:
        report = {"file": str(output), "subjects": len(completed), "status": "partial"}
        print(f"PARTIAL CACHE: {len(completed)}/104 subjects. Resume with the same command.")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=Path.home() / "Downloads" / "physionet_mi_104subjects.h5",
    )
    args = parser.parse_args()
    build(args.output)


if __name__ == "__main__":
    main()
