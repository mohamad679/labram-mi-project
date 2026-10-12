"""Phase 2 independent real-EDF to MOABB and partial-HDF5 contract audit.

No EEG waveform, per-epoch signal, or trained model is uploaded. The report
contains only cohort-level counts and numerical consistency summaries.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

import mne
import numpy as np
from moabb.datasets import PhysionetMI
from moabb.paradigms import MotorImagery

from scripts.prepare_physionet_s3 import edf_path
from src.channels import verify_channel_order
from src.data import load_physionet_mi
from src.eeg_protocol import (
    MOABB_RUN_ORDER,
    audit_metadata_labels,
    run_id_from_moabb_key,
    validate_raw_annotations,
    validate_window,
)
from src.portable_eeg import (
    H5SignalSource,
    append_subject,
    initialize_cache,
    inspect_cache,
)


def audit_subject(subject: int, *, root: Path, cache: Path) -> dict:
    dataset = PhysionetMI(imagined=True, executed=False)
    paradigm = MotorImagery()
    epochs, labels_epoch, metadata_epoch = paradigm.get_data(
        dataset=dataset, subjects=[subject], return_epochs=True,
    )
    X, labels, metadata = load_physionet_mi(subjects=[subject])
    if not np.array_equal(np.asarray(labels), np.asarray(labels_epoch)):
        raise RuntimeError(f"Subject {subject} labels differ between loader and MOABB.")
    for field in ("subject", "session", "run"):
        if not np.array_equal(
            metadata[field].astype(str).to_numpy(),
            metadata_epoch[field].astype(str).to_numpy(),
        ):
            raise RuntimeError(f"Subject {subject}: metadata mismatch in {field}.")
    if X.shape != (len(labels), 64, 400) or not np.isfinite(X).all():
        raise RuntimeError(f"Invalid EEG tensor for subject {subject}.")
    window = validate_window(epochs, cropped_samples=X.shape[-1])
    counts_by_run = audit_metadata_labels(labels, metadata, subject)
    if not np.isclose(float(metadata.attrs["scaling_factor_from_volts"]), 1e6):
        raise RuntimeError("Unexpected MOABB scaling convention.")
    if metadata.attrs["input_unit"] != "uV":
        raise RuntimeError("Unexpected loader unit.")

    raw_checks = []
    for key in sorted(metadata["run"].astype(str).unique(), key=int):
        run = run_id_from_moabb_key(key)
        path = edf_path(root, subject, run)
        if not path.is_file():
            raise FileNotFoundError(path)
        raw = mne.io.read_raw_edf(path, preload=False, verbose="ERROR")
        verify_channel_order(raw.ch_names)
        observed = np.asarray(labels)[metadata["run"].astype(str).to_numpy() == key]
        audit = validate_raw_annotations(run, raw.annotations.description, observed)
        audit["class_counts"] = counts_by_run[run]
        raw_checks.append(audit)
        raw.close()
    if {item["edf_run"] for item in raw_checks} != set(MOABB_RUN_ORDER):
        raise RuntimeError("Imagery EDF run coverage incomplete.")

    append_subject(cache, subject, X, labels)
    label_counts = dict(sorted(Counter(map(str, labels)).items()))
    return {
        "subject_id": subject,
        "epoch_count": len(labels),
        "class_counts": label_counts,
        "window": window,
        "raw_annotation_consistency": raw_checks,
        "all_signal_values_finite": True,
        "channel_order_verified": True,
    }


def compare_hdf5(cache: Path, expected: dict[int, tuple[np.ndarray, np.ndarray]]) -> dict:
    labels_h5, subjects_h5, shape = inspect_cache(cache, require_complete=False)
    if set(subjects_h5) != set(expected):
        raise ValueError("HDF5 subject identity differs from appended records.")
    signal = H5SignalSource(cache, shape)
    compared = 0
    try:
        for subject, (X, labels) in expected.items():
            positions = np.flatnonzero(subjects_h5 == subject)
            if len(positions) != len(X):
                raise ValueError(f"Incorrect HDF5 epoch count for subject {subject}.")
            if not np.array_equal(labels_h5[positions], np.asarray(labels, dtype=str)):
                raise ValueError(f"HDF5 label mismatch for subject {subject}.")
            for offset in range(0, len(positions), 16):
                indices = positions[offset:offset + 16]
                actual = signal[indices]
                reference = np.asarray(X[offset:offset + len(indices)], dtype=np.float32)
                if not np.array_equal(actual, reference):
                    raise ValueError(f"EEG HDF5 differs from original MOABB array: {subject}.")
                compared += len(indices)
    finally:
        signal.close()
    return {"compared_epochs": compared, "identical_float32_signal_samples": True,
            "exact_class_labels_and_subject_ids": True, "shape": list(shape),
            "partial_validation_cache_only": True}


def run_audit(root: Path, output: Path, *, subjects: tuple[int, ...] = (1, 2)):
    if not subjects or len(subjects) != len(set(subjects)):
        raise ValueError("Subject list must contain unique values.")
    os.environ["MOABB_DOWNLOAD_PROVIDER"] = "upstream"
    os.environ["MNE_DATA"] = str(root.resolve())
    os.environ["MNE_DATASETS_EEGBCI_PATH"] = str(root.resolve())
    output.parent.mkdir(parents=True, exist_ok=True)
    cache = output.parent / "phase2_public_eeg_temporary.h5"
    if cache.exists():
        raise FileExistsError("Refuse to reuse unverified Phase 2 validation cache.")
    initialize_cache(cache)

    per_subject = []
    expected = {}
    try:
        for subject in subjects:
            print(f"PHASE2 auditing source EDF and MOABB mapping for subject {subject}",
                  flush=True)
            per_subject.append(audit_subject(subject, root=root, cache=cache))
            # Independent partial-cache readback uses the exact loader outputs.
            X, labels, _ = load_physionet_mi(subjects=[subject])
            expected[subject] = (X, labels)
        hdf5 = compare_hdf5(cache, expected)
    finally:
        cache.unlink(missing_ok=True)

    report = {
        "status": "passed",
        "scope": "Independent run-annotation checks and two-subject partial-HDF5 parity; "
                 "NOT a full-cohort audit or performance benchmark",
        "dataset": "PhysioNet EEGMMIDB 1.0.0",
        "imagery_only": True,
        "executed_movement_included": False,
        "source_edf_runs": list(MOABB_RUN_ORDER),
        "subjects_checked": list(subjects),
        "subjects": per_subject,
        "cache_parity": hdf5,
        "excluded_subjects_not_audited": [88, 89, 92, 100, 104],
        "no_test_set_model_selection": True,
    }
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mne-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/phase2/contract.json"))
    args = parser.parse_args()
    run_audit(args.mne_root, args.output)


if __name__ == "__main__":
    main()
