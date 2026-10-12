"""Versioned PhysioNet EEGMMIDB motor-imagery event/window contract.

MOABB 1.7.2 PhysionetMI loads hand-imagery runs before feet-imagery
runs, yielding metadata run keys 0..5 rather than EDF file run numbers.
PhysioNet T1/T2 semantics depend on the EDF run.
"""
from __future__ import annotations

from collections import Counter
from itertools import combinations

import numpy as np

from src.channels import verify_channel_order, verify_sampling_rate

HAND_RUNS = (4, 8, 12)
FOOT_RUNS = (6, 10, 14)
MOABB_RUN_ORDER = HAND_RUNS + FOOT_RUNS
RAW_EVENT_CODES = frozenset(("T0", "T1", "T2"))
CLASSES = frozenset(("feet", "hands", "left_hand", "rest", "right_hand"))
FREQ_BAND_HZ = (8.0, 32.0)
EPOCH_WINDOW_S = (0.0, 3.0)
TARGET_SAMPLES = 400
PATCH_SIZE = 200


def expected_label(run: int, annotation: str) -> str:
    """Map an original EDF annotation using the source's physical run ID."""
    if run not in MOABB_RUN_ORDER:
        raise ValueError(f"Not one of the six imagined movement runs: {run}")
    if annotation not in RAW_EVENT_CODES:
        raise ValueError(f"Unexpected EDF event code: {annotation!r}")
    if annotation == "T0":
        return "rest"
    if run in HAND_RUNS:
        return "left_hand" if annotation == "T1" else "right_hand"
    return "hands" if annotation == "T1" else "feet"


def run_id_from_moabb_key(key: object) -> int:
    """MOABB metadata uses contiguous zero-based keys assigned in hand-first order."""
    text = str(key)
    if text not in ("0", "1", "2", "3", "4", "5"):
        raise ValueError(f"Unexpected MOABB run key: {key!r}")
    return MOABB_RUN_ORDER[int(text)]


def audit_metadata_labels(labels, metadata, subject: int) -> dict:
    """Verify every retained epoch label is possible for its actual EDF run."""
    required = {"subject", "run", "session"}
    if not required.issubset(metadata.columns):
        raise ValueError(f"Missing MOABB metadata fields: {sorted(required - set(metadata.columns))}")
    if len(labels) != len(metadata) or not len(labels):
        raise ValueError("EEG labels and metadata must be nonempty and aligned.")
    if set(map(str, metadata["subject"])) != {str(subject)}:
        raise ValueError("Metadata mixes EEG participant IDs.")
    if any(str(value) not in CLASSES for value in labels):
        raise ValueError("Unknown motor-imagery class in MOABB epochs.")

    output = {}
    indexed = metadata.copy()
    indexed["_audit_position"] = np.arange(len(indexed))
    for key, group in indexed.groupby("run", sort=False):
        run = run_id_from_moabb_key(key)
        actual = Counter(map(str, np.asarray(labels)[group["_audit_position"].to_numpy()]))
        allowed = (
            {"rest", "left_hand", "right_hand"} if run in HAND_RUNS
            else {"rest", "hands", "feet"}
        )
        if not set(actual).issubset(allowed):
            raise ValueError(f"Run {run} has impossible event labels: {dict(actual)}")
        if set(actual) != allowed:
            raise ValueError(f"Run {run} is missing expected task/rest classes: {dict(actual)}")
        output[run] = dict(sorted(actual.items()))
    if set(output) != set(MOABB_RUN_ORDER):
        raise ValueError("Six expected MI runs are not present.")
    return output


def validate_window(epochs, cropped_samples: int) -> dict:
    """Require explicit 3.0-s onset-aligned source and 2.5-s prefix crop."""
    verify_channel_order(epochs.ch_names)
    sfreq = float(epochs.info["sfreq"])
    verify_sampling_rate(sfreq)
    times = np.asarray(epochs.times)
    if times.ndim != 1 or len(times) != 481:
        raise ValueError(f"Unexpected source epoch length: {len(times)}")
    if not np.isclose(times[0], EPOCH_WINDOW_S[0], atol=1e-9):
        raise ValueError("Epoch does not start at event onset.")
    if not np.isclose(times[-1], EPOCH_WINDOW_S[1], atol=1e-9):
        raise ValueError("Expected original 0..3-second epoch.")
    if cropped_samples != TARGET_SAMPLES or cropped_samples % PATCH_SIZE:
        raise ValueError("Unexpected cropped LaBraM input window.")
    return {
        "original_samples": len(times),
        "cropped_samples": cropped_samples,
        "original_start_s": float(times[0]),
        "original_end_s": float(times[-1]),
        "cropped_end_s": float(times[cropped_samples - 1]),
        "duration_samples_s": cropped_samples / sfreq,
        "sample_rate_hz": sfreq,
        "default_moabb_bandpass_hz": list(FREQ_BAND_HZ),
    }


def validate_raw_annotations(run: int, descriptions, observed) -> dict:
    """Check MOABB labels form an ordered, nearly complete raw-EDF event sequence.

    MOABB may reject edge or other source events. Record all mathematically
    possible omitted annotation indices, including ambiguous cases; never
    silently claim the omitted event was the final annotation.
    """
    raw_codes = [str(code) for code in descriptions]
    if not raw_codes or set(raw_codes) - RAW_EVENT_CODES:
        raise ValueError(f"Unexpected source EDF annotations for run {run}")
    expected = [expected_label(run, code) for code in raw_codes]
    observed = list(map(str, observed))
    if not observed:
        raise ValueError("No MOABB epochs for raw EDF run.")
    discarded = len(expected) - len(observed)
    if discarded < 0 or discarded > 2:
        raise ValueError(f"Too many dropped/extra events in run {run}: {discarded}")
    # List all exact deletion alignments; repeated rest events may leave
    # ambiguous candidates. Do not assume a deletion occurred at the end.
    candidate_deletions = []
    for omitted in combinations(range(len(expected)), discarded):
        keep = [label for i, label in enumerate(expected) if i not in omitted]
        if keep == observed:
            candidate_deletions.append(list(omitted))
    if not candidate_deletions:
        raise ValueError(f"MOABB label sequence contradicts original run {run} EDF annotations.")
    if not {"T0", "T1", "T2"}.issubset(raw_codes):
        raise ValueError(f"Original EDF run {run} lacks required event codes.")
    trailing_indices = list(range(len(raw_codes) - discarded, len(raw_codes)))
    return {
        "edf_run": run,
        "source_annotation_count": len(raw_codes),
        "moabb_epoch_count": len(observed),
        "unmatched_source_events": discarded,
        "possible_missing_annotation_indices": candidate_deletions,
        "missing_event_alignment_unique": len(candidate_deletions) == 1,
        "trailing_event_drop_possible": trailing_indices in candidate_deletions,
        "ordered_label_agreement": True,
    }


def validate_paradigm_defaults(paradigm) -> dict:
    """Verify actual MOABB 1.7.2 preprocessing parameters, not only docs."""
    filters = np.asarray(paradigm.filters, dtype=float)
    if filters.shape != (1, 2) or not np.allclose(
        filters[0], FREQ_BAND_HZ, rtol=0, atol=1e-10
    ):
        raise ValueError(f"Unexpected MOABB bandpass filters: {paradigm.filters}")
    if not np.isclose(float(paradigm.tmin), 0.0):
        raise ValueError("Unexpected MOABB epoch start offset.")
    if paradigm.tmax is not None:
        raise ValueError("Expected dataset-defined interval end (3 seconds).")
    return {"bandpass_hz": list(map(float, filters[0])),
            "tmin_s": float(paradigm.tmin), "tmax": "dataset_interval"}
