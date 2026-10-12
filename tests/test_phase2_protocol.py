"""Offline negative controls for the published PhysioNet event/run contract."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.channels import PHYSIONET_64_CH_NAMES
from src.eeg_protocol import (
    MOABB_RUN_ORDER,
    audit_metadata_labels,
    expected_label,
    run_id_from_moabb_key,
    validate_paradigm_defaults,
    validate_raw_annotations,
    validate_window,
)


@pytest.mark.parametrize(
    ("run", "t1", "t2"),
    [(4, "left_hand", "right_hand"), (8, "left_hand", "right_hand"),
     (12, "left_hand", "right_hand"), (6, "hands", "feet"),
     (10, "hands", "feet"), (14, "hands", "feet")],
)
def test_source_event_mapping_by_real_edf_run(run, t1, t2):
    assert expected_label(run, "T0") == "rest"
    assert expected_label(run, "T1") == t1
    assert expected_label(run, "T2") == t2


def test_moabb_run_order_is_not_edf_numeric_order():
    assert tuple(run_id_from_moabb_key(i) for i in range(6)) == MOABB_RUN_ORDER
    with pytest.raises(ValueError, match="run key"):
        run_id_from_moabb_key("6")


def test_executed_runs_or_unknown_events_rejected():
    for wrong_run in (1, 3, 5, 7, 9, 11, 13):
        with pytest.raises(ValueError, match="imagined"):
            expected_label(wrong_run, "T1")
    with pytest.raises(ValueError, match="event"):
        expected_label(4, "T3")


def test_raw_annotations_agree_only_if_run_specific_mapping_is_correct():
    source = ["T0", "T1", "T2", "T0", "T1", "T2"]
    result = validate_raw_annotations(
        4, source, ["rest", "left_hand", "right_hand", "rest", "left_hand", "right_hand"]
    )
    assert result["unmatched_source_events"] == 0
    with pytest.raises(ValueError, match="contradicts"):
        validate_raw_annotations(4, source, ["rest", "hands", "right_hand", "rest", "left_hand", "right_hand"])
    with pytest.raises(ValueError, match="dropped"):
        validate_raw_annotations(4, source, ["rest"])
    with pytest.raises(ValueError, match="annotations"):
        validate_raw_annotations(4, ["T0", "T9"], ["rest"])


def _example_run_metadata():
    labels, subject, runs = [], [], []
    for key, run in enumerate(MOABB_RUN_ORDER):
        if run in (4, 8, 12):
            sequence = ["rest", "left_hand", "right_hand"]
        else:
            sequence = ["rest", "hands", "feet"]
        labels.extend(sequence)
        subject.extend([1] * 3)
        runs.extend([str(key)] * 3)
    # Non-consecutive index proves the audit uses row position, not pandas index.
    metadata = pd.DataFrame(
        {"subject": subject, "run": runs, "session": ["0"] * len(runs)},
        index=np.arange(10, 10 + len(runs)),
    )
    return np.asarray(labels), metadata


def test_per_run_labels_match_exact_cue_family():
    labels, metadata = _example_run_metadata()
    report = audit_metadata_labels(labels, metadata, 1)
    assert set(report) == set(MOABB_RUN_ORDER)
    labels[0] = "feet"
    with pytest.raises(ValueError, match="impossible"):
        audit_metadata_labels(labels, metadata, 1)


def test_reject_subject_mix_and_missing_run():
    labels, metadata = _example_run_metadata()
    metadata.loc[10, "subject"] = 2
    with pytest.raises(ValueError, match="mixes"):
        audit_metadata_labels(labels, metadata, 1)
    labels, metadata = _example_run_metadata()
    with pytest.raises(ValueError, match="Six expected"):
        audit_metadata_labels(labels[:-3], metadata.iloc[:-3], 1)


def _epochs(times):
    return SimpleNamespace(
        ch_names=list(PHYSIONET_64_CH_NAMES),
        info={"sfreq": 160.0},
        times=np.asarray(times),
    )


def test_crop_window_checks_physical_seconds():
    result = validate_window(_epochs(np.arange(481) / 160), 400)
    assert result["cropped_samples"] == 400
    assert result["duration_samples_s"] == 2.5
    assert result["cropped_end_s"] == pytest.approx(2.49375)
    with pytest.raises(ValueError, match="Epoch does not start"):
        validate_window(_epochs(np.arange(481) / 160 + 0.5), 400)
    with pytest.raises(ValueError, match="source epoch length"):
        validate_window(_epochs(np.arange(400) / 160), 400)


def test_moabb_preprocessing_defaults_are_enforced():
    defaults = SimpleNamespace(filters=[[8, 32]], tmin=0.0, tmax=None)
    assert validate_paradigm_defaults(defaults)["bandpass_hz"] == [8, 32]
    for invalid in (
        SimpleNamespace(filters=[[7, 35]], tmin=0.0, tmax=None),
        SimpleNamespace(filters=[[8, 32]], tmin=1.0, tmax=None),
        SimpleNamespace(filters=[[8, 32]], tmin=0.0, tmax=3.0),
    ):
        with pytest.raises(ValueError):
            validate_paradigm_defaults(invalid)


def test_raw_annotation_internal_drop_is_rejected():
    original = ["T0", "T1", "T2", "T0", "T1", "T2"]
    with pytest.raises(ValueError, match="contradicts"):
        validate_raw_annotations(
            4, original, ["rest", "left_hand", "rest", "left_hand", "right_hand"],
        )
