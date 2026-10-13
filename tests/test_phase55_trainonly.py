"""Adversarial data leakage, split, feature and selection tests; no actual EEG needed."""
import numpy as np
import pytest
from sklearn.metrics import f1_score

from src.phase55_trainonly import (
    BANDS, TrainOnlySignalSource, epoch_standardize,
    make_bandpower_features, make_trainonly_folds, score_five_class,
    select_predeclared,
)


def synthetic_manifest():
    subjects = np.repeat(np.arange(1, 105), 2)
    partitions = {"train": np.arange(0, 150), "validation": np.arange(150, 178),
                  "test": np.arange(178, 208)}
    return partitions, subjects


def test_group_folds_cover_every_train_row_once_and_never_touch_test_or_validation():
    partitions, subjects = synthetic_manifest()
    folds = make_trainonly_folds(partitions, subjects)
    assert len(folds) == 3
    held = []
    for fold in folds:
        assert len(fold["fit_subjects"]) == 42
        assert len(fold["early_subjects"]) == 8
        assert len(fold["outer_subjects"]) == 25
        assert not (set(fold["fit_subjects"]) & set(fold["early_subjects"]))
        held.extend(fold["outer_holdout"])
    assert np.array_equal(np.sort(held), partitions["train"])


def test_folds_refuse_overlapping_original_subject_partitions():
    partitions, subjects = synthetic_manifest()
    broken = dict(partitions)
    broken["test"] = partitions["train"][:2]
    with pytest.raises(ValueError):
        make_trainonly_folds(broken, subjects)


def test_folds_refuse_protocol_changes():
    partitions, subjects = synthetic_manifest()
    with pytest.raises(ValueError):
        make_trainonly_folds(partitions, subjects, n_splits=5)
    with pytest.raises(ValueError):
        make_trainonly_folds(partitions, subjects, seed=11)


def test_signal_source_blocks_heldout_waveform_read():
    partitions, _ = synthetic_manifest()
    backing = {1: "safe", 160: "forbidden"}
    guard = TrainOnlySignalSource(backing, range(13050))
    assert guard[1] == "safe"
    with pytest.raises(PermissionError):
        _ = guard[13050]
    with pytest.raises(ValueError):
        _ = guard[np.array([1, 2])]


def test_standardization_preserves_axis_and_removes_per_channel_mean():
    rng = np.random.default_rng(12)
    eeg = (rng.normal(size=(64, 400)) * np.arange(1, 65)[:, None]).astype("f4")
    out = epoch_standardize(eeg)
    assert out.shape == (64, 400) and out.dtype == np.float32
    assert np.max(np.abs(out.mean(-1))) < 1e-5
    assert np.allclose(out.std(-1), 1., atol=1e-5)
    assert np.isfinite(epoch_standardize(np.zeros((64, 400), dtype="f4"))).all()


@pytest.mark.parametrize("shape", [(64, 399), (65, 400), (1, 64, 400)])
def test_standardization_rejects_wrong_shape(shape):
    with pytest.raises(ValueError):
        epoch_standardize(np.zeros(shape))


def test_four_filterbands_produce_finite_256_features_and_preserve_sensor_axis():
    fs = 160
    t = np.arange(400) / fs
    wave = np.zeros((64, 400), dtype="f4")
    wave[3] = np.sin(2 * np.pi * 10 * t)
    feat = make_bandpower_features(wave)
    assert feat.shape == (1, 256) and len(BANDS) == 4
    assert np.isfinite(feat).all()
    assert feat[0, 3] > feat[0, 64 + 3]  # 10 Hz is stronger in 8-12 than 12-16
    with pytest.raises(ValueError):
        make_bandpower_features(np.zeros((64, 400)), sfreq=200.)


def test_score_detects_total_majority_collapse_not_misleading_accuracy():
    y = np.tile(np.arange(5), 5)
    groups = np.repeat(np.arange(5), 5)
    pred = np.zeros((25, 5))
    pred[:, 3] = 1
    out = score_five_class(y, pred, groups)
    assert out["single_class_collapse"] is True
    assert out["balanced_accuracy"] == pytest.approx(0.2)
    assert out["macro_f1"] == pytest.approx(f1_score(y, np.full(25, 3), average="macro"))
    assert out["predicted_class_counts"] == [0, 0, 0, 25, 0]


def test_selector_does_not_reward_collapsed_model_or_unknown_candidates():
    a = {"macro_f1": 0.3, "balanced_accuracy": 0.22, "single_class_collapse": False}
    b = {"macro_f1": 0.9, "balanced_accuracy": 0.2, "single_class_collapse": True}
    out = select_predeclared({"eegnet_raw": a, "bandpower": b})
    assert out["selected"] == "eegnet_raw"
    assert select_predeclared({"bandpower": b})["selected"] is None
    with pytest.raises(ValueError):
        select_predeclared({"unknown": a})


def test_predictions_bad_normalization_fails_closed():
    y = np.tile(np.arange(5), 5)
    groups = np.repeat(np.arange(5), 5)
    bad = np.ones((25, 5), dtype="f4")
    with pytest.raises(ValueError):
        score_five_class(y, bad, groups)


def test_bandpower_regression_uses_train_partition_only():
    from scripts.run_phase55_trainonly_cv import bandpower_fold

    rng = np.random.default_rng(9)
    features = rng.normal(0, 1, (60, 256)).astype('f4')
    y = np.tile(np.arange(5), 12)
    # Artificial but discriminable class cue; validation heldout remains unseen by fitting.
    features[np.arange(60), y] += 2.5
    fold = {"outer_train": np.arange(50), "outer_holdout": np.arange(50, 60),
            "fit_subjects": list(range(42)), "early_subjects": list(range(42, 50))}
    p, info = bandpower_fold(features, y, np.arange(60), fold)
    assert p.shape == (10, 5)
    assert np.allclose(p.sum(1), 1.)
    assert info['scaler_and_logistic_fit_on'] == 'outer_train_only'
