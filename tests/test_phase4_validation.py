"""Offline, adversarial Phase 4 contract tests (no EEG or network required)."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.run_phase4_finetune import _safe_atomic_json, run, validation_metrics


def test_full_five_class_validation_scores():
    y = np.arange(2436) % 5
    prob = np.zeros((2436, 5), dtype=np.float32)
    prob[np.arange(2436), y] = 1.0
    score = validation_metrics(y, prob)
    assert score["macro_f1"] == 1.0
    assert score["balanced_accuracy"] == 1.0
    assert sum(map(sum, score["confusion_matrix"])) == 2436


@pytest.mark.parametrize("bad", [
    np.ones((2435, 5)) / 5,
    np.full((2436, 5), np.nan),
    np.ones((2436, 5)) / 3,
    np.ones((2436, 4)) / 4,
])
def test_broken_validation_prediction_inputs_rejected(bad):
    with pytest.raises(ValueError):
        validation_metrics(np.arange(2436) % 5, bad)


def test_missing_or_untrusted_hdf5_fails_before_creating_output(tmp_path):
    args = SimpleNamespace(
        device="cpu", epochs=1, patience=1, bootstrap=10, seed=42,
        batch_size=1, accumulate=1, grad_clip=1.0, head_lr=.001,
        partial_lr=.0001, full_lr=.00001,
        cache=tmp_path / "not_real.h5", output_dir=tmp_path / "out",
    )
    with pytest.raises(ValueError, match="SHA-mismatched"):
        run(args)
    assert not args.output_dir.exists()


def test_fail_closed_overwrite(tmp_path):
    path = tmp_path / "diagnostic.json"
    _safe_atomic_json(path, {"pass": True})
    assert json.loads(path.read_text()) == {"pass": True}
    with pytest.raises(FileExistsError):
        _safe_atomic_json(path, {"pass": False})
    assert json.loads(path.read_text()) == {"pass": True}


def test_source_never_extracts_test_waveform(tmp_path):
    text = Path(__file__).parents[1].joinpath("scripts/run_phase4_finetune.py").read_text()
    assert 'train_rows=split["train"], val_rows=split["validation"]' in text
    assert 'split["test"]' not in text
    assert '"validation_predictions.npz"' in text
