"""Verify checkpoint/prediction manifests detect corruption and subject leakage."""
import json

import pytest

from src.artifact_integrity import (
    verify_experiment_manifest,
    write_experiment_manifest,
)


def _metadata():
    return {
        "git_commit": "a" * 40,
        "train_subjects": [1, 2],
        "validation_subjects": [3],
        "test_subjects": [4],
        "configuration": {"model_init": "pretrained", "training_strategy": "linear_probe"},
    }


def test_integrity_round_trip(tmp_path):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"small artificial checkpoint for manifest unit test")
    predictions = tmp_path / "model_heldout_predictions.npz"
    predictions.write_bytes(b"test fixture, not real EEG or benchmark predictions")
    manifest = write_experiment_manifest(checkpoint, _metadata(), predictions)
    verified = verify_experiment_manifest(manifest)
    assert verified["artifacts"]["checkpoint"]["bytes"] == checkpoint.stat().st_size
    assert set(verified["artifacts"]) == {"checkpoint", "predictions"}


def test_detect_checkpoint_tampering(tmp_path):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint-original")
    manifest = write_experiment_manifest(checkpoint, _metadata())
    checkpoint.write_bytes(b"checkpoint-corrupted")
    with pytest.raises(ValueError, match="integrity mismatch"):
        verify_experiment_manifest(manifest)


def test_detect_prediction_tampering(tmp_path):
    checkpoint = tmp_path / "model.pt"
    predictions = tmp_path / "preds.npz"
    checkpoint.write_bytes(b"model")
    predictions.write_bytes(b"predictions")
    manifest = write_experiment_manifest(checkpoint, _metadata(), predictions)
    predictions.write_bytes(b"predictions-changed")
    with pytest.raises(ValueError, match="integrity mismatch"):
        verify_experiment_manifest(manifest)


def test_reject_subject_leakage(tmp_path):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"model")
    metadata = _metadata()
    metadata["test_subjects"] = [2]
    with pytest.raises(ValueError, match="Subject leakage"):
        write_experiment_manifest(checkpoint, metadata)


def test_reject_incomplete_manifest(tmp_path):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"model")
    manifest = write_experiment_manifest(checkpoint, _metadata())
    record = json.loads(manifest.read_text())
    record["schema_version"] = 999
    manifest.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="schema"):
        verify_experiment_manifest(manifest)


def test_manifest_is_not_silently_overwritten(tmp_path):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"model")
    write_experiment_manifest(checkpoint, _metadata())
    with pytest.raises(FileExistsError, match="overwrite"):
        write_experiment_manifest(checkpoint, _metadata())


def test_reject_cross_directory_artifact(tmp_path):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"model")
    external = tmp_path / "other" / "predictions.npz"
    external.parent.mkdir()
    external.write_bytes(b"preds")
    with pytest.raises(ValueError, match="share a directory"):
        write_experiment_manifest(checkpoint, _metadata(), external)
