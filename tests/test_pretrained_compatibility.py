"""Check pretrained source compatibility without model downloads or EEG data."""
from pathlib import Path

import pytest

from src.config import load_training_config
from src.provenance import (
    OFFICIAL_N_TIMES,
    OFFICIAL_REPO_ID,
    OFFICIAL_REVISION,
    OFFICIAL_WEIGHT_SHA256,
    validate_pretrained_window,
)


def test_official_checkpoint_identity_is_immutable():
    assert len(OFFICIAL_REVISION) == 40
    assert len(OFFICIAL_WEIGHT_SHA256) == 64
    assert OFFICIAL_REPO_ID == "braindecode/labram-pretrained"
    assert OFFICIAL_N_TIMES == 3000


def test_official_checkpoint_accepts_documented_window():
    validate_pretrained_window(OFFICIAL_REPO_ID, OFFICIAL_REVISION, 3000)


def test_official_checkpoint_rejects_400_samples():
    with pytest.raises(ValueError, match="configured for 3000 samples"):
        validate_pretrained_window(OFFICIAL_REPO_ID, OFFICIAL_REVISION, 400)


def test_pretrained_400_config_fails_before_dataset_download(tmp_path):
    baseline = Path(__file__).resolve().parents[1] / "configs" / "baseline.yaml"
    source = baseline.read_text(encoding="utf-8")
    source = source.replace("model_init: scratch", "model_init: pretrained")
    source = source.replace("pretrained_repo_id: null", f"pretrained_repo_id: {OFFICIAL_REPO_ID}")
    source = source.replace(
        "pretrained_revision: null",
        f"pretrained_revision: {OFFICIAL_REVISION}",
    )
    path = tmp_path / "incompatible.yaml"
    path.write_text(source, encoding="utf-8")
    with pytest.raises(ValueError, match="3000 samples"):
        load_training_config(path)
