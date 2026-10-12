from pathlib import Path

import pytest

from src.config import load_training_config

BASE = Path(__file__).resolve().parents[1] / "configs" / "baseline.yaml"

def test_baseline_values():
    cfg = load_training_config(BASE)
    assert (cfg.batch_size, cfg.epochs, cfg.learning_rate, cfg.patience) == (32, 30, .001, 5)
    assert cfg.model_init == "scratch"

def test_unknown_key_is_rejected(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text(BASE.read_text() + "\nunknown_parameter: 5\n")
    with pytest.raises(ValueError, match="Unknown training"):
        load_training_config(p)

def test_pretrained_requires_source(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text(BASE.read_text().replace("model_init: scratch", "model_init: pretrained"))
    with pytest.raises(ValueError, match="pretrained_repo_id"):
        load_training_config(p)
