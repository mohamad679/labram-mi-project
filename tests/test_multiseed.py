"""Experiment campaign must preserve held-out subjects across model seeds."""
import pytest
import yaml

from scripts.run_multiseed import make_run_configs


def test_makes_unique_seed_dirs_preserving_split(tmp_path):
    source = tmp_path / "baseline.yaml"
    source.write_text("seed: 42\nsplit_seed: 31\nbatch_size: 32\n")
    paths = make_run_configs(source, tmp_path / "runs", [42, 77])
    configs = [yaml.safe_load(path.read_text()) for path in paths]
    assert [config["seed"] for config in configs] == [42, 77]
    assert all(config["split_seed"] == 31 for config in configs)
    assert len({config["checkpoint_path"] for config in configs}) == 2


def test_refuses_duplicate_seeds(tmp_path):
    with pytest.raises(ValueError, match="unique"):
        make_run_configs(tmp_path / "baseline.yaml", tmp_path / "runs", [42, 42])


def test_requires_a_explicit_split_seed(tmp_path):
    baseline = tmp_path / "baseline.yaml"
    baseline.write_text("seed: 42\n")
    with pytest.raises(ValueError, match="split_seed"):
        make_run_configs(baseline, tmp_path / "runs", [42, 43])
