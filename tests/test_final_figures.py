"""Regression checks: archived figures cannot drift from independently audited JSON."""
from pathlib import Path
from xml.etree import ElementTree

import pytest

from scripts.render_final_figures import FIGURE_DIR, generate, metrics

ROOT = Path(__file__).resolve().parents[1]


def test_six_charts_are_byte_for_byte_synced_with_real_evidence():
    figures = generate(ROOT)
    assert len(figures) == 6
    for name, content in figures.items():
        assert (ROOT / FIGURE_DIR / name).read_text(encoding="utf-8") == content


def test_every_svg_is_well_formed_accessible_and_has_source_notes():
    for name, content in generate(ROOT).items():
        tree = ElementTree.fromstring(content)
        assert tree.tag.endswith("svg"), name
        children = list(tree)
        assert any(child.tag.endswith("title") for child in children)
        assert any(child.tag.endswith("desc") for child in children)
        assert "aria-labelledby=" in content


def test_data_sources_attest_no_test_and_single_one_time_validation():
    p3, p4, p5, oof, final = metrics(ROOT)
    assert final["design"]["test_waveform_inference"] is False
    assert final["design"]["validation_evaluations_reported"] == 1
    assert oof["selection"]["selected"] == "eegnet_epoch_standardized"
    assert p5["models"]["full"]["class_collapse_to_rest"] is True
    assert p3["methods"]["eegnet"]["macro_f1"] > final["results"]["macro_f1"]
    assert p4["results"]["partial"]["macro_f1"] < final["results"]["macro_f1"]


def test_missing_real_evidence_fails_closed(tmp_path):
    with pytest.raises(FileNotFoundError):
        generate(tmp_path)
