"""Structural regression checks for the runnable Kaggle experiment notebook.

No Kaggle, GPU, dataset download or nbformat package is required.
"""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path


NOTEBOOK = Path(__file__).resolve().parents[1] / "run_kaggle.ipynb"


def _load():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _source(cell):
    source = cell["source"]
    return "".join(source) if isinstance(source, list) else source


def test_notebook_is_valid_version_four_and_has_no_committed_outputs():
    notebook = _load()
    assert notebook["nbformat"] == 4
    assert notebook["nbformat_minor"] >= 5
    assert len(notebook["cells"]) >= 15
    for cell in notebook["cells"]:
        assert cell["cell_type"] in {"markdown", "code"}
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None
            assert not cell.get("outputs", [])


def test_all_python_cells_are_parseable_without_notebook_magics():
    for i, cell in enumerate(_load()["cells"]):
        if cell["cell_type"] == "code":
            ast.parse(_source(cell), filename=f"run_kaggle.ipynb#cell-{i}")


def test_notebook_pins_a_commit_and_supports_both_experiments():
    source = "\n".join(_source(cell) for cell in _load()["cells"] if cell["cell_type"] == "code")
    match = re.search(r'SOURCE_COMMIT\s*=\s*"([a-f0-9]{40})"', source)
    assert match is not None
    assert "pretrained_labram.yaml" in source
    assert "baseline.yaml" in source
    assert "SPLIT_SEED = 42" in source
    assert "torch.cuda.is_available()" in source
    assert "heldout_predictions.npz" in source


def test_notebook_exports_real_metrics_without_embedding_fake_results():
    source = "\n".join(_source(cell) for cell in _load()["cells"] if cell["cell_type"] == "code")
    assert "scripts.build_benchmark" in source
    assert "scripts.subject_confidence" in source
    assert "training_history.csv" in source
    assert "results_only_" in source
    assert "source_commit" in source
    assert "metrics[\"accuracy\"]" in source
