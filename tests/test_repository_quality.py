"""Offline tests of repository integrity and local Markdown link resolution."""
from pathlib import Path

from scripts.check_repository_quality import (
    check_markdown_links,
    check_tracked_artifacts,
    markdown_targets,
)


def test_skips_code_fence_links_and_keeps_real_links():
    text = (
        "# Docs\n"
        "```bash\n[example](not-a-real-file.md)\n```\n"
        "[valid](../README.md) and ![photo](assets/photo.png)\n"
    )
    assert markdown_targets(text) == ["../README.md", "assets/photo.png"]


def test_accepts_existing_relative_paths_and_external_urls(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "README.md").write_text("# Home")
    (tmp_path / "docs" / "index.md").write_text(
        "[home](../README.md#hello) [external](https://example.com/a) "
        "[email](mailto:team@example.com)"
    )
    assert check_markdown_links(tmp_path, [Path("docs/index.md")]) == []


def test_detects_broken_links(tmp_path):
    (tmp_path / "README.md").write_text("[missing](not-found.md)")
    errors = check_markdown_links(tmp_path, [Path("README.md")])
    assert len(errors) == 1 and "missing link target" in errors[0]


def test_rejects_links_escaping_checkout(tmp_path):
    (tmp_path / "README.md").write_text("[bad](../../secret.txt)")
    errors = check_markdown_links(tmp_path, [Path("README.md")])
    assert len(errors) == 1 and "escapes repository" in errors[0]


def test_rejects_local_eeg_model_and_secret_paths():
    errors = check_tracked_artifacts([
        Path("data/patient.edf"),
        Path("checkpoints/model.pt"),
        Path(".env"),
        Path("results/figures/chart.png"),
        Path("README.md"),
    ])
    assert len(errors) == 3


def test_file_names_with_url_encoding(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "a figure.png").write_bytes(b"x")
    (tmp_path / "README.md").write_text("![Figure](docs/a%20figure.png)")
    assert check_markdown_links(tmp_path, [Path("README.md")]) == []
