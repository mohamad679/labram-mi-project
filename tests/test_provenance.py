"""Pin exact pretrained source revisions and reject floating model refs."""
import pytest

from src.provenance import validate_pretrained_reference


def test_pinned_revision_is_valid():
    validate_pretrained_reference("braindecode/labram-pretrained", "0" * 40)


@pytest.mark.parametrize("revision", [None, "main", "v1", "abc", "x" * 40])
def test_unpinned_revision_rejected(revision):
    with pytest.raises(ValueError, match="pinned 40-character"):
        validate_pretrained_reference("braindecode/labram-pretrained", revision)


def test_missing_repo_id_rejected():
    with pytest.raises(ValueError, match="repo_id"):
        validate_pretrained_reference(None, "a" * 40)
