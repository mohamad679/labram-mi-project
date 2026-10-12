"""Pretrained source validation — never silently resolve a moving Hub branch."""
import re


def validate_pretrained_reference(repo_id: str | None, revision: str | None) -> None:
    if not repo_id or not re.fullmatch(r"[\w.-]+/[\w.-]+", repo_id, flags=re.ASCII):
        raise ValueError("A Hugging Face repo_id in owner/name form is required.")
    if not revision or not re.fullmatch(r"[a-fA-F0-9]{40}", revision):
        raise ValueError(
            "A pinned 40-character Hugging Face commit SHA is required; "
            "a branch/tag like 'main' is not an auditable pretrained revision."
        )
