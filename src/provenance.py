"""Pinned checkpoint identity and fail-closed temporal compatibility checks."""
import re

OFFICIAL_REPO_ID = "braindecode/labram-pretrained"
OFFICIAL_REVISION = "0563b6c626e7b40d9a36653b763715db94d945d7"
OFFICIAL_WEIGHT_SHA256 = "53b752edb366fd6395dd3cb7d63ae3e1e16aabed040aa0901d64ef32e8f444f8"
OFFICIAL_N_TIMES = 3000


def validate_pretrained_reference(repo_id: str | None, revision: str | None) -> None:
    if not repo_id or not re.fullmatch(r"[\w.-]+/[\w.-]+", repo_id, flags=re.ASCII):
        raise ValueError("A Hugging Face repo_id in owner/name form is required.")
    if not revision or not re.fullmatch(r"[a-fA-F0-9]{40}", revision):
        raise ValueError(
            "A pinned 40-character Hugging Face commit SHA is required; "
            "a branch/tag like 'main' is not an auditable pretrained revision."
        )


def validate_pretrained_window(repo_id: str, revision: str, n_times: int) -> None:
    """Prevent treating incompatible temporal embedding dimensions as fine-tuning."""
    validate_pretrained_reference(repo_id, revision)
    if repo_id == OFFICIAL_REPO_ID and revision == OFFICIAL_REVISION:
        if n_times != OFFICIAL_N_TIMES:
            raise ValueError(
                f"The pinned official LaBraM checkpoint was configured for "
                f"{OFFICIAL_N_TIMES} samples, but this run requests {n_times}. "
                "A validated temporal-embedding adaptation and complete "
                "experiment are required before 400-sample pretrained tuning."
            )
