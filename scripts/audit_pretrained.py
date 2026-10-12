"""Optional network/GPU audit: prove that a pinned Hub weight file was loaded.

Run with a *real* 40-hex revision from the Hugging Face repository.
This command is not run by offline CI and does not train or evaluate EEG data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from src.model import load_labram_model
from src.provenance import validate_pretrained_reference


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as reader:
        for block in iter(lambda: reader.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compare_checkpoint_tensors(model, checkpoint: dict) -> tuple[int, int]:
    """Compare backbone tensors exactly, excluding task-specific classifier."""
    target = model.state_dict()
    match = 0
    eligible = 0
    for name, weight in checkpoint.items():
        if name.startswith(("final_layer.", "interpolation_layer.")):
            continue
        if name not in target or not isinstance(weight, torch.Tensor):
            continue
        if weight.shape != target[name].shape:
            continue
        eligible += 1
        if torch.equal(weight.cpu(), target[name].detach().cpu()):
            match += 1
    return match, eligible


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="braindecode/labram-pretrained")
    parser.add_argument("--revision", required=True, help="Exact 40-character HF commit SHA")
    parser.add_argument("--n-times", type=int, default=400)
    parser.add_argument("--output", default="artifacts/pretrained-audit.json")
    args = parser.parse_args()
    validate_pretrained_reference(args.repo_id, args.revision)

    import braindecode
    import huggingface_hub
    from huggingface_hub import HfApi, hf_hub_download

    info = HfApi().model_info(args.repo_id, revision=args.revision)
    if info.sha != args.revision:
        raise RuntimeError("Hub revision did not resolve to the pinned commit.")
    files = [name for name in HfApi().list_repo_files(
        args.repo_id, revision=args.revision
    ) if name.endswith(".safetensors")]
    if not files:
        raise RuntimeError("No Safetensors file found for a safe, direct weights audit.")
    if len(files) != 1:
        raise RuntimeError("Expected a single Safetensors file; inspect the repo manually.")
    filename = files[0]
    weight_path = Path(hf_hub_download(
        repo_id=args.repo_id, filename=filename, revision=args.revision
    ))
    from safetensors.torch import load_file

    checkpoint = load_file(str(weight_path), device="cpu")
    model = load_labram_model(
        n_outputs=5, n_times=args.n_times, n_chans=64, sfreq=160.0,
        pretrained_repo_id=args.repo_id, pretrained_revision=args.revision,
    )
    matched, eligible = compare_checkpoint_tensors(model, checkpoint)
    if eligible < 10 or matched < 10:
        raise RuntimeError(
            f"Pretraining verification FAILED: {matched}/{eligible} compatible "
            "backbone tensors exactly matched the pinned Hub weights."
        )

    report = {
        "verified_hub_commit": info.sha,
        "repo_id": args.repo_id,
        "weight_file": filename,
        "weight_sha256": file_sha256(weight_path),
        "n_times": args.n_times,
        "n_chans": 64,
        "matching_backbone_tensors": matched,
        "eligible_backbone_tensors": eligible,
        "braindecode_version": braindecode.__version__,
        "huggingface_hub_version": huggingface_hub.__version__,
        "torch_version": torch.__version__,
        "scope": "weights loading only; no EEG data, downstream training or accuracy tested",
    }
    result_path = Path(args.output)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
