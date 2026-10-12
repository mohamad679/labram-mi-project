"""Fail-closed checkpoint and prediction provenance checks.

Stores checksums and split identity, not EEG signal samples.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

SCHEMA_VERSION = 1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_subject_partitions(metadata: dict) -> None:
    required = ("train_subjects", "validation_subjects", "test_subjects")
    if all(key in metadata for key in required):
        groups = [metadata[key] for key in required]
        if any(not isinstance(group, list) or not group for group in groups):
            raise ValueError("Subject split lists must be nonempty.")
    elif {"training_subject_ids", "forward_subject_ids"} <= metadata.keys():
        groups = [metadata["training_subject_ids"], metadata["forward_subject_ids"]]
        if any(not isinstance(group, list) or not group for group in groups):
            raise ValueError("Diagnostic subject lists must be nonempty.")
    else:
        raise ValueError("Manifest requires subject-split or diagnostic participant IDs.")
    sets = [set(group) for group in groups]
    if any(len(group) != len(unique) or any(type(value) is not int for value in group)
           for group, unique in zip(groups, sets)):
        raise ValueError("Subjects must be unique integer IDs within each partition.")
    if any(sets[i] & sets[j] for i in range(len(sets)) for j in range(i + 1, len(sets))):
        raise ValueError("Subject leakage in artifact metadata.")


def write_experiment_manifest(
    checkpoint_path: str | Path, metadata: dict,
    predictions_path: str | Path | None = None,
) -> Path:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint missing: {checkpoint_path}")
    validate_subject_partitions(metadata)
    manifest_path = checkpoint_path.with_suffix(".json")
    if manifest_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing manifest: {manifest_path}")
    files = {"checkpoint": checkpoint_path}
    if predictions_path is not None:
        path = Path(predictions_path)
        if not path.is_file():
            raise FileNotFoundError(f"Predictions missing: {path}")
        files["predictions"] = path
    artifacts = {}
    for kind, path in files.items():
        if path.parent.resolve() != manifest_path.parent.resolve():
            raise ValueError("Artifacts must share a directory with their manifest.")
        artifacts[kind] = {"file": path.name, "sha256": sha256_file(path),
                           "bytes": path.stat().st_size}
    manifest = {"schema_version": SCHEMA_VERSION, "artifacts": artifacts,
                "metadata": metadata}
    pending = manifest_path.with_suffix(".json.tmp")
    if pending.exists():
        raise FileExistsError(f"Pending manifest already exists: {pending}")
    with pending.open("x", encoding="utf-8") as out:
        json.dump(manifest, out, indent=2, sort_keys=True)
        out.write("\n")
    pending.replace(manifest_path)
    verify_experiment_manifest(manifest_path)
    return manifest_path


def verify_experiment_manifest(manifest_path: str | Path) -> dict:
    manifest_path = Path(manifest_path)
    with manifest_path.open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unknown artifact manifest schema.")
    validate_subject_partitions(manifest["metadata"])
    files = manifest.get("artifacts", {})
    if "checkpoint" not in files or set(files) - {"checkpoint", "predictions"}:
        raise ValueError("Invalid artifact inventory.")
    for info in files.values():
        name = info.get("file")
        if not isinstance(name, str) or not name or Path(name).name != name:
            raise ValueError("Invalid artifact filename.")
        path = manifest_path.parent / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing recorded artifact: {path}")
        if path.stat().st_size != info.get("bytes") or sha256_file(path) != info.get("sha256"):
            raise ValueError(f"Artifact integrity mismatch: {path.name}")
    return manifest
