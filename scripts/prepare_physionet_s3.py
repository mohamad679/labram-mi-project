"""Stream PhysioNet MI's six default imagery EDF runs from public AWS S3.

Creates a portable schema-3 HDF5 cache with per-subject durable commits.
Intended to run under Kaggle Python 3.11 with boto3, MOABB, MNE, h5py and tqdm.
No AWS credentials, PyTorch, or GPU required.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

RUNS = (4, 6, 8, 10, 12, 14)
BUCKET = "physionet-open"
PREFIX = "eegmmidb/1.0.0"


def s3_key(subject, run):
    return f"{PREFIX}/S{subject:03d}/S{subject:03d}R{run:02d}.edf"


def edf_path(root, subject, run):
    return (Path(root) / "MNE-eegbci-data/files/eegmmidb/1.0.0"
            / f"S{subject:03d}" / f"S{subject:03d}R{run:02d}.edf")


def unsigned_s3():
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config

    return boto3.client("s3", config=Config(
        signature_version=UNSIGNED, connect_timeout=20, read_timeout=90,
        retries={"mode": "standard", "max_attempts": 8},
    ))


def fetch_run(client, subject, run, root):
    key = s3_key(subject, run)
    target = edf_path(root, subject, run)
    expected = int(client.head_object(Bucket=BUCKET, Key=key)["ContentLength"])
    if expected <= 0:
        raise ValueError(f"Empty upstream EDF: {key}")
    if target.is_file() and target.stat().st_size == expected:
        return expected, 0.0
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_suffix(".edf.partial")
    part.unlink(missing_ok=True)
    started = time.monotonic()
    try:
        client.download_file(BUCKET, key, str(part))
        if part.stat().st_size != expected:
            raise IOError(f"Incomplete S3 download: {key}")
        part.replace(target)
    finally:
        part.unlink(missing_ok=True)
    duration = max(time.monotonic() - started, 0.001)
    print(f"  {target.name}: {expected / 2**20 / duration:.2f} MiB/s", flush=True)
    return expected, duration


def saved_subjects(path):
    import h5py

    with h5py.File(path, "r") as handle:
        return set(json.loads(handle.attrs["completed_subjects"]))


def restore_cache(output, previous):
    import h5py
    from src.portable_eeg import initialize_cache, inspect_cache, verify_cache_metadata

    output.parent.mkdir(parents=True, exist_ok=True)
    if previous:
        previous = Path(previous)
        if not previous.is_file():
            raise FileNotFoundError(previous)
        with h5py.File(previous, "r") as handle:
            verify_cache_metadata(handle)
        if not output.exists():
            shutil.copy2(previous, output)
            print(f"RESTORED previous snapshot: {previous}", flush=True)
        else:
            print("Existing writable cache preserved; older input not overwritten.", flush=True)
    initialize_cache(output)
    inspect_cache(output, require_complete=False)


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build(output, mne_root, previous="", max_new_subjects=104,
          retries=5, retry_delay=8):
    from botocore.exceptions import BotoCoreError, ClientError
    from tqdm import tqdm
    from src.data import load_physionet_mi
    from src.portable_eeg import EXPECTED_SUBJECTS, append_subject, inspect_cache

    mne_root = Path(mne_root)
    os.environ["MOABB_DOWNLOAD_PROVIDER"] = "upstream"
    os.environ["MPLBACKEND"] = "Agg"
    os.environ["MNE_DATA"] = str(mne_root.resolve())
    os.environ["MNE_DATASETS_EEGBCI_PATH"] = str(mne_root.resolve())
    output = Path(output)
    restore_cache(output, previous)
    finished = saved_subjects(output)
    client = unsigned_s3()
    print(f"RESUMING {len(finished)}/{len(EXPECTED_SUBJECTS)} verified subjects", flush=True)
    written = 0
    started = time.monotonic()
    with tqdm(total=len(EXPECTED_SUBJECTS), initial=len(finished),
              desc="PhysioNet MI HDF5", unit="subject") as progress:
        for subject in EXPECTED_SUBJECTS:
            if subject in finished or written >= max_new_subjects:
                continue
            print(f"\nSubject {subject:03d}: fetching six imagery runs from S3", flush=True)
            for run in RUNS:
                for attempt in range(retries + 1):
                    try:
                        fetch_run(client, subject, run, mne_root)
                        break
                    except (BotoCoreError, ClientError, OSError) as exc:
                        if isinstance(exc, ClientError) and str(
                            exc.response.get("Error", {}).get("Code", "")
                        ) in {"403", "404", "AccessDenied", "NoSuchKey"}:
                            raise
                        if attempt == retries:
                            raise RuntimeError(
                                f"S3 transfer failed: {s3_key(subject, run)}"
                            ) from exc
                        delay = min(retry_delay * (2 ** attempt), 120)
                        print(f"  Retrying {subject}/{run} in {delay}s: {type(exc).__name__}",
                              flush=True)
                        time.sleep(delay)
            X, y, meta = load_physionet_mi(subjects=[subject])
            if set(map(int, meta["subject"].to_numpy())) != {subject}:
                raise ValueError(f"Invalid metadata for subject {subject}")
            if X.shape[1:] != (64, 400):
                raise ValueError(f"Invalid EEG shape: {X.shape}")
            append_subject(output, subject, X, y)
            del X, y, meta
            if subject not in saved_subjects(output):
                raise RuntimeError(f"Subject {subject} was not durably committed")
            for run in RUNS:
                edf_path(mne_root, subject, run).unlink(missing_ok=True)
            finished.add(subject)
            written += 1
            progress.update()
            elapsed = time.monotonic() - started
            eta_min = elapsed / written * (104 - len(finished)) / 60
            progress.set_postfix_str(f"ETA~{eta_min:.0f}m; {output.stat().st_size / 2**30:.2f}GiB")
    labels, ids, shape = inspect_cache(output, require_complete=False)
    complete = len(finished) == len(EXPECTED_SUBJECTS)
    if complete:
        inspect_cache(output, require_complete=True)
    report = {
        "status": "complete" if complete else "partial",
        "subjects": len(finished), "epochs": len(labels),
        "shape": list(shape), "sha256": digest_file(output),
        "file": str(output), "s3_bucket": BUCKET,
        "s3_prefix": PREFIX, "runs": list(RUNS),
    }
    output.with_name("cache_build_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps(report, indent=2), flush=True)
    if not complete:
        print("PARTIAL: save this HDF5 as a Kaggle output snapshot before session ends.")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mne-root", required=True, type=Path)
    parser.add_argument("--previous", default="")
    parser.add_argument("--max-new-subjects", type=int, default=104)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--retry-delay", type=float, default=8)
    parser.add_argument("--probe", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.max_new_subjects <= 104 or args.retries < 0 or args.retry_delay < 0:
        parser.error("invalid retry or subject count")
    if args.probe:
        size, seconds = fetch_run(unsigned_s3(), 1, 14, args.mne_root)
        if seconds == 0:
            print("S3 probe already cached; remove file to remeasure")
        else:
            print(f"S3 download speed: {size / 2**20 / seconds:.2f} MiB/s")
        return
    build(args.output, args.mne_root, args.previous, args.max_new_subjects,
          args.retries, args.retry_delay)


if __name__ == "__main__":
    main()
