"""Download 104 eligible PhysioNet subjects locally into one resumable HDF5 file.

Example (Python 3.11; no GPU):
 python -m scripts.prepare_portable_eeg --output physionet_mi_104subjects.h5
 python -m scripts.prepare_portable_eeg --output physionet_mi_104subjects.h5 --verify-only

Interrupted builds can be resumed using the same file path. This file
contains EEG waveforms; keep Kaggle datasets private and never commit to Git.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

from src.portable_eeg import (
    EXPECTED_SUBJECTS,
    append_subject,
    initialize_cache,
    inspect_cache,
)



def _load_with_retry(subject, load_subject, *, max_retries: int, retry_delay: float):
    """Retry only temporary network failures, never bad EEG metadata/shape/labels.

    A retry re-enters MOABB's downloader, reusing already complete raw EDF files
    and the durable per-subject HDF5 commits. Pooch discards unsuccessful
    temporary transfers; it does not consider them complete cache files.
    """
    from requests.exceptions import RequestException

    for retry in range(max_retries + 1):
        try:
            return load_subject(subjects=[subject])
        except (RequestException, TimeoutError, ConnectionResetError) as exc:
            if retry == max_retries:
                raise RuntimeError(
                    f"Subject {subject}: network failure after {max_retries + 1} "
                    "attempts. Re-run the same command to resume safely."
                ) from exc
            delay = min(120.0, retry_delay * 2**retry)
            print(
                f"NETWORK RETRY: subject={subject}, attempt={retry + 1}/"
                f"{max_retries + 1}, error={type(exc).__name__}. "
                f"Reconnecting in {delay:.0f}s. Already saved subjects are safe.",
                flush=True,
            )
            time.sleep(delay)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("physionet_mi_104subjects.h5"))
    parser.add_argument(
        "--mne-data", type=Path, default=Path.home() / "mne_data",
        help="Persistent local raw EDF download directory (separate from final HDF5)",
    )
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--max-retries", type=int, default=8, help="Network retries per failed subject (default: 8)")
    parser.add_argument("--retry-delay", type=float, default=15.0, help="Initial reconnect delay in seconds; exponential backoff up to 120s")
    parser.add_argument("--sha256", action="store_true", help="Calculate full file hash after completion")
    args = parser.parse_args()
    if args.max_retries < 0 or args.retry_delay < 0:
        parser.error("--max-retries and --retry-delay must be nonnegative")
    os.environ["MOABB_DOWNLOAD_PROVIDER"] = "upstream"
    os.environ["MPLBACKEND"] = "Agg"
    args.mne_data.mkdir(parents=True, exist_ok=True)
    os.environ["MNE_DATA"] = str(args.mne_data.resolve())
    os.environ["MNE_DATASETS_EEGBCI_PATH"] = str(args.mne_data.resolve())

    if args.verify_only:
        if not args.output.is_file():
            raise FileNotFoundError(args.output)
        y, subjects, shape = inspect_cache(args.output, require_complete=True)
        print(f"VERIFIED COMPLETE: {len(set(subjects))} subjects, {len(y)} epochs, X={shape}")
    else:
        initialize_cache(args.output)
        import h5py
        from tqdm import tqdm

        from src.data import load_physionet_mi

        with h5py.File(args.output, "r") as handle:
            finished = set(json.loads(handle.attrs["completed_subjects"]))
        print(
            f"RESUMING: {len(finished)}/{len(EXPECTED_SUBJECTS)} subjects "
            f"already committed to {args.output.resolve()}",
            flush=True,
        )
        with tqdm(total=len(EXPECTED_SUBJECTS), initial=len(finished),
                  desc="PhysioNet subjects", unit="subject", dynamic_ncols=True) as progress:
            for subject in EXPECTED_SUBJECTS:
                if subject in finished:
                    continue
                progress.set_postfix_str(f"subject={subject}")
                X, y, metadata = _load_with_retry(
                    subject, load_physionet_mi,
                    max_retries=args.max_retries, retry_delay=args.retry_delay,
                )
                observed = set(map(int, metadata["subject"].to_numpy()))
                if observed != {subject}:
                    raise RuntimeError(f"Wrong subject labels for {subject}: {observed}")
                append_subject(args.output, subject, X, y)
                del X, y, metadata
                finished.add(subject)
                progress.update(1)
        y, subjects, shape = inspect_cache(args.output, require_complete=True)
        print(f"COMPLETE: {len(set(subjects))} subjects, {len(y)} epochs, X={shape}")

    size_gib = args.output.stat().st_size / (1024**3)
    print(f"Single portable HDF5: {args.output.resolve()} ({size_gib:.2f} GiB)")
    if args.sha256:
        hasher = hashlib.sha256()
        with args.output.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                hasher.update(chunk)
        print("SHA256:", hasher.hexdigest())


if __name__ == "__main__":
    main()
