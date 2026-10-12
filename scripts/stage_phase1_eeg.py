"""Stage only the two real PhysioNet participants for Phase 1 CPU gates.

Fetches the six official imagery EDF runs per participant from public AWS S3.
The EDF files stay on the ephemeral CI runner; no EEG is uploaded to Git.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.prepare_physionet_s3 import (
    BUCKET,
    PREFIX,
    RUNS,
    edf_path,
    fetch_run,
    unsigned_s3,
)


def stage(mne_root: Path, output: Path) -> dict:
    client = unsigned_s3()
    records = []
    for subject in (1, 2):
        for run in RUNS:
            size, seconds = fetch_run(client, subject, run, mne_root)
            path = edf_path(mne_root, subject, run)
            if not path.is_file() or path.stat().st_size != size:
                raise RuntimeError(f"Invalid staged EDF: participant {subject}, run {run}")
            records.append({
                "subject": subject,
                "run": run,
                "bytes": size,
                "download_seconds": seconds,
            })
    if len(records) != 12:
        raise RuntimeError("Expected exactly 12 staged imagery EDF files.")
    report = {
        "status": "complete",
        "source": "public PhysioNet S3",
        "bucket": BUCKET,
        "prefix": PREFIX,
        "subjects": [1, 2],
        "imagery_runs": list(RUNS),
        "file_count": len(records),
        "files": records,
        "note": "EDF files remain on ephemeral CI runner; this is not an EEG benchmark.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mne-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/phase1_s3_stage.json"))
    args = parser.parse_args()
    stage(args.mne_root, args.output)


if __name__ == "__main__":
    main()
