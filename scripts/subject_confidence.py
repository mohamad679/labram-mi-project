"""Subject-cluster confidence intervals from actual held-out predictions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.subject_statistics import subject_cluster_bootstrap


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/evaluation/subject_ci.json"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-bootstraps", type=int, default=1000)
    args = parser.parse_args()
    with np.load(args.predictions, allow_pickle=False) as data:
        required = {"y_true", "y_prob", "subject_ids"}
        if not required.issubset(data.files):
            raise ValueError(
                f"Missing subject-cluster input fields: {sorted(required - set(data.files))}"
            )
        result = subject_cluster_bootstrap(
            data["y_true"], data["y_prob"], data["subject_ids"],
            n_bootstraps=args.n_bootstraps, seed=args.seed,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
