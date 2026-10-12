"""Create evidence-linked scientific plots from an actual held-out run."""
import argparse
import json
from pathlib import Path

import numpy as np

from src.reporting import render_evaluation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, help="Training-generated NPZ.")
    parser.add_argument("--output", default="artifacts/evaluation")
    parser.add_argument("--run-metadata", help="Path to training metadata JSON")
    args = parser.parse_args()

    with np.load(args.predictions, allow_pickle=False) as data:
        expected = {"y_true", "y_prob", "class_names"}
        if not expected.issubset(data.files):
            raise ValueError(f"Missing prediction arrays: {sorted(expected - set(data.files))}")
        truth, probabilities = data["y_true"], data["y_prob"]
        names = tuple(str(name) for name in data["class_names"].tolist())

    if args.run_metadata:
        metadata = json.loads(Path(args.run_metadata).read_text(encoding="utf-8"))
        if "test_subjects" not in metadata:
            raise ValueError("No held-out test subject manifest in run metadata.")
        if metadata.get("status", "finished") != "finished":
            raise ValueError("Cannot plot an incomplete run.")

    scores = render_evaluation(truth, probabilities, names, args.output)
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
