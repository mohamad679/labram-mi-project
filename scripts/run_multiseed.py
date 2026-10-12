"""Prepare or execute split-locked, independently seeded full EEG experiments."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml


def make_run_configs(source: Path, run_root: Path, seeds: list[int]) -> list[Path]:
    if not seeds or len(seeds) != len(set(seeds)) or any(s < 0 for s in seeds):
        raise ValueError("Seeds must be a nonempty, unique list of nonnegative integers.")
    with source.open(encoding="utf-8") as stream:
        baseline = yaml.safe_load(stream)
    if not isinstance(baseline, dict):
        raise TypeError("Expected YAML mapping.")
    if "split_seed" not in baseline:
        raise ValueError("Baseline configuration must specify split_seed.")
    run_paths = []
    for seed in seeds:
        run_dir = run_root / f"seed_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        config = baseline.copy()
        config["seed"] = seed
        config["split_seed"] = baseline["split_seed"]
        config["checkpoint_path"] = str(run_dir / "model.pt")
        path = run_dir / "training.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
        run_paths.append(path)
    return run_paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/experiments"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2025])
    parser.add_argument(
        "--execute", action="store_true", help="Actually run full training (GPU recommended)"
    )
    args = parser.parse_args()
    paths = make_run_configs(args.config, args.output_dir, args.seeds)
    manifest = {
        "config_source": str(args.config),
        "split_seed": yaml.safe_load(args.config.read_text(encoding="utf-8"))["split_seed"],
        "model_seeds": args.seeds,
        "prepared_configs": [str(path) for path in paths],
        "status": "prepared",
        "note": "All model seeds share one train/validation/test subject partition.",
    }
    target = args.output_dir / "experiment-plan.json"
    target.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if args.execute:
        for path in paths:
            subprocess.run(
                [sys.executable, "-m", "src.train", "--config", str(path)], check=True
            )
        manifest["status"] = "finished"
        target.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
