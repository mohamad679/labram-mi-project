"""
Phase 1 — Baseline fine-tuning loop.

NOT implemented yet — this is a stub for Phase 1.
Do not start filling this in until Phase 0's exit criteria
(forward pass test) has passed.
"""

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/baseline.yaml")
    args = parser.parse_args()

    # TODO (Phase 1):
    # 1. Load data via src/data.py (subject-aware train/test split)
    # 2. Load model via src/model.py
    # 3. Freeze backbone, fine-tune classification head first
    # 4. Save checkpoints regularly (Kaggle GPU quota risk)
    # 5. Report accuracy/F1 on held-out subject split
    raise NotImplementedError("Phase 1 not started yet — finish Phase 0 first.")


if __name__ == "__main__":
    main()
