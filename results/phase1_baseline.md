# Phase 1 — Baseline Results

LaBraM (InterpolatedLaBraM), fine-tuned on PhysioNet Motor Imagery,
subject-wise leakage-free split (75 train / 14 val / 15 test subjects, of 104).

| Config                          | Test Acc | Test Macro-F1 |
|----------------------------------|----------|----------------|
| Head-only (linear probe)         | 0.32–0.36| 0.21–0.22      |
| Head + last 2 transformer blocks | 0.29     | 0.223          |

## Limitation
Unfreezing additional transformer blocks did not improve generalization —
train macro-F1 reached 0.62 while val macro-F1 stayed ≤0.22, indicating
overfitting rather than better representation learning. This suggests the
~13k training epochs available are insufficient for deeper fine-tuning of
this model on this task; the head-only linear probe is used as the
reported baseline going forward. `right_hand`/`left_hand` discrimination
is consistently the weakest, while `rest` is best-classified.
