# Phase 2 — MC Dropout Uncertainty Quantification

## Objective

This phase adds uncertainty quantification to the Phase 1 head-only LaBraM baseline using Monte Carlo Dropout.

The goal is not to improve classification accuracy directly, but to estimate predictive uncertainty for each test prediction using repeated stochastic forward passes.

## Method

Uncertainty was estimated with MC Dropout using 50 stochastic forward passes per test batch.

Dropout was injected into two verified locations in the LaBraM model:

1. `model.pos_drop`

   * Replaced with `Dropout(p=0.3)`
   * Applied to the token sequence before the transformer blocks

2. `model.final_layer`

   * Wrapped as `Sequential(Dropout(p=0.3), Linear(...))`
   * Applied before the final classification head

The baseline checkpoint used for Phase 2 was the Phase 1 head-only checkpoint:

```text
/kaggle/working/labram_head_finetuned.pt
```

## Dataset and Split

Dataset: PhysioNet Motor Imagery via MOABB

The same subject-wise leakage-free split from Phase 1 was used:

```text
75 train / 14 validation / 15 test subjects
```

Total usable subjects:

```text
104
```

Excluded subjects:

```text
88, 89, 92, 100, 104
```

Reason for exclusion: sampling rate mismatch, 128 Hz instead of 160 Hz.

## Phase 1 Baseline Re-run

Because the Kaggle session was fresh, the Phase 1 head-only baseline checkpoint was re-trained before running uncertainty quantification.

Final Phase 1 test result from this run:

```text
TEST loss      : 1.6233
TEST accuracy  : 0.357
TEST macro-F1  : 0.193
```

Per-class Phase 1 test performance:

| Class      | Precision | Recall | F1-score | Support |
| ---------- | --------: | -----: | -------: | ------: |
| feet       |      0.14 |   0.28 |     0.18 |     342 |
| hands      |      0.12 |   0.07 |     0.09 |     333 |
| left_hand  |      0.16 |   0.11 |     0.13 |     339 |
| rest       |      0.52 |   0.62 |     0.57 |    1260 |
| right_hand |      0.00 |   0.00 |     0.00 |     336 |

## MC Dropout Configuration

```text
MC dropout probability : 0.3
MC stochastic passes   : 50
Inference batch size   : 64
```

## Phase 2 Results

```text
Accuracy         : 0.3065
Macro-F1         : 0.1860
ECE              : 0.0636
Mean entropy     : 1.5961
Mean expected entropy : 1.5362
Mean BALD score  : 0.0600
Mean std_probs   : 0.0692
```

## Per-class Predictive Entropy

| Class      | Mean Predictive Entropy |
| ---------- | ----------------------: |
| feet       |                  1.5966 |
| hands      |                  1.5963 |
| left_hand  |                  1.5961 |
| rest       |                  1.5960 |
| right_hand |                  1.5961 |

## Interpretation

The MC Dropout results show high predictive entropy across all classes. This is expected because the Phase 1 baseline itself is weak, with test accuracy around 0.31–0.36 and macro-F1 around 0.19–0.22.

The model remains especially weak on the fine-grained left-hand and right-hand motor imagery classes. The `right_hand` class received no correct predicted samples in the Phase 1 re-run, which confirms that the baseline is not yet clinically or practically reliable.

The mean BALD score is low but non-zero:

```text
Mean BALD score: 0.0600
```

This suggests that the model has measurable epistemic uncertainty, but most of the predictive uncertainty appears to come from the generally ambiguous or poorly separated class predictions rather than strong disagreement across stochastic model samples.

The ECE value from this run was:

```text
ECE: 0.0636
```

This should be interpreted carefully. A low ECE does not mean the model is accurate or reliable. In this case, the model has low accuracy and high predictive entropy, so its confidence is generally low. Therefore, calibration appears numerically reasonable, but the classifier remains weak.

## Output Artifact

The raw MC Dropout output was saved during the Kaggle run as:

```text
/kaggle/working/mc_dropout_results.npz
```

The saved file contains:

```text
mean_probs
std_probs
pred_entropy
expected_ent
bald_score
predictions
labels
```

This binary artifact was not committed to GitHub. The repository keeps the reproducible code and the summarized result report instead.
