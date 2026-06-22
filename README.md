# LaBraM Fine-Tuning — Motor Imagery BCI

Fine-tuning and evaluation of LaBraM, an EEG foundation model, on the PhysioNet Motor Imagery dataset.

The project focuses on a realistic, phase-gated BCI pipeline:

1. LaBraM model adaptation for PhysioNet Motor Imagery
2. Subject-wise leakage-free baseline fine-tuning
3. Uncertainty quantification with MC Dropout
4. Explainability with attention rollout
5. Lightweight Gradio deployment

This is a portfolio research-engineering project for PhD outreach and AI engineer applications.

## Current Status

| Phase   | Status   | Description                                                    |
| ------- | -------- | -------------------------------------------------------------- |
| Phase 0 | Complete | Data loading, LaBraM model adaptation, forward-pass validation |
| Phase 1 | Complete | Head-only baseline fine-tuning with subject-wise split         |
| Phase 2 | Complete | MC Dropout uncertainty quantification                          |
| Phase 3 | Next     | Attention rollout explainability                               |
| Phase 4 | Planned  | Gradio demo and HuggingFace Spaces deployment                  |

## Dataset

Dataset: PhysioNet Motor Imagery via MOABB

Input format:

```text
64 EEG channels
160 Hz sampling rate
400 time points after cropping
5 output classes
```

Classes:

```text
feet
hands
left_hand
rest
right_hand
```

The split is subject-wise and leakage-free:

```text
75 train subjects
14 validation subjects
15 test subjects
104 usable subjects total
```

Five subjects were excluded because their sampling rate was 128 Hz instead of 160 Hz:

```text
88, 89, 92, 100, 104
```

## Model

Base model: LaBraM via `braindecode`

Key implementation details:

```text
n_times  : 400
n_outputs: 5
strategy : head-only fine-tuning baseline
```

The adapted model contains approximately:

```text
5.8M parameters
```

For Phase 1, only the final classification head was fine-tuned:

```text
final_layer: 1005 trainable parameters
```

## Results

### Phase 1 — Head-only baseline

Main report:

```text
results/phase1_baseline.md
```

Best observed test performance across Phase 1 runs:

| Configuration                    | Test Accuracy | Test Macro-F1 |
| -------------------------------- | ------------: | ------------: |
| Head-only linear probe           |     0.32–0.36 |     0.19–0.22 |
| Head + last 2 transformer blocks |          0.29 |         0.223 |

The head-only baseline was selected because deeper unfreezing overfit strongly under the available compute budget.

A representative Phase 1 re-run before Phase 2 produced:

```text
TEST loss      : 1.6233
TEST accuracy  : 0.357
TEST macro-F1  : 0.193
```

Per-class performance showed that `rest` was the strongest class, while `left_hand` and `right_hand` remained weak.

### Phase 2 — MC Dropout uncertainty quantification

Main report:

```text
results/phase2_uncertainty.md
```

MC Dropout configuration:

```text
dropout probability : 0.3
stochastic passes   : 50
batch size          : 64
```

Dropout was injected at two verified model locations:

```text
model.pos_drop
model.final_layer
```

Phase 2 test results:

| Metric                  |  Value |
| ----------------------- | -----: |
| Accuracy                | 0.3065 |
| Macro-F1                | 0.1860 |
| ECE                     | 0.0636 |
| Mean predictive entropy | 1.5961 |
| Mean expected entropy   | 1.5362 |
| Mean BALD score         | 0.0600 |
| Mean probability std    | 0.0692 |

The raw Kaggle output was saved as:

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

The binary `.npz` result file is not committed to GitHub. The repository stores reproducible code and summarized result reports instead.

## Interpretation

The Phase 1 baseline is intentionally modest. This is not presented as a production-ready BCI classifier.

The main technical contribution is the end-to-end, leakage-free, reproducible pipeline:

```text
PhysioNet MI → LaBraM adaptation → subject-wise split → baseline fine-tuning → MC Dropout uncertainty
```

The model shows high predictive entropy across classes, which is consistent with the weak baseline accuracy and macro-F1.

The low ECE value in Phase 2 should be interpreted carefully. It does not mean the classifier is accurate. It means the model's confidence is numerically aligned with its low-confidence predictions in this particular run.

## Project Structure

```text
src/
  data.py          # MOABB / PhysioNet MI data loading
  model.py         # LaBraM model adaptation
  train.py         # Phase 1 baseline fine-tuning
  uncertainty.py   # Phase 2 MC Dropout uncertainty
  explain.py       # Phase 3 attention rollout
  evaluate.py      # accuracy, F1, calibration metrics

results/
  phase1_baseline.md
  phase2_uncertainty.md

configs/
  baseline.yaml

tests/
  test_data.py

run_kaggle.ipynb
requirements.txt
README.md
```

## Reproducibility on Kaggle

This project is designed to run on Kaggle with free GPU acceleration.

Recommended accelerator:

```text
GPU T4 x2
```

Standard session setup:

```python
!rm -rf labram-mi-project
!git clone https://github.com/mohamad679/labram-mi-project.git
%cd labram-mi-project
!pip install -q -r requirements.txt
```

Dataset symlink setup:

```python
!mkdir -p /root/mne_data/MNE-eegbci-data/files/eegmmidb
!rm -rf /root/mne_data/MNE-eegbci-data/files/eegmmidb/1.0.0
!ln -s /kaggle/input/datasets/gamalasran/physionet-eeg-motor-movement-imagery/files \
        /root/mne_data/MNE-eegbci-data/files/eegmmidb/1.0.0
```

Run Phase 1:

```python
!python -m src.train
```

Run Phase 2:

```python
!python -m src.uncertainty
```

## Scope Boundaries

This project intentionally does not include:

```text
TUEG pretraining
seizure detection
multi-dataset benchmarking
production API/auth
React dashboard
clinical deployment claims
```

The next phase is explainability using attention rollout only.

## License

The base LaBraM implementation is accessed through `braindecode`.

Base model license:

```text
BSD-3-Clause
```
