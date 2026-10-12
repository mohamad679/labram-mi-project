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
| Phase 3 | Complete | Attention rollout explainability                               |
| Phase 4 | Complete | Lightweight Gradio demo, backend inference, and Kaggle UI test |

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

### Phase 3 — Attention rollout explainability

Main report:

```text
results/phase3_attention_rollout.md
```

Attention rollout was computed across all 12 LaBraM transformer blocks.

For each transformer block:

```text
attention heads were averaged
identity residual connection was added
attention rows were normalized
rollout matrices were multiplied across layers
```

The CLS-token rollout was summarized into:

| Output | Shape |
| ------ | ----: |
| Full rollout matrix | 4 × 257 × 257 |
| Patch-token importance | 4 × 256 |
| Canonical channel importance | 4 × 128 |
| Temporal patch importance | 4 × 2 |

Four representative test samples were analyzed.

The selected samples and model predictions were:

| Sample | Dataset index | True label | Predicted label |
| -----: | ------------: | ---------- | --------------- |
| 0 | 0  | right_hand | feet |
| 1 | 1  | rest       | feet |
| 2 | 2  | left_hand  | rest |
| 3 | 87 | feet       | rest |

The top attention-rollout channels varied across samples. Examples include:

```text
Sample 0: A1, FP2, M1, TP9, AF8
Sample 1: F9, PO2, POZ, FP1-F7, P2
Sample 2: A2, M2, T8, O2, CFC8
Sample 3: FP1-F7, M2, FP2-F8, FP2, A1
```

Temporal patch importance was also extracted for each sample:

| Sample | Patch 0 | Patch 1 |
| -----: | ------: | ------: |
| 0 | 0.003973 | 0.003792 |
| 1 | 0.004147 | 0.003623 |
| 2 | 0.003945 | 0.003817 |
| 3 | 0.003741 | 0.004024 |

The model predictions remained weak, consistent with the Phase 1 baseline. However, the Phase 3 pipeline successfully produced token-level, channel-level, and temporal-patch-level explanations from the LaBraM transformer attention structure.

The raw Kaggle output was saved as:

```text
/kaggle/working/attention_rollout_results.npz
```

The generated plots were saved in:

```text
/kaggle/working/attention_rollout_plots/
```

The generated plot files were:

```text
sample_0_top_channels.png
sample_0_time_patches.png
sample_1_top_channels.png
sample_1_time_patches.png
sample_2_top_channels.png
sample_2_time_patches.png
sample_3_top_channels.png
sample_3_time_patches.png
```

The binary `.npz` file and generated PNG plots are not committed to GitHub. The repository stores reproducible code and summarized Markdown reports instead.

### Phase 4 — Recordable Gradio Dashboard
### Demo Video

A short screen recording of the upgraded three-tab Gradio dashboard is available here:

[Watch the Gradio dashboard demo](results/figures/phase4_gradio_demo/gradio_dashboard_demo.mp4)

The dashboard demonstrates:

- deterministic motor-imagery classification
- MC Dropout uncertainty estimation with entropy and BALD
- attention-rollout explainability with channel and temporal-patch visualizations

Main report:

```text
results/phase4_gradio_demo.md
```

Main application file:

```text
app.py
```

The demo accepts one EEG epoch as a `.npy` file and returns:

- predicted motor imagery class
- class probabilities
- input-shape validation details
- checkpoint/device information

Supported input shapes:

```text
(64, 400)
(1, 64, 400)
(400, 64)  # automatically transposed
```

Checkpoint search order:

```text
LABRAM_CHECKPOINT environment variable
/kaggle/working/labram_head_finetuned.pt
checkpoints/labram_head_finetuned.pt
labram_head_finetuned.pt
```

The demo was tested from a fresh GitHub clone on Kaggle.

Backend prediction test:

```text
sample index : 0
sample shape : (64, 400)
true label   : right_hand
predicted    : feet
device       : cuda
checkpoint   : /kaggle/working/labram_head_finetuned.pt
```

Class probabilities from the backend/UI test:

| Class | Probability |
|---|---:|
| feet | 0.2333 |
| hands | 0.1807 |
| left_hand | 0.2032 |
| rest | 0.2274 |
| right_hand | 0.1554 |

The Gradio UI launched successfully on Kaggle and produced a temporary public Gradio URL. A `.npy` sample was uploaded through the UI, and the app returned the same probability distribution and predicted class as the backend test.

Observed UI output:

```text
Predicted class: feet
Original shape: (64, 400)
Model input shape: (64, 400)
Device: cuda
```

The demo is an interactive proof-of-functionality for model loading, input validation, inference, and probability display. It is not a production BCI interface and does not provide clinical guidance.



## Interpretation

The Phase 1 baseline is intentionally modest. This is not presented as a production-ready BCI classifier.

The main technical contribution is the end-to-end, leakage-free, reproducible pipeline:

```text
PhysioNet MI
→ LaBraM adaptation
→ subject-wise split
→ baseline fine-tuning
→ MC Dropout uncertainty
→ attention rollout explainability
→ lightweight Gradio demo
```

The model shows high predictive entropy across classes, which is consistent with the weak baseline accuracy and macro-F1.

The low ECE value in Phase 2 should be interpreted carefully. It does not mean the classifier is accurate. It means the model's confidence is numerically aligned with its low-confidence predictions in this particular run.

The Phase 3 attention-rollout results should also be interpreted carefully. Attention rollout provides a structured way to inspect how information flows through transformer attention layers, but it should not be treated as a clinically validated explanation of EEG physiology.

The Phase 4 Gradio demo demonstrates software integration and interactive inference. It does not change the scientific limitations of the baseline model.

## Project Structure

```text
app.py             # Phase 4 Gradio demo

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
  phase3_attention_rollout.md
  phase4_gradio_demo.md

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

Run Phase 3:

```python
!python -m src.explain
```

Run Phase 4 Gradio demo:

```python
!python app.py
```

Then open the Gradio URL, upload a `.npy` EEG epoch, and click `Predict`.

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

The remaining optional packaging step is persistent HuggingFace Spaces deployment.

## License

The base LaBraM implementation is accessed through `braindecode`.

Base model license:

```text
BSD-3-Clause
```
