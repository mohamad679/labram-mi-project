<div align="center">

# NeuroDecode

**Motor-imagery EEG · reproducible experiments · research integrity**

*A completed, evidence-linked development study — not a production BCI or a Test-validated model.*

[![Repository checks](https://github.com/mohamad679/labram-mi-project/actions/workflows/ci.yml/badge.svg)](https://github.com/mohamad679/labram-mi-project/actions/workflows/ci.yml)
[![Documentation integrity](https://github.com/mohamad679/labram-mi-project/actions/workflows/repository-quality.yml/badge.svg)](https://github.com/mohamad679/labram-mi-project/actions/workflows/repository-quality.yml)
![Study state](https://img.shields.io/badge/study-frozen%20development%20snapshot-147d82)
![Evaluation](https://img.shields.io/badge/Test-never%20evaluated-d7a14f)

[**Final findings**](docs/final-research-summary.md) · [**Scientific figures**](docs/visualization-gallery.md) · [**Experimental evidence**](results/validation/README.md) · [**Research protocol**](docs/README.md) · [**Archive status**](docs/release-readiness.md)

</div>

![Evidence-backed comparison of actual development Validation macro-F1 results](results/figures/final_snapshot/benchmark-overview.svg)

> [!IMPORTANT]
> **Research stopped after Phase 5.5b (13 October 2026).** The final model-selection exercise is complete; there will be **no further seeds, tuning sweeps or new benchmark runs** in this frozen snapshot. The original 15-participant Test partition was **never evaluated**. No final held-out Test performance, production readiness, clinical validity or multi-seed robustness is claimed.

## What was actually studied?

The project uses **PhysioNet EEG Motor Movement/Imagery (EEGMMIDB)**, through a pinned MOABB/MNE preprocessing protocol, for *five-class* classification of `feet`, `hands`, `left_hand`, `rest` and `right_hand`. There are **104 retained participants** and **18,096 EEG epochs** (64 channels × 400 samples, 160 Hz). The subject-disjoint split was frozen before the recorded benchmark runs:

![Visual evidence of the 75/14/15 subject partition with the final Test untouched](results/figures/final_snapshot/frozen-study-design.svg)

| Evaluation set | Participants | Epochs | Actual use |
| :-- | --: | --: | :-- |
| Train | 75 | 13,050 | Model fit; Train-only 3-fold optimization |
| Development Validation | 14 | 2,436 | Recorded exploratory comparisons; one nominated Phase 5.5b assessment |
| Final Test | 15 | 2,610 | **Locked; no waveform inference or reported scores** |

The independently verified **HDF5 SHA-256** is `f6d2c7824f6aaa522286ea8ee5253b0220602117fc69d7b49c3af6e6a1324f7a`. See the [frozen split manifest](configs/phase3_locked_split.json), [preprocessing audit](docs/phase2-data-audit.md) and [source-data provenance](docs/eeg-data-provenance.md). EEG waveforms and training checkpoints are intentionally **not committed to Git**.

## Recorded development findings

**Primary metric: fixed-five-class Macro-F1** (higher is better). All numbers below come from completed, audited **development** runs. They are **descriptive across different training and input-adaptation protocols**, not paired architecture-only experiments or Test results.

| Method | Accuracy | Macro-F1 | Evidence |
| :-- | --: | --: | :-- |
| **EEGNet · original** | 0.4799 | **0.4191** | [Phase 3](results/validation/phase3_validation_rerun_seed42_20261013.json) |
| EEGNet · normalized, selected in Train-only screening | **0.4885** | 0.4125 | [Phase 5.5b](results/validation/phase55b_one_time_validation_seed42_20261013.json) |
| Riemannian tangent-space baseline | 0.2767 | 0.2568 | [Phase 3](results/validation/phase3_validation_rerun_seed42_20261013.json) |
| LaBraM · partial pretrained fine-tuning | 0.3859 | 0.2546 | [Phase 4](results/validation/phase4_validation_seed42_20261013.json) |
| CBraMod · frozen pretrained probe | 0.2779 | 0.2027 | [Phase 5](results/validation/phase5_cbramod_validation_seed42_20261013.json) |
| CBraMod · full pretrained fine-tuning | 0.4828 | 0.1302 | [Phase 5](results/validation/phase5_cbramod_validation_seed42_20261013.json) |
| Majority-class dummy | 0.4828 | 0.1302 | [Phase 3](results/validation/phase3_validation_rerun_seed42_20261013.json) |

**Interpretation matters:** Full-tuned CBraMod predicts **only `rest` for all 2,436 Validation epochs**, achieving misleadingly high accuracy because `rest` is nearly half the evaluation set. It is not a successful imagery decoder. The genuine pretrained weights were authenticated, but the task transfer failed under the recorded setup.

### Did input standardization help?

Yes in **Train-only subject-grouped OOF** screening: EEGNet Macro-F1 rose from **0.3212** to **0.3912** under otherwise matched scratch-EEGNet screening conditions (+0.0700). However, the nominated normalized model reached **0.4125** in the subsequent original Validation assessment, versus **0.4191** for the *historical* EEGNet run. **An improvement on the original Validation cohort was not demonstrated**; the historical runs used different fit/selection protocols.

![Four Train-only CV candidates followed by the single recorded development Validation comparison](results/figures/final_snapshot/train-only-vs-validation.svg)

## Explore the actual evidence

| Five-class confusion matrix | Subject-to-subject spread |
| :-- | :-- |
| ![One-time normalized EEGNet Validation confusion matrix](results/figures/final_snapshot/validation-confusion-matrix.svg) | ![Normalized EEGNet Macro-F1 on each of 14 Validation participants](results/figures/final_snapshot/validation-subject-spread.svg) |

[**Full figure gallery and descriptions →**](docs/visualization-gallery.md) · [**Final interpretation, risks and limitations →**](docs/final-research-summary.md)

## Repository guide

| Location | Purpose |
| :-- | :-- |
| [`docs/final-research-summary.md`](docs/final-research-summary.md) | **Start here:** final findings, scientific limitations, stop decision |
| [`results/validation/`](results/validation/README.md) | Independently reviewed real-run JSON evidence and immutable hashes |
| [`results/figures/final_snapshot/`](docs/visualization-gallery.md) | Six chart SVGs **generated deterministically from evidence JSON** |
| [`scripts/render_final_figures.py`](scripts/render_final_figures.py) | Regenerate or check the published SVGs; **no EEG or training** |
| [`notebooks/`](notebooks/README.md) | Historically executed Kaggle notebooks, retained for provenance — not an invitation to rerun |
| [`src/`, `scripts/`, `tests/`](src/data.py) | Research implementation, audit tools and regression tests |
| [`results/figures/phase3_attention_rollout/`](results/figures/phase3_attention_rollout/sample_0_top_channels.png) | Historical attention inspection; **not causal neural attribution** |
| [`results/figures/phase4_gradio_demo/`](results/figures/phase4_gradio_demo/gradio_ui_prediction.png) | Historical Gradio screenshot/video; needs a compatible private checkpoint |

### Validate the archive without training

```bash
python -m pytest -q
python -m scripts.check_repository_quality
python -m scripts.render_final_figures --check
```

Regenerating verified charts from already committed evidence (without EEG, network or GPU):

```bash
python -m scripts.render_final_figures
```

The original data-processing, training and demonstration tools remain available **as research provenance**, but this study is closed. **Do not run the historical training scripts and interpret a fresh seed/Test score as part of this frozen benchmark.** For reproducibility requirements see [research documentation](docs/reproducibility.md) and [source pinning](docs/phase55b-one-time-validation-protocol.md).

## Scientific and distribution boundaries

- **Subject-independent generalization is still variable.** The final nominated model has Validation per-subject Macro-F1 from **0.218** to **0.704**; its overall Macro-F1 is **0.4125** with exploratory subject-cluster 95% interval **[0.3486, 0.4847]**. Model/epoch selection and reused development data limit inferential strength.
- LaBraM and CBraMod pretrained runs use different architecture-specific preprocessing; their results are **not** evidence of a universally inferior foundation model.
- Neither the attention maps nor retrospective Monte-Carlo dropout / BALD estimates are validated clinical explanations or calibrated uncertainty.
- This project does **not** include a verified held-out Test score, a multi-seed benchmark, released model weights, a clinical validation or an independently reproduced end-to-end installation.
- **No license for the repository's original source code was selected by its owner.** Upstream dataset, weights and library licenses must be evaluated separately; no reuse permission should be inferred from public visibility.

See the [archive acceptance record](docs/release-readiness.md), [model card](docs/model-card.md), and [research integrity audit](docs/research-audit.md).

---

<sub>Independent research implementation using [PhysioNet EEGMMIDB](https://physionet.org/content/eegmmidb/1.0.0/), [MOABB](https://moabb.neurotechx.com/), [MNE](https://mne.tools/), and [Braindecode](https://braindecode.org/). No affiliation with these maintainers. **Frozen development snapshot · 2026-10-13.**</sub>
