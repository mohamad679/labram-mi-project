> **Frozen-project context (2026-10-13):** This document preserves older reproducibility instructions for research provenance. Experiments are **closed**; no further training, seed search or Test evaluation is planned. The completed 104-subject data hash, model comparisons and outstanding reproduction caveats are in the [final scientific summary](final-research-summary.md).

# Reproducibility guide

## Environment

This repository currently uses bounded dependency ranges, **not** an exact tested lockfile. Create a fresh Python 3.10+ environment, install `requirements.txt`, and archive `pip freeze` and hardware/device details for every published run. Avoid interpreting installation success as verified reproducibility.

## Dataset

Data source: PhysioNet EEG Motor Movement/Imagery dataset via MOABB. The loader excludes subject IDs 88, 89, 92, 100 and 104, historically reported as having incompatible sampling rates. The project expects arrays of shape `(epochs,64,400)` and labels in five motor imagery classes.

Data are cached locally in `physionet_mi_cache.npz` or at `PHYSIONET_CACHE_PATH`. New schema-2 caches include ordered channel names, source sampling rate and **microvolt** EEG units (1e6 scaling from MNE volts); **unverified legacy caches fail closed** unless `ALLOW_UNVERIFIED_EEG_CACHE=1` is set for exploratory/non-publishable use. Data loading now uses MOABB `return_epochs=True`; it may differ numerically from legacy NumPy output. [See migration audit](eeg-data-provenance.md). The cache **must not be committed to GitHub**. Preserve the original subject-level dataset identifiers and original file hashes for rigorous reproduction. Check source data licenses and permissions separately.

## Run

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python -m src.train --config configs/baseline.yaml
```

With the default configuration the model is initialized **from scratch**. Do not call this a pretrained experiment. The CPU checkpoint loader and the explicit 400-sample temporal-prefix adaptation were verified in [GitHub Actions](https://github.com/mohamad679/labram-mi-project/actions/runs/38234001055), but no real EEG fine-tuning was performed. A separate explicit pretrained mode requires `model_init: pretrained`, `pretrained_repo_id`, **and `pretrained_revision` with an immutable 40-character Hub commit SHA**. Obtain the revision from [the official pretrained LaBraM Hub repository](https://huggingface.co/braindecode/labram-pretrained). The pinned checkpoint and forward compatibility were validated on CPU at 400 samples; training performance and real-data preprocessing remain unverified.

The training script records its configuration and subject IDs for the train, validation and test sets in the JSON file beside the checkpoint. It also records test accuracy, test macro-F1 and best validation macro-F1. This does not yet capture package hashes or all dataset provenance.

After training:

```bash
python -m src.uncertainty
python -m src.explain
python app.py
```

Phase 2/3 output directories remain Kaggle-oriented in the legacy code. Their use outside Kaggle requires additional portability work.

## Reproducibility contract (not yet fully satisfied)

1. Freeze the exact input data version and hashes.
2. Freeze all dependency versions, including the matching `braindecode` release.
3. Record pretrained checkpoint URI, revision and checksum if using pretrained initialization.
4. Archive random state and subject IDs without exposing raw subject EEG.
5. Repeat with multiple seeds; report means/intervals rather than selected best runs.
6. Provide a versioned release artifact or verified model download instructions.
7. Execute a complete smoke test of uncertainty, explainability and Gradio on a fresh environment.

## Safety and scope

This is research software; it is not a medical device, an operational clinical classifier or an accessibility-grade BCI.

## Auditing pretrained weights (network required)

The [Braindecode interpolation tutorial](https://braindecode.org/stable/auto_examples/model_building/plot_channel_interpolation.html) demonstrates the model using the 128-channel canonical backbone, a 64-channel adapter in this project, and a pretrained Hub source. The tutorial uses `n_times=3000`; our project's input is **400 samples**. This compatibility difference has not been resolved by a successful complete run.

Resolve the actual commit ID (never substitute a guessed value):

```bash
python -c "from huggingface_hub import HfApi; print(HfApi().model_info('braindecode/labram-pretrained').sha)"
```

Audit that exact commit, with no patient data:

```bash
python -m scripts.audit_pretrained --revision <FULL_40_CHARACTER_COMMIT_SHA> --n-times 400
```

This audit downloads a pinned Safetensors file, computes its SHA256, loads the matching pretrained wrapper, and compares compatible backbone tensors exactly. If the experiment fails, **do not describe the model as successfully pretrained or publish its metrics**. If it succeeds, preserve the generated `artifacts/pretrained-audit.json` alongside exact package versions; then run a separate forward pass and complete held-out experiment. Do not claim that a weight-loading audit validates downstream model accuracy.

This audit has **not been run** as part of this change.
