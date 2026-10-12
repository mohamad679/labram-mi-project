# Reproducibility guide

## Environment

This repository currently uses bounded dependency ranges, **not** an exact tested lockfile. Create a fresh Python 3.10+ environment, install `requirements.txt`, and archive `pip freeze` and hardware/device details for every published run. Avoid interpreting installation success as verified reproducibility.

## Dataset

Data source: PhysioNet EEG Motor Movement/Imagery dataset via MOABB. The loader excludes subject IDs 88, 89, 92, 100 and 104, historically reported as having incompatible sampling rates. The project expects arrays of shape `(epochs,64,400)` and labels in five motor imagery classes.

Data are cached locally in `physionet_mi_cache.npz` or at `PHYSIONET_CACHE_PATH`. The cache **must not be committed to GitHub**. Preserve the original subject-level dataset identifiers and original file hashes for rigorous reproduction. Check source data licenses and permissions separately.

## Run

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python -m src.train --config configs/baseline.yaml
```

With the default configuration the model is initialized **from scratch**. Do not call this a pretrained experiment. A separate explicit pretrained mode is available through `model_init: pretrained` and `pretrained_repo_id`, but its Braindecode/model revision compatibility has **not** been verified by this migration.

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
