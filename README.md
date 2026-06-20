# LaBraM Fine-Tuning — Motor Imagery BCI

EEG foundation model (LaBraM, via braindecode) fine-tuned on PhysioNet
Motor Imagery, with uncertainty quantification and explainability.

License of base model: BSD-3-Clause (braindecode/labram-pretrained).

## Project structure

```
src/
  data.py          # Phase 0 — MOABB data loading
  model.py         # Phase 0 — LaBraM model loading
  train.py         # Phase 1 — baseline fine-tuning
  uncertainty.py   # Phase 2 — MC Dropout / Conformal Prediction
  explain.py       # Phase 3 — attention rollout
  evaluate.py       # metrics + calibration
tests/
  test_data.py     # sanity + leakage checks
configs/
  baseline.yaml    # training config
run_kaggle.ipynb   # thin launcher notebook for Kaggle's free GPU
```

## Current phase: Phase 0 — Setup & Data

Exit criteria: a successful forward pass on a real batch, with no
channel/shape errors. See `src/data.py` and `src/model.py` smoke tests.

## How to run on Kaggle (free GPU)

1. Push this repo to your own GitHub (see below).
2. Open `run_kaggle.ipynb` on Kaggle, enable GPU (Settings → Accelerator → T4 x2).
3. The notebook clones this repo and runs the actual scripts — no logic
   lives inside the notebook itself.

## How to push this to your own GitHub

```bash
cd labram-mi-project
git init
git add .
git commit -m "Phase 0: project skeleton"
git branch -M main
git remote add origin https://github.com/<your-username>/<repo-name>.git
git push -u origin main
```

(Create the empty repo first on github.com — New Repository — then use
its URL in the `git remote add` command above.)
