# Research integrity and migration audit

## Why this document exists

This audit separates archived observations from claims proven by the versioned code. Historical metrics were reported before the October 2026 reproducibility refactor. **No full GPU training or independent rerun has been performed by this migration.**

## Critical finding: initialization of pretrained model

The original `src/model.py` constructed `InterpolatedLaBraM(...)` without calling `from_pretrained(...)` or explicitly loading pretrained backbone parameters before the head-only training phase. The historical trained checkpoint is not part of Git. This does **not** establish that the original run used pretrained weights. According to Braindecode documentation, creating a model and loading pretrained weights are distinct operations.

- Current refactor: `model_init: scratch` makes the default initialization explicit.
- Pretrained experiment: set `model_init: pretrained`, supply a verified `pretrained_repo_id`, and record the exact artifact revision/hash. This code path still requires end-to-end compatibility testing.
- Existing archived numbers: retain as historical figures of **unverified checkpoint provenance**. Do not label them pretrained LaBraM results until provenance is demonstrated.

Reference: https://braindecode.org/stable/auto_examples/model_building/plot_channel_interpolation.html

## Other observed gaps

1. Baseline YAML formerly declared 200 epochs, batch size 64 and learning rate 1e-4 while `src/train.py` used 30 epochs, 32 and 1e-3. The script did not read YAML.
2. The former `test_no_subject_leakage` consisted of `pass`. Split disjointness already existed as an assertion in runtime code.
3. The former Kaggle notebook contained username placeholders and a `--config` argument the training script did not parse.
4. Packages were unbounded and there were no GitHub Actions workflows, code license file or tagged releases.
5. The root `gradio_dashboard_demo.mp4` file is a 25-byte text placeholder. The actual video is in `results/figures/phase4_gradio_demo/`.
6. README claimed generated PNGs were uncommitted although rollout PNGs are present.
7. Model inference uses a local checkpoint not provided to new users. Training and full demo reproducibility remain unverified.

## Methodological caution: uncertainty

The research code injects new dropout layers after loading a trained checkpoint. That does not itself establish the Bayesian interpretation usually associated with MC Dropout trained under a matching stochastic model. Predictive entropy, expected entropy and BALD are therefore presented as **exploratory model-output statistics**, not a calibrated clinical or validated epistemic risk score.

## Methodological caution: attention

Attention rollout is an inspection of transformer attention propagation, not a causal attribution or neuroscience ground truth. Canonical LaBraM channel tokens should not be equated with directly measured physical sensor importance without establishing a validated spatial mapping.

## Next scientific acceptance gates

- Confirm and freeze the source, version and hash of any pretrained backbone.
- Run one reproducible experiment with full log, split manifest and dependency lock.
- Evaluate macro-F1, accuracy, class-wise scores and confusion matrix on untouched held-out subjects.
- Compare with meaningful non-foundation baselines; avoid model selection on test data.
- Document multiple random seeds, model artifacts, statistical uncertainty and limitations.
- Validate MC Dropout protocol or explicitly keep it as a heuristic diagnostic.
