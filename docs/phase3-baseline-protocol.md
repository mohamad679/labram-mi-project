# Phase 3 — Locked-split validation benchmark (execution OPEN)

**Scientific status:** STARTED. No real 104-subject baseline performance metrics have yet been accepted. **Do not claim Phase 3 PASS or proceed to Phase 4** before reviewing executed, provenance-complete Kaggle results.

## Experimental question

Does a frozen, pinned-pretrained LaBraM linear probe generalize to **14 unseen validation subjects** better than meaningful EEG-specific non-foundation baselines, after correcting invalid frozen-scratch training in Phase 1 and validating the input contract in Phase 2?

This is **development model comparison** using validation participants, not independent test/confirmatory evaluation. The already inspected historical test accuracy cannot be used for candidate selection.

## Immutable source dataset and split

- Data: PhysioNet EEGMMIDB 1.0.0, *imagery only* and the frozen 104-subject exclusion policy from [Phase 2](phase2-data-audit.md).
- Input: MOABB 1.7.2; 8–32 Hz; 64 physically verified channels; 160 Hz; first 400 samples from a 0–3-s event epoch; microvolts; schema-3 HDF5 `float32`.
- HDF5 SHA-256: `f6d2c7824f6aaa522286ea8ee5253b0220602117fc69d7b49c3af6e6a1324f7a`.
- **Frozen actual participant IDs:** [configs/phase3_locked_split.json](../configs/phase3_locked_split.json), imported from the user's successful Phase-2 Kaggle audit. Seed 42, **75 train / 14 validation / 15 locked test subjects**, 13,050 / 2,436 / 2,610 epochs.
- **Test embargo:** Validation-only scoring, model selection and any hyperparameter assessment. No test waveform forward passes or heldout metrics in Phase 3. Reading existing cohort metadata to reproduce the split does not authorize inspecting test scores.

## Pre-registered initial candidate methods

1. **Dummy prior:** Class prior fitted only on training labels; argmax is the majority-class prediction. Zero signal features.
2. **Riemannian:** Per-epoch OAS shrinkage covariance; Riemannian tangent reference fitted on training epochs alone; train-only balanced `LogisticRegression`, frozen fixed hyperparameters.
3. **EEGNet:** `braindecode.models.EEGNet`, randomly initialized, five classes, same 64×400 EEG windows. Training-only class weights; early stop by validation macro-F1.
4. **LaBraM Linear Probe:** Pinned `braindecode/labram-pretrained@0563b6c626e7b40d9a36653b763715db94d945d7`, 400-sample temporal-prefix adapter; frozen backbone and trainable head, with frozen-backbone evaluation mode; same train/validation participants. Does **not** test full fine-tuning.

References: [Braindecode EEGNet model docs](https://braindecode.org/stable/api.html), [pyRiemann official example](https://github.com/pyRiemann/pyRiemann), and official `src/model.py` implementation already verified in Phase 1.

## Score and evidence policy

- Primary: **validation macro-F1 and balanced accuracy**; secondary: accuracy, per-class recall, five-class confusion matrix.
- Subject-level cluster bootstrap confidence intervals on **validation subjects**, which are not independent test confidence intervals.
- Fixed random seed 42, same HDF5 byte hash and participant IDs; distinct model/config manifest and training trace.
- No fitting/normalization/tangent-space reference on validation or test. Validation used only for evaluating fixed candidates and choosing the best deep-learning checkpoint; bias must be disclosed.
- Each model writes `report.json` and *validation-only* `validation_predictions.npz`; no raw EEG, user data or test-set predictions may enter the shareable results ZIP.

## Current execution status and reproducibility

- Self-contained user-executable Kaggle notebook: **NeuroDecode_Phase3_Baselines_Kaggle.ipynb**, supplied in this conversation. It pins the audited main commit `b3af6247d027cd1a3dd33915368bad03f9bf220c`, verifies the HDF5 file hash, materializes the validation-only runner and fixed split, and runs controlled benchmarks. Because this code is not yet merged into `main`, the notebook explicitly fingerprints its full runner source SHA-256 in the output archive.
- This draft PR freezes the split and scientific protocol. **The full experimental runner is currently in the self-contained notebook artifact, not in the repository branch yet.** Its computational correctness must be validated by a full Kaggle run before promoting it to accepted project code.
- Do not commit metrics inferred from historic test data or synthetic smoke tests.

## Supervisor gate: conditions for PASS

- [ ] Kaggle logs show completed real-cache SHA validation and exact 75/14/15 participant IDs.
- [ ] Dummy, Riemannian, EEGNet and pretrained LaBraM all produce valid **validation-only** probabilities on the same 2,436 epochs.
- [ ] No signal transformations or learned covariance/tangent statistics are fitted on validation/test.
- [ ] Execution artifacts and versioned source fingerprints allow code/results reconciliation; ambiguous failures must be resolved and rerun.
- [ ] Per-class scores, confusion matrices, subject uncertainty and a reasoned scientific comparison are reviewed.
- [ ] Full executable source and tests are incorporated into a reviewed GitHub PR; CI and any real-data smoke checks pass.
- [ ] Test remains locked; Phase 4 requires the user's approval after Phase 3 gate PASS.

**No model-quality improvement is claimed until the above conditions hold.**
