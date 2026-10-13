# Phase 5.5b — single nominated normalized EEGNet, one-time development Validation

**Status: prepared and awaiting the first real full-cohort Kaggle execution. No Phase-5.5b performance claim yet.** Phase 6 and all 15 held-out Test participants remain closed.

## Why this run is scientifically bounded

Phase 5.5a's **Train-only** 75-subject three-fold out-of-fold comparison selected `eegnet_epoch_standardized` (pooled Train OOF macro-F1 **0.3912236253**, vs **0.3211742663** for raw EEGNet), as independently documented in [`phase55_trainonly_oof_audit_20261013.json`](../results/validation/phase55_trainonly_oof_audit_20261013.json). This selection did **not** use the 14 historical Validation or 15 Test subjects.

The only question for Phase 5.5b is how this *preselected* normalized EEGNet generalizes to the **unchanged** 14 historical development Validation subjects. No other architecture, post-hoc threshold, tuning grid, or alternative preprocessing may be chosen based on this outcome. The existing original Phase-3 EEGNet historical Validation macro-F1 was **0.4190536187**; this is a descriptive, same-cohort but **different-training-protocol** reference, **not** an isolated normalization ablation.

## Freeze before opening historical Validation

The exact recorded settings appear in [the frozen configuration](../configs/phase55b_final_fit_protocol.json). The model, preprocessing and budget replicate Phase 5.5a's EEGNet-normalized candidate, without making up a new hyperparameter search:

- Original immutable Phase-2 104-participant EEGMMIDB schema-3 HDF5 SHA-256: `f6d2c7824f6aaa522286ea8ee5253b0220602117fc69d7b49c3af6e6a1324f7a`. EEG epochs `(64,400)` at 160 Hz, 8–32 Hz, µV; five fixed labels `feet`, `hands`, `left_hand`, `rest`, `right_hand`.
- Frozen Phase-3 participant partitions: 75 Train (13,050 epochs), 14 historical Validation (2,436), 15 **embargoed Test** (2,610). A waveform-source permit explicitly rejects every index not in its specific authorized partition.
- Fixed Train-only inner `GroupShuffleSplit(random_state=142, test_size=8)` on the **sorted 75 original Train participant IDs**. The **8 Early Stopping subjects are [9,25,56,61,70,81,83,90]**; the other **67 Train subjects fit parameters**. All 14 historical Validation and 15 Test subject IDs are disjoint. No model or train-fitted parameter is optimized on original Validation.
- Scratch `braindecode==1.5.1` EEGNet, `(64,400)`, 160 Hz, five-class output; **per-epoch, per-electrode z-score** identically to Phase 5.5a's `src.phase55_trainonly.epoch_standardize` (no cohort-wide statistics).
- Seed **42**, AdamW LR **0.001**, weight decay **1e-4**, microbatch **32**, maximum **25** epochs, patience **5**, class-balanced CrossEntropyLoss using labels of **only the 67 fitted subjects**. Finite-loss guard + FP32 finite-gradient L2 clip at **1.0**; never set `error_if_nonfinite=False` or silently fall back to random weights/alternative hyperparameters.
- **Checkpoint selection: highest inner-Train 8-subject macro-F1**. The selected checkpoint is persisted, SHA-256 recorded and strictly reloaded; its inner-Train probabilities must reproduce the selected epoch to within `1e-6`.
- **Exactly one historical Validation EEG forward pass after checkpoint is fixed**. Report accuracy, macro-F1, balanced accuracy, per-class F1/recall, full confusion matrix, class collapse, and subject-cluster 2,000-replicate exploratory percentile intervals. All probabilities and participant IDs are exported; original signal waveforms and checkpoint binaries are not in shareable ZIP.
- **Zero Test waveform access/inference**. Do not tune the model again on the 14 Validation subjects if the result disappoints. Multiple reruns or other architectures require explicitly disclosed new protocol and validation-selection caveats.

## Kaggle provenance and reproducibility

The [Phase 5.5b Kaggle notebook](../notebooks/NeuroDecode_Phase55b_OneTime_Validation_Kaggle.ipynb) uses the proven official public Amazon S3 -> six imagery EDFs per participant -> source-locked MOABB 1.7.2/MNE 1.13.2 HDF5 cache builder *only if* the original verified HDF5 is missing. It avoids the old Notebook Output attachments and the rate-limited `kaggle kernels output` endpoint. The notebook performs strict Git commit and runner/protocol/preprocessing SHA-256 checks, runs one GPU training job, and publishes one results-only ZIP `NeuroDecode_Phase55b_OneTime_Validation_Results.zip`. Kaggle GPU and Internet must be enabled. Optional single-HDF5 private Dataset publishing is disabled by default; Kaggle 429 rate limits must not trigger uncontrolled retries.

The *real* 104-person GPU execution is **not supplied yet**. CI can establish code/test and one-step public-Train EEG feasibility only. Saved checkpoint binaries remain private in Kaggle and are **not independently reloaded from the results-only ZIP**. Reported Kaggle checkpoint reload must be distinguished from independent evidence review.

## Acceptance gate

- [ ] GitHub CI lint/test/docs and genuine public Train EEG smoke all pass at final head.
- [ ] HDF5 SHA-256, source SHAs, nominated candidate, exact 67/8 inner subject allocation, and original 14/15 heldouts confirmed in real GPU run.
- [ ] Training finishes in FP32, finite gradients, one inner-Train-selected checkpoint SHA and **numerically reproduced early-stop predictions**.
- [ ] Exactly 2,436 same-order original Validation predictions, finite normalized probabilities, five-class metrics and confusion matrix independently recomputed, subject-level uncertainty reported.
- [ ] No historical Validation-based model selection and no Test inference. Original Validation is a development set, not final confirmatory Test.
- [ ] Archive scientific results and limitations on PR before authorizing merge; **Phase 6 starts only after independent review and separate approval**.

**Gate: OPEN.**
