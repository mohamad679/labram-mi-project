# Phase 5.5b — single nominated normalized EEGNet, one-time development Validation

**Status: COMPLETED, independently audited, and scientifically closed (2026-10-13).** Validation macro-F1 **0.4125** did **not** improve on the historical EEGNet macro-F1 **0.4191**; this is a valid neutral/negative development result, not evidence of model superiority. The owner stopped further model/seed experiments. The 15-subject Test remains untouched and Phase 6 will not run.

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

The completed real 104-person GPU execution and small results-only ZIP were supplied by the owner on 2026-10-13. CI independently establishes code/test and one-step public-Train EEG feasibility. Saved checkpoint binaries remain private in Kaggle and are **not independently reloaded from the results-only ZIP**. Reported Kaggle checkpoint reload must be distinguished from independent evidence review.

## Exact user-executable source pin and reproducible notebook

The 20-cell [Kaggle execution notebook](../notebooks/NeuroDecode_Phase55b_OneTime_Validation_Kaggle.ipynb) pins the **pre-notebook executable source commit** `e160e10cc195ac627f9afbd1b155d07eda71a4c5`, and independently SHA-checks:

- Runner `scripts/run_phase55b_validation.py`: `116bb70a8a988400b29ba120d79bad4ef39d3a42cc1a86a1e762570b57b0b13d`.
- Locked partition/nomination guards `src/phase55b_protocol.py`: `806fa50527801e54f8df83e59c607f91a093b00193484e519a064a737ba33205`.
- Original selected per-epoch normalizer `src/phase55_trainonly.py`: `16d14df18c5b02ad85ed46e9fae9d73d15bab0be257c9e852069fa21173bf6f8`.

All three pull-request GitHub CI workflows passed on notebook commit `2b4461d9134c5b53c0a286ed24909bfa91b6ba80`: Ruff + 134 unit tests (2 skipped), documentation integrity, and authentic public Train-S003 normalized-EEGNet gradient smoke. These are **prerequisite engineering gates**, not evidence of final 104-person training or held-out Validation performance. The actual Kaggle result ZIP has now been independently audited; the acceptance gate is complete with explicit limitations.

## Actual Kaggle run — independently verified final development outcome

The supplied ZIP `NeuroDecode_Phase55b_OneTime_Validation_Results.zip` has
SHA-256 `8e88e4463d1fa96856cb127f57e4a706042ff441095ccb1c1ad5d9109f635391`
and passed its ZIP CRC checks. The embedded `report.json` records the locked
source hashes, the exact **67 parameter-fit / 8 Train-only early-stopping**
participants, FP32 finite-gradient updates, 12 completed epochs and best
inner-selected checkpoint **epoch 7**. The checkpoint was saved and reloaded
in Kaggle with a reported zero probability discrepancy on inner Train.
The checkpoint binary was not in the shareable ZIP and has not been
independently reloaded by the reviewer.

The `validation_predictions.npz` (loaded without pickle) contains exactly
2,436 unique frozen-Validation row identifiers, exactly 14 reserved subject
IDs (174 epochs each), no Test subject IDs, five finite normalized class
probabilities per row, and all five classes represented in predictions.
Independently calculated metrics and confusion matrix match the Kaggle report:

| Model and cohort | Accuracy | Macro-F1 (primary) | Balanced accuracy |
|---|---:|---:|---:|
| Original EEGNet historical Validation reference | .4799 | **.4191** | see Phase 3 evidence |
| **Nominated normalized EEGNet** on the same 14-subject Validation | **.4885** | **.4125** | **.4236** |

Per-class F1 for `feet / hands / left_hand / rest / right_hand`:
`.2551 / .3604 / .4414 / .6593 / .3466`.
Its original 14-subject cluster bootstrap 95% interval for macro-F1 is
`[.3486,.4847]`, exploratory and not multi-seed.
The model does not collapse onto one class, but performance varies
substantially across subjects (subject 58: .2181; subject 15: .7036).

**Conclusion:** The Train-only model-selection experiment found a positive
normalized-versus-raw EEGNet OOF difference in the **75 original Train**
subjects; **the nominated normalized variant did not beat the historical
raw EEGNet on the already-exposed original 14-subject Validation**.
The difference is **−0.00652 Macro-F1** (normalized minus historic);
as checkpoints/training protocols differ, this should **not** be sold as a
strictly isolated effect of normalization. The 15 original Test participants
were **not evaluated**. The owner chose to stop, preserve all negative
evidence, and close further optimization and multi-seed work. Do not
fabricate a final Test score or advertise confirmed cross-subject
generalization.

See the [machine-readable independent evidence](../results/validation/phase55b_one_time_validation_seed42_20261013.json).
## Acceptance gate

- [x] GitHub CI lint/test/docs and genuine public Train EEG smoke all pass at final head.
- [x] HDF5 SHA-256, source SHAs, nominated candidate, exact 67/8 inner subject allocation, and original 14/15 heldouts confirmed in real GPU run.
- [x] Training finishes in FP32, finite gradients, one inner-Train-selected checkpoint SHA and **numerically reproduced early-stop predictions**.
- [x] Exactly 2,436 same-order original Validation predictions, finite normalized probabilities, five-class metrics and confusion matrix independently recomputed, subject-level uncertainty reported.
- [x] No historical Validation-based model selection and no Test inference. Original Validation is a development set, not final confirmatory Test.
- [x] Archive scientific results and limitations on PR before authorizing merge; **Phase 6 starts only after independent review and separate approval**.

**Gate: PASS WITH EXPLICIT LIMITATIONS for faithful one-time development Validation execution and independent prediction-metric verification; NO model-superiority PASS and NO final Test-verified conclusion. Additional seeds, architectures and Test evaluation are CLOSED by owner instruction.**
