# Phase 5.5a — Train-only, subject-grouped candidate optimization

> **Stage 5.5a has finished and the nominated model has already undergone its one-time 5.5b development Validation evaluation.** The final normalized EEGNet **did not improve** on the earlier EEGNet Validation Macro-F1. See [the final snapshot](final-research-snapshot.md). New model/seed tuning, original Test and Phase 6 are now closed.

**Status: PASS for single-seed 75-participant TRAIN-ONLY group cross-validation and candidate nomination, with limits (2026-10-13).** This does not certify improved accuracy on original Validation or final Test, which remain unscored under Phase 5.5a. Phase 6 has not started.

## Why this change is necessary

Historical 14-subject development Validation produced EEGNet Macro-F1 0.4191, LaBraM partial 0.2546, CBraMod probe 0.2027 and CBraMod full 0.1302; full CBraMod predicted all 2,436 epochs as `rest`. Merely trying more architectures on those same 14 people risks model-selection overfitting. EEGMMIDB events include **different run types for left/right fists vs two-fists/feet**; this task/run structure must be considered when interpreting apparent decoding. We need a **subject-held-out, train-only** model-selection stage and class-wise failure checks before requesting any new evaluation of the historical Validation.

## Frozen input and strict safety guard

The **only** permitted HDF5 is the existing 104-participant schema-3 cache, SHA-256 `f6d2c7824f6aaa522286ea8ee5253b0220602117fc69d7b49c3af6e6a1324f7a`, 18,096 epochs × 64 electrodes × 400 samples, 160 Hz, microvolts, originally bandpass filtered 8–32Hz using MOABB 1.7.2. Channel order and five-class vocabulary are unchanged. [Immutable partition manifest](../configs/phase3_locked_split.json): 75 Train/13,050 epochs, 14 Validation/2,436, 15 Test/2,610. No EEG recordings from the latter two partitions may be read in 5.5a. Metadata is used **only** for preexisting split assertions. A `TrainOnlySignalSource` rejects any non-Train row **before** HDF5 reading.

## Preregistered candidate screening (no tuning on original Validation)

Exactly **three outer GroupKFold partitions of the 75 original Train subjects**, each with 25 entirely held-out Train participants. Within the 50 outer-Train participants, a seeded subject-disjoint GroupShuffleSplit reserves **eight for inner early-stopping**, leaving **42 for CNN parameter fitting**. The inner groups, fixed seed 42 plus outer-fold number, are shared by all scratch CNN candidates. No outer-OOF group chooses CNN epoch.

Predeclared candidates (one configuration each, not a hyperparameter search):

| Candidate | Source input | Fitting |
|---|---|---|
| `bandpower` | Original uV input, **four within-8–32Hz sub-bands** 8–12, 12–16, 16–22, 22–30 Hz, 4th-order zero-phase Butterworth per epoch; log-variance for 64 channels (256 features) | Outer-Train only: StandardScaler + class-balanced LogisticRegression `C=1`, max_iter 3,000. This is **not FBCSP**. No early stopping (fits all 50 outer-Train participants). |
| `eegnet_raw` | Original 64×400 microvolt signal | Scratch Braindecode 1.5.1 EEGNet; 42 outer-fit people / 8 inner epoch-selection people. |
| `eegnet_epoch_standardized` | Per-epoch, per-channel z-score; no population statistics, tiny constant-channel epsilon | Same EEGNet architecture and training budget as raw. |
| `eegnex_epoch_standardized` | Same stateless per-epoch per-channel z-score | Scratch Braindecode 1.5.1 EEGNeX, kernel 32 and dropout .5; same budget. |

CNNs are trained without AMP (FP32), AdamW LR 0.001, weight decay 1e-4, batch 32, 25 maximum epochs, patience five **on inner Train-only subject holdout**. Class weights are calculated **only from the actual inner fit participants**. Gradient L2 clip 1.0 raises on NaN/Inf; no silent gradient skipping. Source code, preprocessing and seed are fixed before runs. The original 15-subject Test remains fully embargoed.

**Input preprocessing caution:** Original source was already 8–32Hz filtered. Filtering **short 2.5-second epochs again** can introduce edge transients, so this bandpower method is a simple exploratory comparator, not a claim of ideal filtering. The 0–3s source epoch's final 0.5s was already cropped from the immutable cache; optimizing alternate time windows requires a **separate, separately hashed new-data study**, not an unnoticed replacement of the frozen HDF5. EEGNeX and EEGNet are from scratch, **not pretrained foundation models**.

## Executable Kaggle artifact (source-locked)

Run the [20-cell NeuroDecode Phase 5.5 Train-only CV notebook](../notebooks/NeuroDecode_Phase55_TrainOnly_CV_Kaggle.ipynb) with **Kaggle GPU and Internet enabled**. Do not attach the huge old Notebook Output. Its existing-file-first original Amazon S3 preprocessing builder is unchanged from the previous approved HDF5 cache notebook; it builds in an isolated Python 3.11 MOABB environment under `/kaggle/temp` if the original complete file is absent, then fail-closes on the established full SHA-256. Optional one-HDF5 private Dataset publishing is **off by default** to avoid the recently encountered Kaggle 429 rate limit; it may be enabled explicitly after reviewing private visibility.

All Train-only benchmark code is checked out at pinned GitHub source commit
`29cd4c8cfbb987ad18aa90624b46b92c97ecc766` and its Python source SHA-256 values are asserted before any GPU training:
- `scripts/run_phase55_trainonly_cv.py` SHA-256: `f2b4388897c5c360d0ad12b6c6d8ab5fcdf5ab6ca6ffdf9c400aa98381c8bec0`.
- `src/phase55_trainonly.py` SHA-256: `16d14df18c5b02ad85ed46e9fae9d73d15bab0be257c9e852069fa21173bf6f8`.

The notebook creates `NeuroDecode_Phase55_TrainOnly_CV_Results.zip` with only aggregate protocols/reports and four **13,050-row Train-only** OOF NPZs. It excludes all raw EEG, model weights, original 14-subject Validation predictions and original 15-subject Test predictions. Selected checkpoints stay in private Kaggle output. The notebook has been syntax-compiled locally but **not yet executed in Kaggle**, so no scientific claims are derived from its prospective output.

## Completed real Kaggle Train-only candidate screen and independent verification — 2026-10-13

The owner submitted a **real full 75-Train-subject GPU run** with original
HDF5 and prelocked 3-fold protocol. The complete results ZIP SHA-256 is
`0729f0cf50c93ade6f35389258667ed9718416629b386cfc53aa118277282fde`
(10 entries; all passed ZIP CRC). The executed runner SHA-256
`f2b4388897c5c360d0ad12b6c6d8ab5fcdf5ab6ca6ffdf9c400aa98381c8bec0`
and feature adapter SHA-256
`16d14df18c5b02ad85ed46e9fae9d73d15bab0be257c9e852069fa21173bf6f8`
match the exact notebook-selected Git revision
`29cd4c8cfbb987ad18aa90624b46b92c97ecc766`.
The original immutable input hash was checked in Kaggle, and the
`protocol.json` SHA-256 was independently recomputed against `summary.json`.

**Data integrity review:** Independently re-opened all four 13,050-row
Prediction NPZs with `allow_pickle=False`. Their class order, true
labels, original global Train row indices and subject IDs were identical.
The same 75 original Train subjects are covered **exactly once** by 3×25
distinct outer folds. Each inner partition had 42 fit subjects and 8
early-stop subjects, all disjoint from outer holdouts and from the 14
original Validation / 15 locked Test participants. Five-class
probabilities are finite, nonnegative and sum to one. Every pooled
and per-fold reported macro-F1, balanced accuracy, accuracy, fixed-label
confusion matrix and all-five-class presence was recomputed exactly.

| Candidate | Train-only pooled OOF Accuracy | **Macro-F1** | Balanced accuracy | Outer-fold macro-F1s |
|---|---:|---:|---:|---|
| Bandpower + Logistic Regression | .2870 | .2544 | .2737 | .2372 / .2507 / .2620 |
| EEGNet raw | .3656 | .3212 | .3430 | .3058 / .3366 / .3158 |
| **EEGNet standardized** | **.4461** | **.3912** | **.4124** | **.3392 / .4253 / .4063** |
| EEGNeX standardized | .4312 | .3697 | .3907 | .3400 / .3774 / .3861 |

**Predetermined nomination:** `eegnet_epoch_standardized` is the
highest pooled Train-only out-of-fold macro-F1 model among noncollapsed
candidates. It improves macro-F1 over the identically trained
`eegnet_raw` variant by **+.0700** on the exact paired OOF Train rows.
All candidates predicted all five classes, so no total class collapse
exclusion was needed. The absolute Macro-F1 on Train-only OOF is **not**
the same estimand as historic EEGNet Validation macro-F1 .4191: the
participants, training sample counts and checkpoint selection differ.
Do not subtract those numbers to claim an independent accuracy gain.

**Exploratory paired 75-Train-subject bootstrap:** For each of 3,000
replicates with seed 42, participants (not epochs) were jointly
resampled, and fixed-five-class macro-F1 was computed from aggregated
five-by-five confusion matrices:

| Pooled Train-only OOF macro-F1 difference | Point | 95% paired subject percentile interval |
|---|---:|---:|
| Standardized EEGNet − Raw EEGNet | +.0700 | [+.0516,+.0901] |
| Standardized EEGNet − Standardized EEGNeX | +.0215 | [+.0096,+.0338] |
| Standardized EEGNet − Bandpower | +.1369 | [+.1174,+.1583] |

These intervals are **exploratory and selection-biased** because
the top candidate was chosen on the same OOF cohort. Also, group
CV has one fixed three-fold partition and one model seed. EEGNet per-
participant OOF macro-F1 still ranged approximately **.155–.657**,
showing considerable cross-participant heterogeneity. Do not
represent this as independently replicated robustness.

Full independently recomputed machine-readable evidence:
[phase55_trainonly_oof_audit_20261013.json](../results/validation/phase55_trainonly_oof_audit_20261013.json).
The model-weight binaries and original HDF5 were **not** distributed
in the results-only ZIP. Runtime reports attest checkpoint
save/reload and HDF5 SHA, rather than external binary-weight re-execution.

**Decision:** Phase **5.5a passes its narrowly defined single-seed
Train-only candidate-selection gate**, not a final-performance gate.
A future separately implemented **5.5b** must freeze a single candidate
(`eegnet_epoch_standardized`), define an all-Train
fit/inner-early-stop protocol before opening any original 14-subject
Validation EEG, permit exactly one controlled historic-Validation run
for that candidate, and leave all 15 Test subjects embargoed. No further
architecture selection on historic Validation.

## Model choice and reporting

Every candidate must return all **13,050 Train-only out-of-fold rows** exactly once. Metrics: fixed-class five-way Macro-F1 (primary), balanced accuracy, all five class recalls, class counts, confusion matrices and participant-level Macro-F1. An all-one-class output is explicitly labeled **class collapse** and ineligible for the nominal winner regardless of its apparent overall accuracy. Among noncollapsed models select highest pooled out-of-fold Macro-F1; tie-break balanced accuracy, then alphabetical model name. This is **TRAIN-ONLY SELECTION**, and not a reliable final generalization estimate: the same Train cohort ultimately nominates the candidate. Differences between convex bandpower classifier (50 outer-train subjects) and early-stop CNN training (42+8) are declared, so this is a **practical fixed-protocol comparison**, not a parameter-matched architecture ablation.

The returned ZIP contains only `protocol.json`, `summary.json`, candidate reports and NPZs of Train-OOF five-class predictions; no original signal waveforms, no large model weights, no historical Validation or Test predictions. Save the Kaggle notebook version privately if independently reloading CNN weight files is needed. Only after **independently verifying** the complete Train-OOF ZIP can we pre-register a **single chosen candidate** for subsequent historic Validation evaluation. Further model additions are a different preregistered protocol; never retroactively fish on the 14-subject Validation.

**Scientific acceptance gate: PASS WITH LIMITATIONS, specific to Train-only OOF model-selection integrity.** All predeclared candidate predictions, held-out Train subjects, exact source/hash, fixed scoring and real EEG gradient CI were checked. The historic 14-participant Validation and 15-person Test were **not evaluated** during this stage. Phase 5.5b requires a separately locked protocol; Phase 6 is **not authorized**.
