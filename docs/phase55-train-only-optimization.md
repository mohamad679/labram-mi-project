# Phase 5.5a — Train-only, subject-grouped candidate optimization

**Status: IMPLEMENTED IN DRAFT / real 104-person Kaggle Train-only CV evidence PENDING.** This research extension comes **after** Phase 5 and **before** any Phase 6; it does not revise the Phase 1–5 historical experiments, claim improved EEG accuracy, or access original Validation (14 subjects) or Test (15 subjects) waveforms.

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

## Model choice and reporting

Every candidate must return all **13,050 Train-only out-of-fold rows** exactly once. Metrics: fixed-class five-way Macro-F1 (primary), balanced accuracy, all five class recalls, class counts, confusion matrices and participant-level Macro-F1. An all-one-class output is explicitly labeled **class collapse** and ineligible for the nominal winner regardless of its apparent overall accuracy. Among noncollapsed models select highest pooled out-of-fold Macro-F1; tie-break balanced accuracy, then alphabetical model name. This is **TRAIN-ONLY SELECTION**, and not a reliable final generalization estimate: the same Train cohort ultimately nominates the candidate. Differences between convex bandpower classifier (50 outer-train subjects) and early-stop CNN training (42+8) are declared, so this is a **practical fixed-protocol comparison**, not a parameter-matched architecture ablation.

The returned ZIP contains only `protocol.json`, `summary.json`, candidate reports and NPZs of Train-OOF five-class predictions; no original signal waveforms, no large model weights, no historical Validation or Test predictions. Save the Kaggle notebook version privately if independently reloading CNN weight files is needed. Only after **independently verifying** the complete Train-OOF ZIP can we pre-register a **single chosen candidate** for subsequent historic Validation evaluation. Further model additions are a different preregistered protocol; never retroactively fish on the 14-subject Validation.

**Scientific acceptance gate:** Github Ruff/pytest and genuine public Train subject S003 EEG gradient smoke PASS; same immutable HDF5 hash; exactly 75 Train subjects in CV; no validation or Test waveform; all four 13,050-row OOF NPZs independently audited; per-class and per-subject statistics; precise optimization SHA; no nonfinite updates; code documentation. Until the Kaggle run and audit are complete, status remains **OPEN**, PR **Draft**, Phase 6 **not authorized**.
