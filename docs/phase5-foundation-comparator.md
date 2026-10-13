# Phase 5 — Additional pretrained EEG foundation comparator: selection and gate

**Status: single-seed development Validation-only execution PASS WITH DECLARED LIMITATIONS (2026-10-13).** CBraMod failed to outperform the earlier EEGNet or best LaBraM in this experiment; the full-tuning variant suffered complete majority-rest prediction collapse. These are negative results, not decoding success. The locked Test, multi-seed assessment and Phase 6 remain closed.

## Pretrained competitor feasibility audit

- **CBraMod:** Selected as the first *conditional* foundation comparator. Official [Braindecode 1.5.1 model](https://braindecode.org/stable/generated/braindecode.models.CBraMod.html) is channel-agnostic and has public pretrained [safetensors](https://huggingface.co/braindecode/cbramod-pretrained). Pin HF revision `584cdc415913739a05d84bf0c1cb3db397764507` and exact safetensors SHA-256 `a939ace9aa1e229f08391ad8bb2d197b507ae2c519a50addf087f0151b2df5c3`. The checkpoint is BSD-3-Clause tagged; the [ICLR 2025 paper](https://openreview.net/forum?id=NPNUHgHF2w) discusses pretrained input scaling.
- **EEGPT:** Weight-bearing pretrained checkpoint exists, but its source-pretraining channel/time/frequency assumptions (62 channels and ~250–256 Hz) would require additional mapping and provenance tests. **Deferred; do not train randomly initialized EEGPT and call it pretrained.** [Braindecode pretrained matrix](https://braindecode.org/stable/auto_examples/model_building/plot_load_pretrained_models.html).
- **BIOT, BENDR, CodeBrain:** Real pretrained weights exist, but source channel counts (typically 16–20) and sampling assumptions require explicit electrode-subset/source-preprocessing validation; **deferred**, not declared unsuitable for all EEG tasks.

These choices are feasibility screening, **not a ranked accuracy comparison**. All pretrained weights must be loaded with explicit SHA-256 checks. No arbitrary `strict=False` silent missing backbone tensors are permitted.

## Input adapter — predeclared, nonlearned, inspectable

**Frozen source:** The Phase-2 HDF5 SHA-256 `f6d2c7824f6aaa522286ea8ee5253b0220602117fc69d7b49c3af6e6a1324f7a`; 104 subjects, 18,096 samples × 64 channels × 400 samples, 160 Hz, microvolts, imagery-only, five-class label vocabulary. [Frozen split](../configs/phase3_locked_split.json) 75 Train (13,050), 14 Validation (2,436), **15 Test (2,610) locked**.

**CBraMod input:** Preserve the **entire same 2.5-s EEG recording window** per epoch. Apply `scipy.signal.resample_poly(..., up=5, down=4)` to convert 160→200 Hz: 64×400 → 64×500. Normalize input amplitude by **100 μV per unit** per original CBraMod pretraining protocol, then append exactly **100 samples of zero** (0.5 s) to satisfy a 200-sample/patch, 3-patch input of 64×600. No data-derived normalization is fitted on Validation or Test, and the adapter neither interpolates channels nor crops original signal information.

**Scientific caveat:** This is *not* architecture-identical input. Zero-padding introduces a third partially empty patch, and the original source is prefiltered to 8–32 Hz. CBraMod was pretrained on data with distinct prefilters and 200-Hz timebase. These are material model-transfer distribution shifts and must be disclosed in reported comparisons, not hidden as perfect apples-to-apples control.

## Preregistered first comparison

1. **Frozen-pretrained CBraMod probe:** Strictly load all safetensors backbone tensors, keep pretrained backbone in `.eval()` throughout train, then train a fresh 5-class `LayerNorm(200)+Linear(200,5)` head on *pooled encoder features*. No source or Validation waveforms enter the output files.
2. **Full-pretrained CBraMod adaptation:** Same pinned backbone, train all its weights with LR `3e-5` and head LR `1e-3`, weight decay `1e-4`; selected epoch by Validation macro-F1.

Training seed 42, microbatch 8 (to cap T4 GPU use), up to 10 epochs and patience 3. Both modes use train-only five-class balanced loss weights. No grid search or Test evaluation. Save best checkpoints privately; export only Validation predictions, audit metadata and per-class metrics. This is *one development seed*, not a multi-seed result. The model is not compared to LaBraM based on raw accuracy alone; primary criteria are **macro-F1 and balanced accuracy**.

**Technical note:** The CBraMod Hub config does not encode pretrained signal geometry (`n_chans`/`n_times`/`sfreq` are null). We construct an explicitly dimensioned encoder, verify every safetensors key and shape and load it **strictly**; if this fails under Braindecode 1.5.1, the stage **must stop** and the discrepancy must be audited. A random fallback is prohibited.

## Verified public pretrained-weight feasibility smoke — 2026-10-13

The [public GitHub Actions run](https://github.com/mohamad679/labram-mi-project/actions/runs/38366074365) succeeded using **participant 3, which belongs to frozen Train**. All six original imagery EDF runs were staged from PhysioNet S3. CBraMod weights at pinned revision
`584cdc415913739a05d84bf0c1cb3db397764507` matched the official safetensors
SHA-256 `a939ace9aa1e229f08391ad8bb2d197b507ae2c519a50addf087f0151b2df5c3`,
and **all 211 original pretrained backbone tensors were strictly loaded** without dropping keys. An actual (64,600) transformed Train epoch produced five-class logits; the frozen backbone received no gradient, and the newly initialized classifier was updated by SGD.

Machine-readable report: [phase5_cbramod_pretrained_smoke_20261013.json](../results/validation/phase5_cbramod_pretrained_smoke_20261013.json).

**Not yet established:** that either CBraMod probe or fine-tuning has a useful full-cohort 14-participant Validation Macro-F1; the above single-epoch/single-step smoke cannot establish convergence, performance or superiority.

## Kaggle notebook prepared for full 104-subject Validation experiments

[NeuroDecode_Phase5_CBraMod_AutoS3_PrivateDataset_Kaggle.ipynb](../notebooks/NeuroDecode_Phase5_CBraMod_AutoS3_PrivateDataset_Kaggle.ipynb) contains 20 cells, including the public S3 HDF5 reconstruction. It checks out the pinned **pre-PR execution source commit**
`561870b2604d2f00198168049164337a5b9e1b16`,
verifies both exact Python source byte digests, loads only the already saved Phase-2 HDF5
and downloads only the pinned pretrained safetensors. The Kaggle experiment is not the Ruff-reformatted later GitHub source; its exact runner digest is
`ad14418845d5c16c0a3a9904c6edac7d5a55c4eaa9ed3f1b2ffe6c840ce3f69e`.
It writes **only Validation metrics and predictions** into a shareable ZIP, excluding raw EEG or checkpoint binaries.

## Kaggle no-Input workaround: original public S3 HDF5 builder and one-file private Dataset

To avoid the old notebook's 13,586-file Output and Kaggle `ListKernelSessionOutput`
429 rate-limit, a **new** [20-cell Phase-5 Auto-S3 / private-Dataset
notebook](../notebooks/NeuroDecode_Phase5_CBraMod_AutoS3_PrivateDataset_Kaggle.ipynb)
does **not use** the Kaggle Notebook-Output API or require old Add Input.
The base 12-cell Phase-5 pretrained-CBraMod runner and its locked hash are
preserved (only its temporary Git clone path is moved for resumability).

- If the verified 104-subject HDF5 exists already in Working or as a single
  mounted input, use it directly. Otherwise run the **literal unchanged** S3
  downloader embedded from the original user-executed cache-builder notebook
  (standalone builder SHA-256
  `a6943fdc0a677777ff429717257d9897d220bea97bc51bec35388e2ec43b9073`).
- The downloader fetches 624 original imagery EDF runs from official unsigned
  Amazon S3, processes them under independently isolated **Python 3.11,
  MOABB 1.7.2 and MNE 1.13.2**, with the original immutable preprocessing
  Git revision `74e511fb9c1ac4f0543b249b571e28e811ec92c3`. It
  durably commits one participant at a time; downloaded EDFs are deleted only
  after an HDF5 append succeeds. Runtime depends on the Kaggle connection;
  the prior fast build duration is **not guaranteed**.
- The final cache must match precisely
  `f6d2c7824f6aaa522286ea8ee5253b0220602117fc69d7b49c3af6e6a1324f7a`
  or **CBraMod training stops**. Scientific split and Test embargo do not change.
- To avoid massive Notebook Outputs, raw EDFs, the isolated environment, and
  Git working copies live under **`/kaggle/temp`**. Only the final HDF5,
  short logs, and selected experiment artifacts are in `/kaggle/working`.
- An optional `UPLOAD_PRIVATE_DATASET=True` cell creates
  **only** a hard-linked `physionet_mi_104subjects.h5` and Kaggle
  `dataset-metadata.json` in a dedicated folder, then calls
  `kaggle datasets create` **without `--public`**. The ephemeral
  hard-link is unlinked afterwards to avoid duplicate Notebook Output files.
  Kaggle's default is private; users must **confirm visibility in Your
  Datasets**. Upload is not guaranteed to succeed on Kaggle rate limits,
  authentication failure, timeout, or an existing Dataset slug; failure
  does not permit training on a nonmatching HDF5 nor automatically retry.
  No live dataset publication is claimed from the local syntax test.
- Model runner content and validation-only code are unchanged relative to
  the initial Phase-5 notebook. The new notebook passed local syntax and
  source-identity checks, **not yet a fresh full Kaggle S3+GPU run**.

Kaggle CLI reference: https://github.com/Kaggle/kaggle-cli/blob/main/docs/datasets.md

## Real full-cohort CBraMod Kaggle outcome — 2026-10-13

The owner executed the Validation-only CBraMod probe and full-tuning experiment
with a fixed Phase-2 HDF5 and source-defined 75/14/15 participant split.
The first attempt using automatic mixed precision hit **nonfinite gradient
norms**; it was correctly aborted by `error_if_nonfinite=True`.
A replacement run launched the same immutable runner **without `--amp`**
(full FP32, as requested in the user-executed replacement cell), without
weakening the gradient guards. The exported ZIP is SHA-256
`4dfd594fceb9066b543bd1ffbaec17a0dfe192800a0efa6e65e9b792d8fb63de`.
The exported reports do **not** contain an explicit `amp_enabled` metadata
field; the FP32 designation depends on the actual user-executed command,
not on the ZIP alone.

The reported source runner digest is
`ad14418845d5c16c0a3a9904c6edac7d5a55c4eaa9ed3f1b2ffe6c840ce3f69e`,
and its immutable source revision is
`561870b2604d2f00198168049164337a5b9e1b16`.
The reported official CBraMod pretrained safetensors SHA is
`a939ace9aa1e229f08391ad8bb2d197b507ae2c519a50addf087f0151b2df5c3`.
Every source input was verified during the run as HDF5 SHA-256
`f6d2c7824f6aaa522286ea8ee5253b0220602117fc69d7b49c3af6e6a1324f7a`.
The source preprocessing and 15-person Test embargo remain unchanged.

**Independent evidence audit:** The seven ZIP entries passed archive CRC
validation. Both Prediction NPZs were read without object deserialization;
each has exactly 2,436 Validation rows, 14 correctly frozen subject IDs,
identical truth and sample order, fixed five-class output order, and finite,
nonnegative probabilities summing to 1.0. Independently recomputed
accuracy, macro-F1, balanced accuracy and all 5×5 confusion counts exactly
match Kaggle report JSON. The NPZ sample/subject ordering also matches the
original Phase-3 EEGNet and Phase-4 LaBraM Prediction archives. **No held-out
Test participant predictions appear in the new archive.**

| Model / strategy | Validation accuracy | Macro-F1 | Balanced accuracy | Best epoch |
| --- | ---: | ---: | ---: | ---: |
| EEGNet (Phase 3, same 14 subjects) | .4799 | **.4191** | **.4418** | 24 |
| LaBraM pretrained partial (Phase 4) | .3859 | .2546 | .2618 | 7 |
| LaBraM pretrained full (Phase 4) | .3485 | .2481 | .2615 | 6 |
| **CBraMod pretrained probe** | .2779 | .2027 | .2091 | 4 |
| **CBraMod pretrained full** | .4828 | **.1302** | **.2000** | 1 |

**Critical failure mode:** The CBraMod **full-tuning** selected model
predicted `rest` for **all 2,436 Validation epochs**, with exact predicted
class counts `[0,0,0,2436,0]` and recall 0.0 for feet, hands, left-hand
and right-hand imagery. Its superficially higher accuracy .4828 is
**identical to the majority-class dummy accuracy**, so this is a
**classification collapse**, not an improvement. Full mode nonetheless
reported nonzero backbone gradient and a changed backbone tensor digest,
so the model *was* trained; training did not yield a useful classifier.

The CBraMod **probe** predicted all five labels, with Macro-F1 .2027.
Its backbone checksum was byte-identical before and after training,
while both strategies loaded the same initial pretrained backbone.
For both candidate checkpoints the user-executed Kaggle report says
`checkpoint_reloaded=true` and
`max_abs_probability_reload_difference=0.0`.
Because weights were excluded from the user ZIP, the reviewer has
**not independently reloaded the checkpoint binaries**.

**Exploratory paired 14-subject cluster-bootstrap**, 2,000 replicated
resamples of subject IDs with seed 42, using Validation predictions:

| Difference in macro-F1 | Point difference | 95% development interval |
| --- | ---: | ---: |
| CBraMod Probe − CBraMod Full | +.0724 | [+.0558,+.0904] |
| LaBraM Partial − CBraMod Probe | +.0519 | [+.0305,+.0734] |
| EEGNet − CBraMod Probe | +.2164 | [+.1494,+.2772] |

The same Validation cohort selected the best epoch for each candidate.
These are **development intervals with selection bias**, not independent
Test confidence intervals. The approaches also have materially different
spectral/temporal transforms (CBraMod uses 200-Hz resampling,
amplitude scaling, and 100 zero-padded samples), so differences are
a *descriptive practical comparison*, not a strictly controlled model
architecture ablation. All results are **one model training seed**.

Machine-readable aggregate review:
[phase5_cbramod_validation_seed42_20261013.json](../results/validation/phase5_cbramod_validation_seed42_20261013.json).
This document intentionally **does not infer that a private Kaggle
Dataset was uploaded successfully**: that is separate from training and
is not evidenced in the submitted benchmark ZIP.

**Scientific conclusion:** We successfully audited a genuinely pretrained
additional foundation comparator on the exact same subject-disjoint
development Validation, with no claimed held-out Test inference.
This phase is a **scientifically valid negative result** under the
declared preprocessing/training conditions. CBraMod did *not* beat
EEGNet or the best fine-tuned LaBraM; its full-tuning failure is a
strong reason to avoid claiming performance success or blindly
increasing the model benchmark roster.

## Scientific supervisor gate — final disposition

- [x] Public pretrained checkpoint hash and all 211 pretrained backbone tensors strictly loaded in real public Train subject-3 smoke.
- [x] Offline input adapter regression tests and latest GitHub CI passed.
- [x] Full frozen 104-subject HDF5 and exact immutable 75/14/15 subject split checked during completed Kaggle execution.
- [x] Probe/full completed and emitted independently audited same-order Validation-only predictions for 2,436 epochs each; **Test stayed locked**.
- [x] Reported selected checkpoints SHA-hashed and reloaded in Kaggle; exported probabilities and 5×5 confusion matrices independently recomputed. Binary checkpoints were not supplied.
- [x] Frozen Probe backbone unchanged, Full Backbone weights and gradient changed; **Full classifier collapse explicitly disclosed**.
- [x] Same-cohort (same prediction-row order) comparisons to EEGNet and LaBraM reviewed with development/selection caveats.
- [x] Immutable executed source commit, run SHA, pretrained SHA, input hash and scientific findings archived.

**Gate: PASS WITH DECLARED LIMITATIONS ONLY for the Phase-5 first
pretrained-competitor feasibility and exploratory development-Validation
comparison. No pretrained CBraMod model-quality PASS is claimed.**
No multi-seed, no final independent Test and **Phase 6 requires separate
user authorization**.
