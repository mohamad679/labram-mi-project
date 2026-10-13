# Phase 5 — Additional pretrained EEG foundation comparator: selection and gate

**Status: IN PROGRESS / GATE OPEN.** Previous phases 1–4 passed their explicitly limited scientific validation gates. A new pretrained competitor must provide genuine weights, a validated adaptation from the *same* 104-subject input contract, and an independently reviewed real Kaggle run. Do **not** claim benchmark superiority or unlock Test from this proposal alone.

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

## Scientific supervisor gate

- [ ] Offline adapter checks and public pretrained-weight authenticity checks PASS.
- [ ] Real public **Train participant 3** forward/backward and frozen-backbone gradient test PASS in GitHub CI (do not touch held-out Test participant 1 for this smoke).
- [ ] Kaggle run uses same HDF5 SHA and exact 75/14/15 participant split, with zero Test waveform access.
- [ ] Both probe/full candidates produce valid 2,436 Prediction rows with correct 5-class order and complete frozen subject metadata.
- [ ] Selected model checkpoints are SHA-hashed, reloaded and numerical predictions reproduced.
- [ ] Independent reconstruction of Validation F1/balanced accuracy and model initialization fingerprint; review participant bootstrap with selection-bias caveats.
- [ ] All code, notebook, provenance, model-card source and CI green on a reviewable PR; merge only after real Kaggle evidence.

**No Test inference, no Phase 6, and no PASS yet.**
