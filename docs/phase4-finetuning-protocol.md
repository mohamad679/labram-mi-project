# Phase 4 — Pretrained LaBraM partial/full fine-tuning scientific gate

**Status: OPEN, independent evidence pending.** No claim of Phase 4 PASS or performance improvements until a complete real 104-subject GPU experiment has been submitted and independently reconciled. The 15 frozen Test subjects remain embargoed.

## Hypothesis and fixed inputs

Phase 3 found a single-seed (42) Validation macro-F1 of **0.4191 for EEGNet** and **0.1861 for a frozen-backbone pretrained LaBraM probe**. This is development evidence only; it does **not** establish whether trainable LaBraM backbone layers can adapt to motor imagery.

- PhysioNet EEGMMIDB v1.0.0, imagery-only, 104 retained participants, 18,096 epochs, 64 × 400 float32 µV values.
- Immutable HDF5 SHA-256 `f6d2c7824f6aaa522286ea8ee5253b0220602117fc69d7b49c3af6e6a1324f7a`.
- [Exact frozen subject split](../configs/phase3_locked_split.json), seed 42: 75 train (13,050 epochs), 14 Validation (2,436), **15 Test (2,610) untouched**.
- Pretrained source `braindecode/labram-pretrained` at immutable revision `0563b6c626e7b40d9a36653b763715db94d945d7`, with the previously audited temporal-prefix transfer to 400 samples. The 64-to-128-channel interpolation is inherited from Braindecode's adapter, never silently replaced.

## Fixed model regimes and control

All **three** initial candidates use this same pinned pretrained checkpoint (no scratch in this phase):

1. **Probe control:** `final_layer` only; entire frozen backbone in evaluation mode.
2. **Partial fine-tune:** the last **2** modules in `blocks`, terminal `norm`/`fc_norm` if trainable, and the final head. All earlier transformer blocks, patch/temporal embeddings and interpolation weights remain frozen.
3. **Full fine-tune:** every trainable model parameter, including initial embedding layers, transformer blocks and the task head.

The predeclared optimization budget and shared conditions are fixed in `scripts/run_phase4_finetune.py`: model seed 42, microbatch 16, accumulation 2, effective batch ~32, head learning rate `1e-3`, partial backbone `1e-4`, full backbone `3e-5`, AdamW weight decay `1e-4`, clipping 1.0, 15 maximum epochs, patience 5 (Validation macro-F1). Backbone learning-rate differences are **part of the declared strategy**, not post-hoc hyperparameter optimization. All use class weights computed from Train labels alone.

No scoring, training, or forward evaluation of Test signals is permitted. The cohort metadata alone can be read to validate that the split is frozen. Each validation-only best checkpoint is SHA-256 hashed, reloaded and compared numerically against the selected probabilities. For partial/probe, frozen parameters must be exactly unchanged.

## Reproducibility evidence and constraints

- Lightweight offline tests verify the exact unfrozen module names, gradients, SGD updates and frozen-layer invariance on a small synthetic module. These tests do **not** establish results on actual LaBraM.
- A separately required [real PhysioNet pretrained-gradient CI gate](../.github/workflows/phase4-gradient-audit.yml) executes all three regimes with actual public EEG and the official pinned LaBraM checkpoint. It demonstrates gradient paths and optimizer updates; it does **not** prove training convergence or classification accuracy.
- Full GPU training is provided by [Phase 4 Kaggle notebook](../notebooks/NeuroDecode_Phase4_Finetune_Kaggle.ipynb). It discovers the already verified 104-subject HDF5 input in Kaggle; it must **not redownload** this large cache. Outputs include per-epoch losses, validation predictions, per-subject bootstraps, model and input hashes, frozen-weight checks and reload evidence.
- Selected model state files can be large, so the small shareable results ZIP deliberately omits all checkpoint binary files and all EEG waveforms. Preserve the private Kaggle output version if independent weight-level re-evaluation is needed.

## Scientific supervisor criteria for Phase 4 PASS

- [ ] Independent repository Ruff/pytest checks pass on final PR head.
- [ ] Real pretrained-model gradient-flow smoke on **genuine PhysioNet EEG** proves probe-only, last-2-block, and full-backbone optimizer invariants.
- [ ] Kaggle verifies full HDF5 SHA, frozen 75/14/15 participant split, and no Test waveform access.
- [ ] All three modes finish with valid, finite, same-order 2,436-epoch Validation predictions, real training gradients and saved/reloaded selected checkpoints.
- [ ] Frozen-parameter hashes demonstrate no accidental optimization in probe/partial; same source pretrained early block hash recorded across modes.
- [ ] Validation macro-F1, balanced accuracy, five-class confusion matrices and participant-level uncertainty independently recalculated.
- [ ] Exact executed code, dependency versions, artifacts and scientific limits are committed and reviewed.

**Phase 4 PASS, when justified, only certifies reproducible single-seed development comparison.** Multi-seed robustness, untouched Test inference and scientific publication claims must be evaluated independently in later stages. Do not use the Phase-3 Validation leaderboard or historical Test to justify changing the frozen Test split.
