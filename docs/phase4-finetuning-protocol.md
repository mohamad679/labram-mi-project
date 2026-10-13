# Phase 4 — Pretrained LaBraM partial/full fine-tuning scientific gate

**Status: PASS WITH DECLARED LIMITATIONS for the strictly defined single-seed development Validation comparison.** This does **not** constitute independent Test, multi-seed, or publication-level validation. The 15 frozen Test subjects remain embargoed.

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

## Completed Kaggle full-cohort execution and independent review — 2026-10-13

The owner returned a real Tesla T4 Kaggle execution ZIP, SHA-256
`21e83455d6151bca61604c597f16e288a132285a4aa2f4acdfcbda91fc30886e`.
The notebook **checked out and SHA-verified** the pinned actual runner:
commit `15468a4392faa8019ed3f3ac5746314756851514`,
source SHA-256 `c1dadc4dc5bc0a3b92ff20d973be1b271861cf0f123cf7959be55c8fc408623d`.
The executed code checked the original 104-subject Phase-2 HDF5 SHA-256,
identical 75/14/15 split, and the immutable pretrained revision.

**Independent result review:** Each of the three exported `validation_predictions.npz`
files was reopened and its **2,436** five-class probabilities checked for
finite nonnegative normalized values. All had exactly the same 14 Validation
participant IDs, same sample ordering, same true labels and no Test subject
IDs. Accuracy, five-class fixed-label macro-F1, balanced accuracy and all
five-class confusion matrices were independently recomputed and matched the
reported values *exactly*. Machine-readable aggregate evidence is in
[results/validation/phase4_validation_seed42_20261013.json](../results/validation/phase4_validation_seed42_20261013.json).

| Pretrained LaBraM strategy | Accuracy | Validation macro-F1 | Balanced accuracy | Best epoch | Trainable parameters |
| --- | ---: | ---: | ---: | ---: | ---: |
| Head-only probe | .2414 | .1863 | .2014 | 1 | 1,005 |
| **Partial**, last 2/12 blocks | **.3859** | **.2546** | **.2618** | 7 | 966,365 |
| Full | .3485 | .2481 | .2615 | 6 | 5,818,341 |

**Training and weights:** The head-only / partial / full regimes trained
for 6 / 12 / 11 epochs respectively, stopping under the versioned
patience-5 Validation policy. Kaggle reports selected checkpoint SHA-256
values and successful reload with maximum Validation probability difference
**0.0** for every method. The hash of *every frozen trainable parameter*
is unchanged in probe and partial; the same initial first-block pretrained
SHA-256 is recorded across all three strategies. **Selected checkpoint binary
files are not in the shared results ZIP**; only the executing Kaggle
environment verified their reload. Full has no frozen parameters, so its
`frozen_parameters_unchanged=true` output is vacuous and must not be
interpreted as evidence of frozen weights.

**Exploratory paired 14-participant cluster bootstrap:** Independently
recalculated with seed 42 and 2,000 participant-resampled replicates:

| Macro-F1 difference | Point estimate | 95% development-Validation percentile interval |
| --- | ---: | ---: |
| Partial − Probe | +.0683 | [+.0510,+.0894] |
| Full − Probe | +.0618 | [+.0377,+.0901] |
| Partial − Full | +.0065 | [−.0168,+.0306] |
| Phase-3 EEGNet − Phase-4 Partial | +.1645 | [+.1074,+.2179] |

The Phase-3 EEGNet comparison uses the **same ordered 14 participant
Validation rows** but separate model optimization settings; it is a
development reference, not a controlled architecture-only ablation.
The earlier EEGNet remains stronger on this Validation split (**macro-F1
.4191**) than either LaBraM fine-tuning regime. The confidence intervals
reuse the Validation set for **best-epoch model selection**, hence are
**exploratory and optimistically selected**, NOT independent confirmatory
subject generalization intervals. Partial and Full differ by too little
to establish a reliable ranking: their paired interval crosses zero.

**Decision:** Accept Phase 4's precise hypothesis-test *implementation*
and **single-seed Validation-only execution** with transparent limitations.
The evidence supports that trainable LaBraM blocks improve this experiment
over head-only probing, but **does not show competitive decoding superiority
over EEGNet**, multi-seed robustness, or Test generalization. Test remains
locked and subsequent stages require separate user authorization.

## Scientific supervisor criteria — final disposition

- [x] Independent repository Ruff / pytest checks passed on the PR head.
- [x] Genuine PhysioNet EEG gradient-flow CI confirmed probe-only, last-2-block,
      and full-backbone optimization with nonzero intended gradients and weights.
- [x] Kaggle checked full HDF5 SHA, frozen 75/14/15 participant split and
      disallowed Test signal inference; all reported predictions are Validation-only.
- [x] All three pretrained modes completed, with finite, same-order **2,436**
      Validation probabilities; selected checkpoints were saved and reloaded
      in Kaggle and probability differences were exactly zero.
- [x] Frozen parameter SHA-256 hashes were equal in probe and partial;
      the same initial first-transformer-block hash was reported across all modes.
- [x] Every five-class metric/confusion matrix independently recomputed from
      supplied NPZs; exploratory paired participant bootstrap documented.
- [x] Exact runner commit/SHA, pinned dataset/pretraining evidence,
      notebook, report and limitations recorded in version-controlled GitHub.
- [x] No held-out Test evaluation or new Phase 5 work.

**Scientific Gate: PASS — only for this single-seed development Validation
fine-tuning comparison, with the declared limitations.** Stronger conclusions
need multi-seed and untouched-Test evaluation in later authorized stages.
