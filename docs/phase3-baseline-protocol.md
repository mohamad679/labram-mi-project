# Phase 3 — Locked-split validation benchmark (execution OPEN)

**Scientific status:** PARTIAL PASS on a real 104-subject validation-only run; acceptance blocked by Riemannian nonconvergence and missing selected-model checkpoints. **Do not claim Phase 3 PASS or proceed to Phase 4** before reviewing executed, provenance-complete Kaggle results.

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

## First real-data run: 2026-10-13 Kaggle validation review

The project owner provided a complete results-only ZIP from one real-data run,
with SHA-256 `882e0b944fd937986b5715dd3cbe88b9bb55691388d08d0eeaa009f17c889767`.
The exact embedded notebook-runner source matches SHA-256
`e55f0d3c0a141c7162685654bc44923633518bfcd35796152bc3fb7a65251b09`.
Its environment used Python 3.13.15, torch 2.11.0+cu128, Braindecode 1.5.1,
MOABB 1.7.2, MNE 1.13.2, pyRiemann **0.12** (not the initially requested
0.10, which was dependency-incompatible), and scikit-learn 1.6.1.

All four runs completed with the frozen original 104-subject HDF5 hash.
Validation-only predictions were provided for exactly **2,436 epochs and
14 participants**, with identical true-label/subject order across candidates.
Recomputing the complete five-class confusion matrix, accuracy, balanced
accuracy and fixed-five-class macro-F1 independently from each NPZ exactly
matched the Kaggle JSON report. No locked Test subject IDs or Test
predictions occur in the exported prediction files; the original runner's
source code also only computes Train/Validation predictions.

| Method | Accuracy | Validation macro-F1 | Balanced acc. | 95% validation subject-bootstrap CI for macro-F1 |
| --- | ---: | ---: | ---: | ---: |
| Dummy prior | .4828 | .1302 | .2000 | [.1302, .1302] |
| Riemannian OAS + tangent LR | .2898 | .2675 | .2944 | [.2235, .3032] |
| EEGNet | .4770 | **.4063** | **.4276** | [.3422, .4593] |
| Pinned LaBraM pretrained linear probe | .3818 | .1861 | .2016 | [.1676, .1997] |

**Interpretation:** EEGNet has substantially better balanced and macro-F1
validation performance than the frozen-backbone LaBraM probe in this *single
seed* comparison. Dummy achieves slightly higher raw accuracy by always
predicting the majority `rest` class, with zero recall for the other
four classes. This does not prove EEGNet's final test generalization or
that all valid ways to fine-tune LaBraM underperform.

Exploratory paired participant bootstrap (1,500 repetitions, 14 validation
subjects) gives EEGNet–LaBraM macro-F1 difference **+.2202** with percentile
interval **[+.1699,+.2673]**, and EEGNet–Riemannian **+.1388** with interval
**[+.0873,+.2025]**. Because the *same validation group selected the best
network epoch*, these are **optimistically selected development-set
intervals, NOT independent confirmatory confidence intervals**.

### Scientific deficiencies detected before acceptance

1. **Riemannian convergence failure:** sklearn emitted `ConvergenceWarning`
   with L-BFGS reaching `max_iter=400`; the reported Riemannian value is
   useful as a preliminary reference but **not accepted as a numerically
   converged baseline**.
2. **Selected model states not saved:** EEGNet's selected best epoch is 12,
   LaBraM's is 1; both reports state `best_state_dict_saved=false`. Although
   exported predictions are self-consistent, the exact selected learned
   models cannot be reloaded and checked from those archives.
3. **Source review/reproducibility:** the executed runner is embedded in
   the user notebook, with hash verified, but has not yet been committed
   as reviewed executable GitHub project code. A notebook-only code SHA
   is provenance evidence, **not a substitute for an archived source file**.
4. **Single seed / selection bias:** no multi-seed generalization evidence
   is available; model changes must not be selected using Test.

**Supervisor verdict: PARTIAL PASS / BLOCKED from final scientific Phase-3
acceptance.** A revised Kaggle notebook has been supplied separately to
require L-BFGS convergence after train-only scaling, save checkpoint
weights and SHA-256, reload/check the 2,436 selected Validation
predictions, and preserve the original Test embargo. That new runner
SHA-256 is `9100671e93ccf3d21c74d123273cbe96e78ea30ebc5791b8e47e8f15e0ddf210`.
Its syntax and four isolated offline regression checks pass, but **there
is no completed real EEG rerun yet**. Any revised hyperparameter/scaling
protocol is a new versioned candidate; historical preliminary figures
must not be silently overwritten.

## Current execution status and reproducibility

- Self-contained user-executable Kaggle notebook: **NeuroDecode_Phase3_Baselines_Kaggle.ipynb**, supplied in this conversation. It pins the audited main commit `b3af6247d027cd1a3dd33915368bad03f9bf220c`, verifies the HDF5 file hash, materializes the validation-only runner and fixed split, and runs controlled benchmarks. Because this code is not yet merged into `main`, the notebook explicitly fingerprints its full runner source SHA-256 in the output archive.
- This draft PR freezes the split and scientific protocol. **The full experimental runner is currently in the self-contained notebook artifact, not in the repository branch yet.** Its computational correctness must be validated by a full Kaggle run before promoting it to accepted project code.
- Do not commit metrics inferred from historic test data or synthetic smoke tests.

## Supervisor gate: conditions for PASS

- [x] Kaggle logs show completed real-cache SHA validation and exact 75/14/15 participant IDs.
- [x] Dummy, Riemannian, EEGNet and pretrained LaBraM all produce valid **validation-only** probabilities on the same 2,436 epochs (Riemannian still needs converged rerun).
- [ ] No signal transformations or learned covariance/tangent statistics are fitted on validation/test.
- [ ] Execution artifacts and versioned source fingerprints allow code/results reconciliation; ambiguous failures must be resolved and rerun.
- [x] Per-class scores, confusion matrices, subject uncertainty and a reasoned **exploratory** scientific comparison are reviewed.
- [ ] Full executable source and tests are incorporated into a reviewed GitHub PR; CI and any real-data smoke checks pass.
- [x] Test remains locked; Phase 4 requires the user's approval after Phase 3 gate PASS.

**No model-quality improvement is claimed until the above conditions hold.**
