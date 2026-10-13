# Phase 3 — Locked-split validation benchmark (execution OPEN)

**Scientific status:** The reproducible **single-seed validation-only baseline gate is PASS WITH LIMITATIONS** after a successful Kaggle rerun and code review. This does not validate held-out test performance or multi-seed model superiority. **Do not claim Phase 3 PASS or proceed to Phase 4** before reviewing executed, provenance-complete Kaggle results.

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

**Historical supervisor verdict on the FIRST run: PARTIAL PASS / BLOCKED.** A revised Kaggle notebook has been supplied separately to
require L-BFGS convergence after train-only scaling, save checkpoint
weights and SHA-256, reload/check the 2,436 selected Validation
predictions, and preserve the original Test embargo. That new runner
SHA-256 is `9100671e93ccf3d21c74d123273cbe96e78ea30ebc5791b8e47e8f15e0ddf210`.
Its syntax and four isolated offline regression checks pass, but **there
is no completed real EEG rerun yet**. Any revised hyperparameter/scaling
protocol is a new versioned candidate; historical preliminary figures
must not be silently overwritten.

## Corrected full-cohort real-data rerun — 2026-10-13

A **second Kaggle execution** used the same frozen cohort/SHA and the same
75/14/15 subject partition; all four models produced validation-only
predictions on the same **2,436 epochs / 14 subjects**. The submitted ZIP
SHA-256 is
`cff52a9cc28148343f70772523de849d083955bf5a459f3c2fab346e79442d8f`.
The exact executed runner has SHA-256
`9100671e93ccf3d21c74d123273cbe96e78ea30ebc5791b8e47e8f15e0ddf210`,
identical to [scripts/run_phase3_validation.py](../scripts/run_phase3_validation.py).
The complete [Kaggle runner notebook](../notebooks/NeuroDecode_Phase3_Validation_Rerun_Kaggle.ipynb)
and [offline negative controls](../tests/test_phase3_validation.py) are archived.
See [machine-readable independent audit](../results/validation/phase3_validation_rerun_seed42_20261013.json).

| Validation model | Accuracy | Macro-F1 | Balanced accuracy | 95% validation participant-bootstrap Macro-F1 |
| --- | ---: | ---: | ---: | ---: |
| Dummy, train-only prior | .4828 | .1302 | .2000 | [.1302,.1302] |
| **Riemannian**, converged OAS/TangentSpace/StandardScaler/LR | .2767 | .2568 | .2822 | [.2126,.2888] |
| **EEGNet**, epoch 24 | .4799 | **.4191** | **.4418** | [.3529,.4729] |
| **LaBraM pretrained linear probe**, epoch 1 | .3818 | .1861 | .2016 | [.1676,.1997] |

**All reported metrics and 5×5 confusion matrices were independently
recomputed from the four safe-to-share validation-only NPZ files.**
Every file uses the same ordered labels/participant IDs, normalized finite
probabilities, and exactly the published 14 validation participants. The
submitted ZIP contains no locked-Test predictions. The runner's code also
uses Test participants only to assert the fixed subject-level split.

**Resolved deficiencies from the first run:**

- Riemannian fits OAS covariances, Riemannian tangent reference and
  StandardScaler on the designated **training** epochs only. L-BFGS
  converged at iteration **317/3000** without the earlier warning.
  This standardization is a *protocol change*, so its score is not an
  identical-setting retry of the first run.
- EEGNet selected epoch **24**, saved 19,755-byte model state SHA-256
  `921a35a20c722ab34e75ba7f1a5b30a6a186368a6b5cae7006e60951faff3f77`.
  LaBraM selected epoch **1**, saved 23,336,634-byte model state SHA-256
  `eaa084dd6063eb934f985ea325bcfc6f0c8bb166c4a893401d270bb44d04255d`.
  Kaggle-run reports explicitly confirm both selected states were
  **reloaded and produced numerically matching validation predictions**.
  The weight files are kept in the Kaggle working output tree; **they
  were intentionally excluded from the submitted results-only ZIP**.
  Thus the reviewer did *not* independently reload those binary files.
- Full executed runner source and its notebook are now archived in this
  GitHub pull request, with the exact byte hash validated against the
  submitted Kaggle evidence.

**Exploratory paired participant bootstrap:** With 2,000 cluster samples
from 14 development/validation participants, EEGNet − LaBraM probe
Macro-F1 difference was **+.2330** with 95% percentile interval
**[+.1805, +.2860]**; EEGNet − converged Riemannian was **+.1622**
with interval **[+.1093, +.2241]**. These numbers reuse the very same
validation group chosen for network early stopping, so they are
**exploratory and subject to model-selection bias**, not confirmatory
evidence of superiority on unseen held-out Test participants.

**Scientific acceptance:** Phase 3's defined goal — generate a transparent,
numerically verifiable, single-seed **validation-only** baseline comparison
on the fixed Phase-2 EEG cohort — is met. The scientific conclusion is
limited to **EEGNet > LaBraM frozen-backbone linear probe on this development
validation run**. This does **not** cover LaBraM full fine-tuning, repeated
random seeds, test generalization, or publication-standard final inference.
Those claims require future phases and must not be inferred from this gate.

## Current execution status and reproducibility

- Both owner-executed Kaggle runs and the **corrected, accepted validation rerun** are archived by reproducible per-file source SHA-256 and the submitted results-only ZIP checksum. The exact executable runner and source-only notebook are now tracked in this PR.
- The original benchmark used pinned Phase-2 code commit `b3af6247d027cd1a3dd33915368bad03f9bf220c` plus the separately archived runner. To reproduce the *accepted* result in a new Kaggle session, use [the corrected notebook](../notebooks/NeuroDecode_Phase3_Validation_Rerun_Kaggle.ipynb) and attach the same immutable Phase-2 HDF5 file.
- The corrected model checkpoint binaries were **not in the submitted ZIP**. Preserve a private Kaggle output version if exact binary checkpoint reloading needs independent re-execution. All reported metrics are still verifiable without weights from the NPZ files.
- The single-seed evaluation and resulting validation-only intervals **must not** be advertised as held-out Test performance or replicated evidence.

## Supervisor gate: final evidence disposition

- [x] Kaggle logs and code verified the complete HDF5 SHA and 75/14/15 frozen subject split.
- [x] All four methods have valid, independently recomputed validation probabilities on exactly the same 2,436 epochs.
- [x] Covariance/tangent/scaling fits and learned model updates use training epochs only; validation is reserved for evaluation/early stopping.
- [x] Previous Riemannian convergence and deep checkpoint persistence deficiencies resolved in the corrected real Kaggle rerun.
- [x] Exact runner source + notebook + split lock + regression tests archived in GitHub; repository PR CI passed.
- [x] Confusion matrices, per-class scores, subject-cluster uncertainty and transparently limited scientific comparison reviewed.
- [x] Fifteen-subject held-out Test partition remains locked; **no Phase 4 work begins without separate authorization**.

**Supervisor scientific gate: PASS FOR SINGLE-SEED, VALIDATION-ONLY BASELINE
EXECUTION WITH DECLARED LIMITATIONS.** Independent test and multi-seed claims
remain blocked until their respective later scientific-validation stages.
