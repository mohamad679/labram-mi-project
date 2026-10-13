# Final scientific report — frozen NeuroDecode development snapshot

**Status: CLOSED FOR NEW EXPERIMENTS · 2026-10-13**  
**Scope: 104-person PhysioNet EEGMMIDB five-class motor-imagery study; no final Test assessment.**

This is the last record of the study under its owner's decision to discontinue new seeds, additional architectures, hyperparameter sweeps and Test inference. *Closing experimental work is not the same as establishing an externally validated, publishable model.* Numerical findings are preserved, including failures and the unsuccessful translation of Train-only optimization to the original Validation.

![All authentic development-Validation model scores](../results/figures/final_snapshot/benchmark-overview.svg)

## Data contract and chain of evidence

The identical HDF5 (`SHA-256 f6d2c7824f6aaa522286ea8ee5253b0220602117fc69d7b49c3af6e6a1324f7a`) holds 18,096 epochs from 104 subjects: 64 channels × 400 samples, microvolts, 160 Hz, imagery-only 8–32 Hz MOABB preprocessing. The [locked partition](../configs/phase3_locked_split.json) consists of **75 Train (13,050 epochs), 14 development Validation (2,436) and 15 embargoed Test (2,610)** subjects, mutually disjoint.

All recorded studies used five classes in the fixed order `feet`, `hands`, `left_hand`, `rest`, `right_hand`. Macro-F1 averages each class equally. Accuracy is particularly vulnerable to the observed `rest` class prior: 1,176 of 2,436 Validation events (48.28%). The Test EEG was **not used for model selection, training or inference** in the accepted experimental sequence.

![Frozen subject-disjoint study design](../results/figures/final_snapshot/frozen-study-design.svg)

## Final numerical conclusions

| Phase / setting | Population scored | Model | Accuracy | Macro-F1 | Balanced accuracy |
| :-- | :-- | :-- | --: | --: | --: |
| Phase 3 | 14-subject development Validation | Original EEGNet | .4799 | **.4191** | .4418 |
| Phase 3 | Same Validation | Riemannian tangent space | .2767 | .2568 | .2822 |
| Phase 4 | Same Validation | Genuine pretrained LaBraM · partial | .3859 | .2546 | .2618 |
| Phase 5 | Same Validation | Genuine pretrained CBraMod · frozen probe | .2779 | .2027 | .2091 |
| Phase 5 | Same Validation | Genuine pretrained CBraMod · full | .4828 | .1302 | .2000 |
| Phase 5.5a | 75-Train-participant 3-fold **OOF**, not historic Validation | EEGNet raw | .3656 | .3212 | .3430 |
| Phase 5.5a | Same Train OOF | EEGNet epoch-standardized | .4461 | **.3912** | .4124 |
| Phase 5.5a | Same Train OOF | EEGNeX standardized | .4312 | .3697 | .3907 |
| Phase 5.5b | **One-time 14-subject development Validation** | Preselected EEGNet epoch-standardized | **.4885** | .4125 | .4236 |

Sources: [Phase 3](../results/validation/phase3_validation_rerun_seed42_20261013.json), [Phase 4](../results/validation/phase4_validation_seed42_20261013.json), [Phase 5](../results/validation/phase5_cbramod_validation_seed42_20261013.json), [Phase 5.5a](../results/validation/phase55_trainonly_oof_audit_20261013.json), [Phase 5.5b](../results/validation/phase55b_one_time_validation_seed42_20261013.json).

**Observed practical lesson:** Training solely on the original Train partition and selecting between candidate models using grouped Train-only OOF reduced direct reuse of the historical Validation cohort. It selected the normalized EEGNet. That candidate **did not exceed** the earlier EEGNet macro-F1 on historical Validation (.4125 versus .4191). The two historical Validation models were not trained under identical inner-fit protocols, so their small difference is descriptive rather than a controlled ablation. **No additional attempts will be used to chase a higher reported score.**

![Train-only signal normalization and its translation to the historical Validation cohort](../results/figures/final_snapshot/train-only-vs-validation.svg)

## Why classwise reporting changes the interpretation

The final normalized EEGNet checkpoint was selected at **epoch 7** (12 epochs completed, patience five, 67 Train-fit and eight distinct Train early-stop participants), never using historical Validation for epoch choice. The one-time Validation yielded 2,436 valid five-class probability vectors. Its **Macro-F1 .412535**, **accuracy .488506**, **balanced accuracy .423619** and full 5×5 confusion matrix were recomputed from the original user-submitted two-file ZIP (`SHA-256 8e88e4463d1fa96856cb127f57e4a706042ff441095ccb1c1ad5d9109f635391`).

![Validation confusion matrix with actual counts and per-row proportions](../results/figures/final_snapshot/validation-confusion-matrix.svg)

![Final model's true per-class F1 and support](../results/figures/final_snapshot/validation-class-f1.svg)

| Class | True Validation count | Final F1 |
| :-- | --: | --: |
| Feet | 319 | .2551 |
| Both fists (`hands`) | 311 | .3604 |
| Left fist | 324 | .4414 |
| Rest | 1,176 | .6593 |
| Right fist | 306 | .3466 |

The Phase 5 CBraMod Full failure is severe: **all 2,436 epochs were predicted as `rest`**, giving apparent accuracy .4828 but Macro-F1 .1302, exactly the majority-class dummy reference. A numerical accuracy bar alone would conceal this failure. CBraMod's different resampling and zero-padded third patch also introduce a material transfer-domain mismatch.

## Inter-participant uncertainty

![Actual per-subject development Validation macro-F1](../results/figures/final_snapshot/validation-subject-spread.svg)

The selected Phase 5.5b model's per-participant Macro-F1 ranged from **.218** (S058) to **.704** (S015), with aggregate **.4125**. The user-run subject-cluster bootstrap reported a 95% development interval **[.3486, .4847]** based on 2,000 subject-level resamples (not independent held-out confirmation). Repeated use of development data and fixed model seed **42** limit how much this interval can establish.

## Integrity of the scientific snapshot

- Real PhysioNet S3 provenance and data conversion were separately audited. The 104-participant HDF5 was checked against an immutable file hash. The exact executed source/weight hashes and result-archive checksums are in each stage's audit JSON.
- Genuine pretrained LaBraM and CBraMod backbones were verified; no missing checkpoint silently replaced with random weights. The failed CBraMod AMP execution was not misreported as successful.
- Predictions/metrics for the accepted stages were independently audited against fixed class vocabularies, sample counts and participant groups. Some model checkpoint binaries were **not supplied** in results ZIPs, so an independent binary-checkpoint re-execution is **not** claimed.
- Phase 5.5a used 3-fold outer grouped Train-only screening, with independent Train-only inner early-stop subjects. Phase 5.5b froze a single nominee and used the 14-person historical Validation once. The 15 original Test participants were left unused.
- CI success establishes source and integration feasibility, **not** model quality, generalization or patient safety.

## Scientific limitations and responsible use

1. **No final independent Test estimate.** Results should be described as development-only and not as a validated held-out benchmark.
2. **Single recorded training seed.** The user intentionally stopped before multi-seed or independent external validation; do not fabricate robustness from available subject bootstrap intervals.
3. **Different model protocols.** Pretrained CBraMod, LaBraM, Riemannian, original EEGNet and normalized EEGNet used distinct adapters, sample geometry or optimization. Cross-model results are a practical comparison, **not** an identical preprocessing ablation.
4. **Nontrivial class and participant heterogeneity.** Accuracies are confounded by the large Rest prior and subject variability. Classwise scores and subject-level distributions are essential.
5. **The five-class PhysioNet task is run-structured.** Left/right fist and both-fists/feet cues occur in different EDF run categories. This changes the interpretation of cross-task signals and should be disclosed.
6. **No physiological / clinical claim.** Attention rollout is not causal cortical attribution, and post-hoc uncertainty statistics have not been calibrated for safety-critical BCI use.
7. **Limited artifact distribution.** No raw EEG files or model-weight checkpoints are in Git. Kaggle result archives have separate user-controlled storage; the committed JSON is a verified scientific summary of those archives.
8. **Source-license status unresolved.** The repository has no explicitly owner-selected source-code license; upstream dataset and model-checkpoint terms are independent.

## Terminal decision

**No new training, model selection, additional random seeds, additional Test evaluations or alternative architecture campaigns are authorized in this archival snapshot.** Reproducibility helpers, source and audit tests remain in Git because removing them would destroy evidence. This is a **completed development research record, not a model-ready or publication-ready classifier**.

[Back to project front page](../README.md) · [Visual gallery](visualization-gallery.md) · [Archive acceptance and unresolved issues](release-readiness.md)
