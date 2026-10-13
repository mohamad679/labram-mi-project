# Benchmarking and published figures — frozen evidence

**Publication status: final development-only research snapshot. No new model/seed training or original Test inference is authorized.** The complete list of real, source-traced plots is in the [visualization gallery](visualization-gallery.md).

## The 2026 completed studies

| Recorded cohort | Number of epochs | What can be compared |
|---|---:|---|
| **75 original Train participants**, three outer group folds | **13,050 OOF per method** | Four originally preselected candidates on Train; see [Phase 5.5a](../results/validation/phase55_trainonly_oof_audit_20261013.json). **Not comparable to Validation as an absolute benchmark.** |
| **14 original development Validation participants** | **2,436 per tested method** | Single-seed EEGNet/LaBraM/CBraMod snapshots and one nominated final EEGNet; see the [final study](final-research-snapshot.md). |
| **15 original held-out Test participants** | **2,610 reserved epochs** | **No scores, figures or claims; intentionally untouched**. |

## Which figures are directly supported?

![Development Validation model comparison](../assets/figures/validation_model_comparison.svg)

The vector SVGs under `assets/figures/` were generated only from the final user ZIP's actual `report.json` and `validation_predictions.npz` metrics or previously independently verified, versioned phase JSON records. The final ZIP SHA-256 is `8e88e4463d1fa96856cb127f57e4a706042ff441095ccb1c1ad5d9109f635391`; its two entries passed ZIP integrity checks. See [full final evidence metadata](../results/validation/phase55b_one_time_validation_seed42_20261013.json).

The figures include: (1) honest six-protocol development-Validation Macro-F1, (2) separate four-method Train-only OOF comparison, (3) actual row-normalized five-class confusion matrix, (4) exact per-class F1 and support, (5) participant-level performance distribution and (6) observed inner Train-only checkpoint macro-F1 curve. A separate vector cover image is decorative and **contains no measurement claims**.

## Rechecking actual metrics

For a separately supplied **authorized** predictions NPZ, `scripts/build_benchmark.py` computes standard confusion and reliability figures and `src/subject_statistics.py` computes exploratory participant-bootstrap intervals. The script is a historically retained research utility; its existence does **not** imply any benchmark on untouched Test participants. It should not be invoked to generate new scores for this closed study.

The CSV/JSON reports in `results/validation/` preserve fixed dataset and runner digests, subject partitions, exact official foundation checkpoint revisions and scientific limitations. Model binaries and the raw HDF5 are **not public repository artifacts**. Exact checkpoint-weight reload is therefore attested by user-executed Kaggle logs rather than independently reproduced from the shared ZIP.

**Do not mix:** old legacy `results/phase1_baseline.md` historical Test references with the validated 2026 development-Validation series. Neither those old unverified numbers nor the later OOF candidate-selection scores establish a fresh final-Test or clinically useful model.

[← Research conclusion](final-research-snapshot.md) · [All scientific figures](visualization-gallery.md) · [Model card](model-card.md)
