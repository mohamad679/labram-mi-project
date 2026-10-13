# Audited scientific evidence registry

**Frozen 13 October 2026.** These JSON files are evidence summaries from real, user-executed research sessions and their independent checks. Not all checkpoint binaries or raw result archives are stored in Git; each audited report records what was and was **not** directly re-executed. This repository has **no final Test evaluation**.

| Stage | Machine-readable record | Status / evidence scope |
| :-- | :-- | :-- |
| Data provenance | [Full PhysioNet HDF5 audit](phase2_full_cohort_kaggle_20261013.json) | 104 subjects / 18,096 EEG epochs, frozen SHA-256 |
| Phase 3 | [Original EEGNet, Riemannian, Dummy and frozen LaBraM probe](phase3_validation_rerun_seed42_20261013.json) | Genuine 14-subject development Validation |
| Phase 4 | [Genuine pretrained LaBraM](phase4_validation_seed42_20261013.json) | Actual Validation probe / partial / full |
| Phase 5 | [Genuine pretrained CBraMod](phase5_cbramod_validation_seed42_20261013.json) | Probe and full; full-mode **Rest class collapse** |
| Phase 5.5a | [Train-only 3-fold cross-validation audit](phase55_trainonly_oof_audit_20261013.json) | Exactly 75 Train subjects; nominated normalized EEGNet |
| Phase 5.5b | [One-time historic Validation audit](phase55b_one_time_validation_seed42_20261013.json) | 14 subjects, normalized EEGNet .4125 Macro-F1 |

Other JSON reports in this directory preserve earlier provenance, CPU gradient smoke checks and scientific guard tests. These are **not interchangeable with benchmark scores**. The [final narrative](../../docs/final-research-summary.md) and [source-backed figures](../../docs/visualization-gallery.md) link directly to the records above.
