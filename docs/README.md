# Evidence-first documentation

**Research status:** Closed development snapshot, 2026-10-13. **No further model or seed training; the original 15-subject Test cohort was not evaluated.** Results and limitations are recorded, not automatically extended into future benchmark claims.

[← Read the visual project overview](../README.md) · [Open all results figures](visualization-gallery.md)

## The essential three documents

| Guide | What it establishes |
|---|---|
| [Final scientific snapshot](final-research-snapshot.md) | All audited outcomes, negative findings, evidence limitations and final research freeze |
| [Visualization gallery](visualization-gallery.md) | Seven real-data SVG figures, each traced to versioned experimental records |
| [Model card](model-card.md) | Which models were trained, data boundaries, out-of-scope medical use |

## Reconstruct the finished sequence

| Completed stage | Method/evidence |
|---|---|
| Phase 1 | [Real EEG and official pretrained adapter acceptance](phase1-scientific-acceptance.md) |
| Phase 2 | [Full 104-subject PhysioNet/MOABB data provenance](phase2-data-audit.md) |
| Phase 3 | [Development-Validation baselines and frozen split](phase3-baseline-protocol.md) · [machine record](../results/validation/phase3_validation_rerun_seed42_20261013.json) |
| Phase 4 | [Pretrained LaBraM probe/partial/full](phase4-finetuning-protocol.md) · [machine record](../results/validation/phase4_validation_seed42_20261013.json) |
| Phase 5 | [Real pretrained CBraMod and class-collapse finding](phase5-foundation-comparator.md) · [machine record](../results/validation/phase5_cbramod_validation_seed42_20261013.json) |
| Phase 5.5a | [75-subject Train-only candidate screen](phase55-train-only-optimization.md) · [machine record](../results/validation/phase55_trainonly_oof_audit_20261013.json) |
| Phase 5.5b | [One-time final development Validation](phase55b-one-time-validation-protocol.md) · [machine record](../results/validation/phase55b_one_time_validation_seed42_20261013.json) |

## Research implementation reference

- [Data integrity, EEG units and channels](eeg-data-provenance.md)
- [Model/data processing architecture](architecture.md)
- [Static reproducibility / environment guidance](reproducibility.md)
- [Repository release boundaries and ownership decisions](release-readiness.md)
- [Scientific legacy and unverified archived claims](research-audit.md) · [Original archived report](legacy-project-report.md)
- [Original public Amazon S3 HDF5 reconstruction](kaggle-s3-cache.md)
- [Security and data privacy](../SECURITY.md)

The version-controlled notebooks in [`notebooks/`](../notebooks/) are retained for **historical run traceability**, not invitations to tune the Validation set or open the Test cohort. Source and checkpoint hashes are recorded by the corresponding phase evidence. See the [research snapshot](final-research-snapshot.md) for what was and was not independently reproduced.

## Legal and scientific scope

The repository has **no owner-selected source license**, no independent Test study, and no independently downloadable final model weights. It is not a medical or accessibility-grade classifier. Public visibility is not a grant of reuse rights. Existing upstream software, models and PhysioNet data have their own terms and attribution requirements.