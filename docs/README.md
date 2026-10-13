# NeuroDecode documentation · completed development snapshot

**Experimental work is frozen as of 2026-10-13.** The project has an actual 104-person development Validation audit, a Train-only optimized EEGNet, and **no final Test evaluation**. For a short and accurate overview, begin at the [repository front page](../README.md).

![Evidence-linked results across completed EEG experiments](../results/figures/final_snapshot/benchmark-overview.svg)

## Read in this order

| Area | Recommended record |
| :-- | :-- |
| **Scientific conclusion and limits** | [Final research summary](final-research-summary.md) |
| **Charts and older demonstrations** | [Visualization gallery](visualization-gallery.md) |
| **Frozen protocol and remaining release gates** | [Archive readiness](release-readiness.md) |
| Data and original signal provenance | [EEG source](eeg-data-provenance.md) · [Phase 2 processing audit](phase2-data-audit.md) |
| Original Validation baseline | [Phase 3 review](phase3-validation.md) · [real result JSON](../results/validation/phase3_validation_rerun_seed42_20261013.json) |
| Genuine pretrained LaBraM comparison | [Phase 4 research protocol](phase4-finetuning-protocol.md) · [real result JSON](../results/validation/phase4_validation_seed42_20261013.json) |
| Genuine pretrained CBraMod comparison | [Phase 5 source and negative finding](phase5-foundation-comparator.md) |
| Strict Train-only optimization | [Phase 5.5a protocol and result](phase55-train-only-optimization.md) |
| Final one-time Validation | [Phase 5.5b protocol](phase55b-one-time-validation-protocol.md) · [final result JSON](../results/validation/phase55b_one_time_validation_seed42_20261013.json) |
| Reproduce plots without EEG | [Deterministic figure generator](../scripts/render_final_figures.py) |

## Archived implementation (retained for traceability)

The following material belongs to the **historical implementation**, not a new benchmark plan. It can contain older descriptions that predate the validated 2026 stages. The [final research summary](final-research-summary.md) controls when old interpretations conflict.

[Legacy report](legacy-project-report.md) · [Research audit](research-audit.md) · [Architecture and modules](architecture.md) · [Model card](model-card.md) · [Reproducibility requirements](reproducibility.md) · [Historical Kaggle setup](kaggle-experiment.md) · [Historical local cache](local-cache-and-kaggle.md) · [Amazon S3 preparation](kaggle-s3-cache.md) · [Historical benchmark utility](benchmarking.md)

No EEG raw recordings or trained weight binaries are committed. Original Kaggle notebooks and exact executed source revisions are preserved as historical experiment artifacts, not as suggested further runs. Python unit tests, source-integrity audits and deterministic figure regeneration remain valid **non-training** maintenance tasks.

**This is not a clinical, diagnostic, assistive-safety or final Test-validated release.**
