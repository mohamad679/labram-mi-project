# Model card — final development research snapshot

**Identity:** NeuroDecode, five-way Motor Imagery EEG research comparison. **Status:** scientific development snapshot frozen 2026-10-13; not a licensed, downloadable, production model release.

## Purpose and intended setting

Explore subject-disjoint EEG motor-imagery classification on PhysioNet EEGMMIDB (104 eligible participants, 18,096 epochs; 64 channels × 400 samples; 160 Hz; five labels). Valid use is reproducibility analysis, instruction and methodological comparison. **Not intended** for diagnosis, neurocognitive profiling, patient stratification, safety-critical BCI or clinical decisions.

## Models actually evaluated

- EEGNet trained from scratch (original Phase 3 development baseline): Macro-F1 **.4191** on the frozen 14-person development Validation.
- EEGNet from scratch with per-epoch/channel z-score (Train OOF-selected in Phase 5.5a, evaluated once in Phase 5.5b): Macro-F1 **.4125** on the original same-cohort Validation; no demonstrated improvement on the historical baseline.
- Genuine pretrained LaBraM (frozen probe / last-two-block partial / full adaptation), Phase 4; best partial validation Macro-F1 **.2546**.
- Genuine pretrained CBraMod (frozen / full), Phase 5; frozen probe Macro-F1 **.2027**; full adapter **collapsed completely to `rest`**, Macro-F1 **.1302**.

These are development-only, mostly single-seed and heterogeneous-training-protocol figures, **not heldout Test metrics**. Training checkpoints were excluded from the shareable ZIPs; weight-level independent reload is incomplete, although Kaggle-run reload checks and official pretraining-integrity reports were recorded.

## Dataset, limitations and risks

The five classes are `feet`, `hands`, `left_hand`, `rest`, `right_hand`. `rest` occupies ~48% of original Validation and can misleadingly inflate naive accuracy. Per-class F1 and Macro-F1 are therefore primary; group-disjoint source participants prevent the most obvious subject identity leakage. Data preprocessing (8–32 Hz MOABB filter, microvolt scale and 2.5-s crop) and the models' differing adapter expectations limit universal conclusions. End-user/device domain transfer has not been established. Even on the internal Validation cohort the standardized model has pronounced participant heterogeneity and a subject-bootstrap 95% Macro-F1 interval of **[.3486,.4847]**.

**The original 15-subject Test was deliberately not evaluated** during the versioned study. No multiple-model-seed replication or separate population study occurred. Earlier archived demo/MC Dropout/attention reports include historical results of unverified checkpoint provenance, segregated from the new empirical evidence.

## Reproducibility and legal status

Model/subject evidence: [Final research snapshot](final-research-snapshot.md) · [Development visuals](visualization-gallery.md) · [Input hash and split](phase2-data-audit.md) · [Official source checks](phase5-foundation-comparator.md). The repository contains code and small reproducibility manifests, **not** data files, subject-level raw EEG, signed model weights or a commercially/clinically usable distribution. **No owner-selected source code license has been provided.** Upstream PhysioNet recordings and LaBraM/CBraMod pretrained weights remain subject to their original licenses and citations.