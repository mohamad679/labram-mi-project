# Model card · final NeuroDecode research snapshot

**Status:** Development-only, research discontinued as of 2026-10-13. **Not** a released, production, clinical or Test-validated classifier.

## Task and dataset

Five-class imagined movement from PhysioNet EEG Motor Movement/Imagery using 64 EEG channels and the frozen 2.5 s / 160 Hz preprocessed epochs. The evaluated cohort comprises **104 subjects**: 75 Train, 14 development Validation and **15 Test participants never evaluated**. Signal preprocessing, dataset hash and class vocabulary are recorded in the [final scientific report](final-research-summary.md).

## Models and recorded results

- **Original EEGNet:** development Validation Macro-F1 **0.4191**, Accuracy **0.4799**. Highest recorded Macro-F1 in this development snapshot, but checkpoint/epoch selection used the development Validation protocol of Phase 3.
- **EEGNet + per-epoch channel normalization:** selected by **3-fold group-held-out Train-only** screening (OOF Macro-F1 **0.3912** vs original Train-only raw EEGNet **0.3212**). The subsequent single historical Validation evaluation achieved Macro-F1 **0.4125**, Accuracy **0.4885**. This **did not demonstrate** a gain over the earlier EEGNet Validation Macro-F1.
- **LaBraM pretrained partial:** Macro-F1 **0.2546** on recorded Validation.
- **CBraMod pretrained probe/full:** Macro-F1 **0.2027** / **0.1302** on recorded Validation. Full mode collapsed to predicting **Rest** for every Validation trial.

Official pretrained checkpoint revisions were explicitly audited; the older historical LaBraM model-demo checkpoint from before the scientific audit remains **unverified** and must not be conflated with authenticated Phase 4 weights. No model weights are included in this Git repository.

## Intended and excluded uses

**Intended:** understanding motor-imagery preprocessing and experiment integrity, code review, controlled retrospective research-method comparison, educational illustration of model limitations.

**Excluded:** diagnosis, medical decisions, patient screening, clinical claims, accessibility or safety-critical BCI, calibrated uncertainty, neural causal explanation, real-time production decoding, or citing these numbers as definitive Test performance.

## Essential limitations

- **No Test predictions or independent Test performance estimate**; 15 subjects remain embargoed.
- One recorded model seed and substantial cross-participant heterogeneity, final normalized EEGNet Validation Macro-F1 **.218–.704** by participant and overall **.4125**.
- Strong class imbalance, especially Rest (~48% of Validation); Macro-F1 and classwise confusion are more informative than Accuracy alone.
- Different training and preprocessing adapters across families prevent strict architecture-only comparisons.
- Subject-cluster intervals are exploratory, not independently prevalidated confidence intervals.
- The archived Gradio dashboard and attention/MC-dropout visualizations depend on historical, unshipped checkpoints and lack clinical/causal validation.
- Source license and some upstream redistribution permissions are not resolved.

[Evidence](../results/validation/README.md) · [Final report](final-research-summary.md) · [Figures](visualization-gallery.md) · [Archive status](release-readiness.md)
