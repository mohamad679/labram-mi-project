# Changelog

All notable research evidence and repository-publication decisions are recorded here. Dates are UTC research-workflow dates. **This is not a conventional software version history or a claim of peer review.**

## 2026-10-13 — Completed research & presentation freeze

- Finished and independently reviewed the **one-time original development-Validation assessment** of the normalized EEGNet nominated in Train-only three-fold CV: **accuracy .4885, Macro-F1 .4125, balanced accuracy .4236**, 14 Validation subjects, 2,436 rows; **no Test predictions**.
- Explicitly recorded the **non-improvement** compared with historical Phase-3 raw EEGNet on the original 14-subject Validation (Macro-F1 **.4191**). This is a descriptive, not fully controlled ablation comparison.
- Recorded Phase 5.5a four-candidate **75-Train-subject OOF** results; standardized EEGNet improved on raw EEGNet inside the matched screening protocol (**.3912 versus .3212** Macro-F1), but that did not establish a final Validation improvement.
- Recorded genuine official-pretrained **LaBraM and CBraMod** weight audits; described total CBraMod full-fine-tuning **rest-class collapse**.
- Ended further model/seed/Phase6 and heldout-Test work per repository owner instruction.
- Added seven source-derived vector graphics, revamped README and docs, clarified clinical/scientific limits, archived stale documentation and removed superseded launchers and oversized obsolete demonstration media from HEAD without altering history.
- **Not published:** independently untouched 15-person Test metrics, replicated-seed estimates, external validations, model checkpoint binaries, software license or peer-reviewed release.

## Previous 2026-10-13 — Audited research phases

- Phase 1: real EEG preprocessing and official pretrained-weight integration feasibility gates.
- Phase 2: frozen 104-person PhysioNet EEGMMIDB HDF5 SHA-256 and 75/14/15 subject split.
- Phase 3: Dummy, Riemannian, EEGNet and LaBraM frozen probe on the original development Validation.
- Phase 4: pretrained LaBraM head-only, partial and full adaptation.
- Phase 5: pretrained CBraMod frozen probe and full fine-tuning with scientifically negative outcome.
- Phase 5.5a: four-model subject-disjoint Train-only cross-validation.
- Phase 5.5b: final single nominated normalized EEGNet assessment on historical development Validation.

Authoritative citations and experiment-level SHA-256 evidence: [Final scientific snapshot](docs/final-research-snapshot.md) and [Machine-readable phase records](results/validation/).

## Before 2026-10-13 — Historical prototype

The older LaBraM/MC Dropout/attention/Gradio descriptions and legacy metrics are preserved as historical records with unverified older checkpoint provenance. They are not included in the new empirically audited series.
