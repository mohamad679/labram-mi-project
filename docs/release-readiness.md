# Archival research snapshot: completed work and unresolved gates

**State: frozen development record (2026-10-13).** This is a consciously closed research project, **not** a claim that all model-validation or commercial-release gates passed. The owner has stopped new experiments and does not plan further seeds, architecture searches or a Test evaluation in this snapshot.

## Evidence-backed accomplishments

- [x] Exactly 104 included PhysioNet participants with immutable 18,096-epoch HDF5 SHA-256 and verified preprocessing.
- [x] Immutable subject split: 75 Train, 14 development Validation, 15 original Test **unused**.
- [x] Authentic pretrained LaBraM/CBraMod checkpoints audited against exact revisions and tensor digests.
- [x] Real participant-disjoint development results and five-class confusion counts independently reconciled for Phases 3–5.
- [x] Phase 5.5a compared four predetermined models using Train-only 3-fold subject-grouped OOF; standardization nominated objectively.
- [x] Phase 5.5b ran the nominated EEGNet with 67 Train-fit + 8 Train early-stop and exactly one 14-subject development Validation prediction pass.
- [x] All five class recalls and participant heterogeneity documented, including failure of CBraMod Full to predict any class except Rest.
- [x] Six chart SVGs generated from the committed actual metric/evidence JSON; no synthetic benchmarking results.
- [x] Basic CI, Python compilation, non-training unit tests and internal-document link audits preserved.
- [x] No private EEG waveform files or model checkpoint binaries committed to the public Git repository.

## Deliberately unresolved — do not silently label as passed

- [ ] **A final held-out Test score**: Test remained untouched by explicit decision.
- [ ] Multi-seed robustness, prospective external-site reproducibility or independent dataset generalization.
- [ ] Fully independently reconstructed complete training checkpoint binaries for every candidate.
- [ ] Fully pinned dependency lock with fresh machine end-to-end reproduction of every Kaggle run.
- [ ] A source-code **LICENSE explicitly selected by the owner**, plus review of upstream dataset/model distribution rights.
- [ ] Author-approved citation metadata and any journal publication or model-card release.
- [ ] Medical/assistive device safety, calibrated uncertainty and validated neurophysiological attributions.

Passing CI is **not** evidence of predictive performance. The final development Macro-F1 is **.4125 for the nominated normalized EEGNet**, compared descriptively to **.4191** on the earlier original EEGNet run; a model-quality improvement on the original Validation was **not shown**. See [the complete report](final-research-summary.md).

## Repository preservation policy

Keep source code, immutable split/config manifests, Kaggle provenance notebooks, audited JSON reports, key demos and tests so the archived conclusions can be checked. Remove **only** redundant empty markers and clearly superfluous files that do not contribute provenance or working functionality. Preserve the Git history rather than rewriting experiments to imply better findings.

The GitHub repository may be made **read-only** using *Settings → General → Danger Zone → Archive this repository* after this documentation/figure cleanup has been merged. Archiving is a GitHub administrative action, not part of the scientific acceptance gate. This repository currently has **no owner-selected code license**; public access alone does not grant downstream redistribution rights.
