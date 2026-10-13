# Release scope and closure gates

**Status: completed research snapshot, not a validated model release.** The project owner explicitly stopped additional model experiments, random seeds, Test inference and Phase 6 on 2026-10-13. This document therefore records *what is available* rather than soliciting an unbounded future research campaign.

## Verified and versioned

- [x] Frozen 104-person EEGMMIDB provenance, 75/14/15 split and accepted HDF5 digest.
- [x] Genuine pretrained LaBraM / CBraMod checkpoint-loading audits and real public EEG training smoke runs.
- [x] Real Kaggle owner-executed Phase 3–5.5b development results, with independently recomputed Prediction metrics and small SHA-recorded source reports.
- [x] Five-class F1, confusion counts, class-collapse warnings, subject-level exploratory intervals and seven real-data vector figures.
- [x] Fast offline unit-test / static quality GitHub Actions and historical reproducible source references.
- [x] Clear method limitations, data/reuse-rights boundary, legacy/demo separation, final model selection and no-Test access statement.

## Important **not completed** items — do not imply otherwise

- [ ] A final, separately untouched 15-subject Test evaluation or external dataset generalization result.
- [ ] Multiple seed replication and confirmatory effect intervals.
- [ ] Independent checkpoint binary reload with distributable signed weights.
- [ ] Clean-room, fully exact dependency lock / reproducible deployment of the Gradio demo.
- [ ] Owner-approved source-code license and verified permissions to redistribute third-party artifacts.
- [ ] Owner-confirmed contributor/author listing, formal DOI publication or citation metadata.

A finished documentation/portfolio snapshot is compatible with these unresolved scientific gates. It is **not** a validated benchmark-release artifact, medical device or online EEG prediction service. The exact evidence index is [final-research-snapshot.md](final-research-snapshot.md). Further model experiments and open-Test assessment are explicitly **closed**, rather than tacitly scheduled.