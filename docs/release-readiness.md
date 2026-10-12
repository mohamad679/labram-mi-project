# Release readiness and evidence gates

**Status: research prototype, not a versioned validated release.** This file is a transparent acceptance checklist. Checkboxes are not claims of completion.

## Repository hygiene

- [x] Basic offline unit tests and GitHub Actions workflow exist.
- [x] Structured research README, model card and reproducibility notes exist.
- [x] Archived results remain traceable and explicitly distinguished from verified runs.
- [ ] A source-code license has been selected and reviewed by the owner.
- [ ] Full installation instructions have passed on a fresh machine or container using pinned dependencies.
- [ ] The name, repository description, website/topics and maintainer-facing metadata have been finalized.
- [ ] An appropriate citation record (with owner-confirmed authorship) has been published.

## Scientific reproducibility

- [x] A pinned LaBraM checkpoint and CPU transfer/forward check were demonstrated.
- [x] EEG scaling equivalence was confirmed for one real public PhysioNet subject.
- [ ] Exact environment lock, provenance hashes, preprocessing manifest and all run parameters are archived.
- [ ] Train/validation/test subjects remain unchanged across compared model initialization seeds.
- [ ] Full eligible-dataset pretrained and scratch experiments have completed with multiple seeds.
- [ ] Real per-class predictions, confusion matrices, calibration plots and subject-level intervals are published.
- [ ] Claimed statistical differences have been evaluated fairly, with no test-set model selection.
- [ ] Independent held-out replication or external dataset validation has been performed where feasible.

## Model distribution and safety

- [ ] Model weights have a verifiable release link, file checksum and provenance metadata.
- [ ] Distribution respects all applicable upstream model and dataset licenses.
- [ ] The dashboard was smoke-tested on fresh installation using a distributable compatible checkpoint.
- [ ] Interpretability/uncertainty claims are limited to methods actually validated.
- [ ] No clinical, diagnostic or safety-critical use is implied.

## Publication rule

A release candidate must have all critical scientific and environment gates reviewed, not merely green Python unit tests. Code quality, functional integration and scientifically demonstrated effectiveness are separate evidence categories. Version tags and badges should reflect the actual achieved category.

For ongoing work, use [the repository issues](https://github.com/mohamad679/labram-mi-project/issues) and [the reproducibility guide](reproducibility.md).
