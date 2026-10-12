---
name: Scientific reproducibility
about: Report a mismatch or propose an evidence-backed validation
title: "[Reproducibility] "
labels: []
assignees: []
---

## Scientific claim or experiment

State the exact claim being assessed and whether it is historical, verified, or unverified.

## Protocol

- Git commit:
- Dataset/version and permissions:
- EEG channels, sampling rate, signal units, preprocessing:
- Subject-level train/validation/test split:
- Seed(s) and selected checkpoint:
- Pretrained Hub repository, immutable revision and SHA256:
- Device, Python and pinned package versions:

## Metrics and evidence

Provide aggregate class-wise metrics, number of participants, and traceable (non-private) artifact hashes or workflow links. Never publish raw data or a selected "best" test result without full run accounting.

## Known limitations

Explain leakage risks, statistical assumptions and which part has **not** been independently repeated.

See the [research release gates](../../docs/release-readiness.md).
