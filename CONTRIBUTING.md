# Repository maintenance policy — frozen research snapshot

The owner closed new model training, random-seed campaigns, Validation-based architecture search, Test evaluation and Phase 6 after the completed 2026-10-13 evidence audit. This is a research *record*, not an invitation to tune reported performance figures.

Corrective contributions should be limited to obvious source/provenance defects, documentation and accessibility improvements, or security fixes that **do not change the frozen experimental outcomes**. Before proposing a maintenance change:

- Preserve frozen subject partitions, existing checksums, runner commits, and provenance of all reported scores. Never rewrite history to make a disappointing model appear successful.
- Never add raw EEG, HDF5, patient signal samples, model checkpoint binaries or API credentials to this public repository.
- Distinguish verified Kaggle-run metric recomputation from unsupplied checkpoint weight-level re-execution and from a true independent Test study.
- Use `python -m pytest -q` and `python -m scripts.check_repository_quality` for offline maintenance checks. They are not model-quality benchmarks.
- Do not submit a new model/seed Test result to the final dashboard: it would require an explicitly new study and cannot be silently pooled with this closed research snapshot.

**Licensing:** There is no owner-selected license for original source code; GitHub visibility alone does not authorize redistribution. Upstream pretrained weights, software and PhysioNet EEG recordings retain their own terms. See [README.md](README.md), [SECURITY.md](SECURITY.md), and [the scientific closure record](docs/final-research-snapshot.md).