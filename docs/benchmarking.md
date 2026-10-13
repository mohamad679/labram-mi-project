# Reproducible result charts and retained legacy benchmark tool

The study is **closed** and only its confirmed **development** predictions/metrics are presented. No new Test evaluation or training is planned.

## Evidence-backed final report

- [Final outcome](final-research-summary.md)
- [Full visual gallery](visualization-gallery.md)
- [Independently reviewed real-run JSON summaries](../results/validation/)
- [Audited model selection and follow-up](phase55-train-only-optimization.md) · [Phase 5.5b protocol](phase55b-one-time-validation-protocol.md)

## Zero-network, no-EEG chart reproduction

The final six figures are deterministic SVG documents, generated **only from committed independently audited JSON results**:

```bash
python -m scripts.render_final_figures --check
python -m scripts.render_final_figures
```

This never downloads EEG, trains, alters original predictions, generates Test metrics or fabricates a missing run. Figures in `results/figures/final_snapshot/` are version-controlled to make the scientific conclusions inspectable directly on GitHub.

## Historical generic benchmark helper (kept, not endorsed for new runs)

The previous `scripts/build_benchmark.py` can generate confusion and reliability plots from a separate compatible real `.npz` file and outputs to local folders. It was part of the older research implementation; **it has not been used to invent or replace any numbers in the frozen 2026 final result charts**. Its documentation/history was retained to preserve provenance. Any reference to running fresh multi-seed, retraining or final Test scoring in old notebooks refers to unexecuted historical plans, not a continuation of this closed study.
