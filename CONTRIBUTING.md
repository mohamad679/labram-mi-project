# Contributing

This repository is a research prototype. Changes should improve scientific validity, transparency or the ability to reproduce a run.

## Before contributing

- Keep raw subject-level EEG data, generated checkpoints and secret tokens out of Git.
- Discuss model or dataset license issues before distributing third-party artifacts.
- Never treat historical results as verified pretrained benchmarks without supporting provenance.
- Use a feature branch and submit a pull request; do not rewrite archived reports without leaving a traceable explanation.
- Run `python -m pytest -q` and `python -m scripts.check_repository_quality` before submitting. Tests are offline; full GPU reproducibility is a separate review step.
- For a change to results, specify dataset revision, subject split, random seed, hyperparameters, software versions, metrics and checkpoint source.
- For Gradio updates, test with both valid and malformed example inputs. Label synthetic examples as synthetic.

## Pull request requirements

Describe what changed, why it matters, what commands were run, whether a full GPU test was performed, and any scientific limitations. Avoid adding badges or diagrams that imply unverified performance.

**License pending:** This repository does not yet publish an explicit license for its own source code. Until a license is selected, public visibility does not grant general permission to reuse or redistribute code. Contributors should obtain appropriate authorization before using it.

## Reference guides

See the [documentation hub](docs/README.md) for architecture and evidence levels, the [release gate checklist](docs/release-readiness.md) for publication criteria, and [SECURITY.md](SECURITY.md) for private reporting of sensitive vulnerabilities. Use the issue templates when reporting scientific reproducibility issues.
