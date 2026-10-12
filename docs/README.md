# Documentation hub

**NeuroDecode-LaBraM** is a research-engineering prototype for five-class motor-imagery EEG, with a LaBraM-based classifier, exploratory uncertainty estimates and transformer attention inspection. This documentation distinguishes code quality, real weight/data integration checks, and scientifically validated model performance.

## Start here

**Kaggle freezing during download?** Use the [single-file resumable local EEG downloader](local-eeg-cache.md), then attach the finished HDF5 file as a private Kaggle Dataset. The [Kaggle notebook](../run_kaggle.ipynb) now requires this file instead of bulk-download in Kaggle.


**Want to run the full dataset on Kaggle?** Start with [the GPU Notebook and setup instructions](kaggle-experiment.md), then [open the Notebook](../run_kaggle.ipynb). The original model checkpoint and test accuracy are not claimed as independently verified.


| Goal | Document |
| --- | --- |
| Understand the architecture and boundaries | [System architecture](architecture.md) |
| Inspect existing visual outputs without overstating claims | [Visualization gallery](visualization-gallery.md) |
| Install and reproduce the pipeline | [Reproducibility guide](reproducibility.md) |
| Understand EEG channel order and voltage conversion | [EEG data provenance](eeg-data-provenance.md) |
| Check pretrained adaptation evidence | [Phase 3 experimental gates](phase3-validation.md) |
| Generate metrics and plots from real predictions | [Benchmarking](benchmarking.md) |
| Identify known scientific weaknesses | [Research integrity audit](research-audit.md) |
| Read model limitations and intended use | [Model card](model-card.md) |
| See release acceptance requirements | [Release readiness](release-readiness.md) |
| Inspect original, unverified historical results | [Archived project report](legacy-project-report.md) |

## Evidence labels

**Verified offline:** deterministic unit tests, syntactic compilation, lint, and local Markdown links; these do not prove real-data model validity.

**Verified integration:** the [pinned CPU pretrained weight audit](https://github.com/mohamad679/labram-mi-project/actions/runs/38234001055) exercised the 400-sample adaptation, while the [public PhysioNet preprocessing audit](https://github.com/mohamad679/labram-mi-project/actions/runs/38234916355) compared two MOABB output paths for one subject.

**Not yet scientifically verified:** a full independent pretrained-versus-scratch benchmark over the eligible dataset, across multiple random model seeds and a fixed held-out subject split. No claimed clinical readiness.

## Useful commands

```bash
python -m pytest -q
python -m scripts.check_repository_quality
python -m src.train --config configs/baseline.yaml
```

Running full training may download public EEG and require substantial GPU resources. The default baseline initializes from scratch. Refer to the reproducibility guide for checkpoint and dataset assumptions. The current repository name is `labram-mi-project`; the documentation uses the proposed project identity **NeuroDecode-LaBraM**.

## Responsible sharing

The public repository does not currently contain an explicitly selected license for its own source code. The licenses of PhysioNet recordings, Braindecode, external pretrained weights, and this project's original implementation must be assessed independently before redistributing artifacts. Avoid uploading raw EEG recordings or subject-level dataset caches to Git.
