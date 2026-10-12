# Kaggle full-data experimental execution

> [Download or open the complete Kaggle Notebook](../run_kaggle.ipynb).
> This is a runnable study launcher, **not** a claim that the full GPU
> experiment has already been reproduced or validated.

## Kaggle Python 3.13 / Torchaudio compatibility

Kaggle's Notebook kernel may run Python **3.13**, whereas the pinned `torchaudio==2.5.1` CUDA 12.1 wheel is not available for that Python version. The notebook automatically bootstraps **CPython 3.11** using [uv](https://docs.astral.sh/uv/) into `/kaggle/working/neurodecode_python311` and runs PyTorch, Braindecode, MOABB, MNE, training, benchmark plots and environment freezing with the isolated interpreter. It intentionally leaves the Kaggle notebook's Python 3.13 kernel untouched.

Do **not** manually retry `pip install torch==2.5.1 torchaudio==2.5.1` in the notebook kernel; that is the incompatible environment. Instead, import the updated Notebook and run all cells from the beginning, with Internet enabled to download managed Python and the correct `cp311` CUDA wheels. A failed installation does not mean the model or EEG data were invalid.

The isolated interpreter prints and verifies its Python version and GPU availability before any dataset download. Full end-to-end Kaggle training still needs a successful real GPU run for validation.

## Before starting

1. In Kaggle Code, import `run_kaggle.ipynb`.
2. Set **Accelerator → GPU** and **Internet → On** in Notebook Settings.
3. Keep the Notebook private while it contains cached EEG data. Read the source-data terms before distributing any dataset copy.
4. Use **Run All**. The initial public PhysioNet data download may be sizable.
5. After completion, choose **Save Version**, then download `results_only_pretrained_seed_42.zip`.

The Notebook clones a **fixed GitHub commit**, pins a CPU-proven PyTorch/Braindecode/MOABB stack with the CUDA 12.1 PyTorch build, validates the GPU and source, verifies the eligible 104-subject cohort and its channel/units cache, and performs subject-disjoint learning and evaluation. It then generates classification metrics, subject-cluster bootstrap intervals, confusion matrix, reliability diagram and true training/validation curves.

## Study design

A fresh session runs **one full-data experiment**, not all six. On the initial run, the Notebook chooses `pretrained` and `seed=42`. To complete the comparison, repeat for all of these configurations while keeping the code commit and `SPLIT_SEED=42` unchanged:

| Model | Seeds |
| --- | --- |
| Pinned pretrained LaBraM (400-sample temporal adaptation) | 42, 123, 2025 |
| Same LaBraM architecture, random initialization | 42, 123, 2025 |

`BATCH_SIZE=16` is the initial setting; on GPU out-of-memory try 8 or 4 and record the change. Each experiment produces a separate run directory, and the optional `RUN_LABEL` supports controlled retries without overwriting unfinished checkpoints.

## Dataset reuse

The default cache path is `/kaggle/working/physionet_mi_cache.npz`. The cache must include the project’s validated channel and scaling metadata (schema v2). To reuse a cache between Kaggle sessions, attach it as a private Kaggle Dataset and set `ATTACHED_CACHE_PATH` to its exact `/kaggle/input/...` path.

Do **not** upload EEG caches or raw recordings to the project Git repository. Kaggle output visibility matters as well: avoid making notebooks or output datasets public when they include recordings or user-generated subject-level data.

## Files produced

Per model/seed, expect a trained checkpoint and metadata, a held-out predictions file, an environment snapshot, training log, figures, tabular metrics, participant-level confidence intervals and a small shareable `results_only_*.zip`. The ZIP excludes the full checkpoint, raw EEG cache and held-out per-subject prediction array; share the ZIP for the first independent audit.

## Validation boundaries

- Static notebook syntax and content controls run in GitHub CI.
- Previous **CPU** experiments validated pretrained weight transfer and a real EEG one-step optimization test.
- A full Kaggle GPU training run has **not** been executed by the project upgrade process. A Kaggle image might require environment troubleshooting, and session limits can interrupt a long run.
- The baseline training script cannot yet resume a partially finished optimizer checkpoint. Start a new run-specific directory for failed experiments.
- Scores are not suitable for clinical or operational use. Model superiority must be established only after a complete fair multi-seed benchmark.

See [benchmarking](benchmarking.md), [data provenance](eeg-data-provenance.md), and [research integrity](research-audit.md) for scientific caveats.
