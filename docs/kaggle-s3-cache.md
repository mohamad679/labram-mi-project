> **Final-state clarification (2026-10-13):** Earlier text below describes the builder before the first complete user-executed run. The user later completed the **104-subject HDF5** with SHA-256 `f6d2c7824f6aaa522286ea8ee5253b0220602117fc69d7b49c3af6e6a1324f7a` ([provenance record](../results/validation/phase2_full_cohort_kaggle_20261013.json)). Successful private Kaggle Dataset publication was **not** established by the results. No new data-build runs are planned; the instructions below are preserved as a historical recovery guide.

# Kaggle-only PhysioNet cache builder via public Amazon S3

The full PhysioNet EEG Motor Movement/Imagery recordings can be downloaded from the official public mirror, `s3://physionet-open/eegmmidb/1.0.0/`, **without an AWS account or AWS credentials**. This guide offers an alternative when a local Mac or remote Windows machine has slow access to `physionet.org`.

**Standalone user Notebook:** `NeuroDecode_PhysioNet_S3_Kaggle_Cache_Builder.ipynb` is supplied as a downloadable artifact in the user conversation. The repository also includes the core standalone CLI implementation `scripts/prepare_physionet_s3.py`.

## Kaggle workflow

1. Open a private Kaggle Notebook, use CPU/None accelerator, and turn **Internet On**.
2. Install a **Python 3.11** virtual environment with `moabb==1.7.2`, `mne==1.13.2`, `numpy`, `scikit-learn`, `h5py`, `boto3` and `tqdm`. Use the exact scientific preprocessing source commit **`74e511fb9c1ac4f0543b249b571e28e811ec92c3`**, as also used in `run_kaggle.ipynb`.
3. First test downloading one EDF using `--probe`; **Kaggle's actual S3 throughput has not yet been measured**.
4. Process the eligible subjects incrementally into `/kaggle/working/physionet_mi_104subjects.h5`. For each participant, the downloader fetches exactly six default imagery runs (4/6/8/10/12/14), calls the same `src.data.load_physionet_mi`, checks the resulting subject and (64, 400) channel/time shape, then appends to the portable verified HDF5 schema 3. Raw EDFs are deleted **after** an HDF5 subject is committed, reducing peak disk usage.
5. Make a **Kaggle Notebook output snapshot** before ending the session. If interrupted, attach that earlier saved output via **Add Input** and use `--previous /kaggle/input/.../physionet_mi_104subjects.h5`; the script copies the HDF5 into writable working storage and resumes. It does not rewrite previously committed subjects.
6. Once 104 eligible subjects pass `inspect_cache(..., require_complete=True)`, create a **private Kaggle Dataset** from the completed Notebook output and attach it to `run_kaggle.ipynb`. This training notebook must never be run against an incomplete cache.

Run a CLI preparation from the pinned source directory, with the helper script available, e.g.:

```bash
python -m scripts.prepare_physionet_s3 --probe \
  --output /kaggle/working/physionet_mi_104subjects.h5 \
  --mne-root /kaggle/working/mne_data

python -m scripts.prepare_physionet_s3 \
  --output /kaggle/working/physionet_mi_104subjects.h5 \
  --mne-root /kaggle/working/mne_data \
  --max-new-subjects 10
```

For another session, `--previous` accepts the attached partial HDF5 path. If you set `--max-new-subjects 104`, the process attempts all 104, but a Kaggle session can still time out or crash; choosing small batches and saving output snapshots is safer for long-running jobs.

## Limits / evidence

- **Do not confuse a durable HDF5 subject commit with permanently saved Kaggle storage.** `/kaggle/working` is not guaranteed to survive a terminated session unless files persistence or a saved Notebook output snapshot is configured. Kaggle describes how outputs can be reused: https://www.kaggle.com/docs/notebooks
- The script was subjected to offline structural and synthetic S3 file-transfer tests. Successful S3 throughput, real 104-subject MOABB processing and private Kaggle output persistence are **not yet verified** by a complete run.
- The six EDFs are for the default `PhysionetMI(imagined=True, executed=False)` protocol. Do not change the modality or introduce execution runs without changing and revalidating the protocol.
- The public source dataset has **Open Data Commons Attribution 1.0** terms; cite DOI `10.13026/C28G6P`. The HDF5 contains EEG recordings and should **not be committed to GitHub**.
- The generator does not train LaBraM and reports **no model performance metrics**.
