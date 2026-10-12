# Download PhysioNet once on your computer; reuse one HDF5 file on Kaggle

This solves two separate bottlenecks: (1) downloading every public EEG recording again after a Kaggle reset, and (2) loading the full 104-subject EEG matrix into Kaggle RAM, which can crash the session. The portable file stores EEG epochs on disk and is read lazily, one sample at a time.

> Scientific status: source code and synthetic resume tests passed GitHub CI. The full 104-subject laptop download, private Kaggle upload, and multi-epoch GPU experiment have **not** yet been observed to complete end-to-end. Actual data download time and file size depend on hardware and compression.

## 1. Requirements on your own computer (no GPU needed)

- Working internet connection and enough **free disk** for the original EDF download cache **and** a second, potentially multi-gigabyte, portable HDF5 file.
- Git and **Python 3.11**. Do not use a Python 3.13 environment for the verified modeling environment; the local dataset builder is independent of CUDA, but Python 3.11 avoids known compatibility problems.
- A stable persistent location to save the final file. Windows PowerShell, macOS Terminal, and Linux shells can run the same Python module.

Download this repository and freeze it at the source code revision used in the paired Kaggle Notebook:

```bash
git clone https://github.com/mohamad679/labram-mi-project.git
cd labram-mi-project
git checkout 74e511fb9c1ac4f0543b249b571e28e811ec92c3
```

Create a Python 3.11 virtual environment. On macOS/Linux:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
```

Then on all systems:

```bash
python -m pip install --upgrade pip
python -m pip install "moabb==1.7.2" "mne==1.13.2" "numpy>=1.26,<3" "scikit-learn>=1.3,<2" "h5py>=3.10,<5" "tqdm>=4.66,<5" "PyYAML>=6,<7"
```

## 2. Download and prepare **one file**, subject by subject

```bash
python -m scripts.prepare_portable_eeg --output physionet_mi_104subjects.h5 --sha256
```

This script:
- Uses MOABB's **upstream PhysioNet downloader**, bypassing NEMAR sourcedata retries.
- Calls the same validated EEG preprocessing used by the model, **one eligible subject at a time**.
- Checks 64 EEG channels, 160 Hz, 400 time points, microvolt units, class labels and finite values.
- Writes a **single** compressed HDF5 file and updates the durable completed-subject checkpoint only after each subject has been saved.
- Displays a local subject-count progress bar.
- If network, program or computer fails, run **the exact same command with the same output path again**. The existing completed subjects are skipped; any interrupted append is truncated to the last committed boundary.

The downloader may also keep raw EDF files in a separate local folder (`~/mne_data` by default). **Only the completed HDF5 file needs to be uploaded to Kaggle.** You can delete the original EDF cache after verifying the HDF5 if you no longer need it for future preprocessing work.

### Recovery from transient PhysioNet network timeouts

PhysioNet downloads can stall with `ReadTimeoutError` / `requests.exceptions.ConnectionError` for an individual EDF, for example `S007R14.edf`. This does **not** mean the durable HDF5 cache is lost. The script prints `RESUMING: N/104 subjects`, preserves committed subjects, automatically retries the failed subject (eight retries by default) and waits 15, 30, 60, then at most 120 seconds between attempts.

To select retry limits:

```bash
python -m scripts.prepare_portable_eeg \\
  --output ~/Downloads/physionet_mi_104subjects.h5 \\
  --max-retries 12 --retry-delay 20 --sha256
```

If all retries fail, running **the same command again** continues from completed subjects. The specific EDF interrupted mid-transfer may need a fresh transfer, while already cached complete EDF recordings are reused. Never delete the `~/mne_data` directory or the HDF5 file during recovery. An HTTPS `InsecureRequestWarning` is a separate TLS-security warning, not a download timeout; do not suppress certificate verification warnings as a timeout workaround.

Verify the final file:

```bash
python -m scripts.prepare_portable_eeg --output physionet_mi_104subjects.h5 --verify-only --sha256
```

Success is reported only after all **104** eligible subjects and the exact expected five classes are present. The `--sha256` value can be copied into the Kaggle Notebook's `EXPECTED_CACHE_SHA256` field for a full-file integrity check. Never attach a partially completed cache to a benchmark.

## 3. Upload it to a private Kaggle Dataset

In your Kaggle account:

1. Navigate to **Datasets → New Dataset** (or use Kaggle's upload interface).
2. Upload the **one** `physionet_mi_104subjects.h5` file. The upload can be large; keep both the source file and local raw-download cache until success is confirmed.
3. Choose **Private**, verify that data distribution and reuse comply with PhysioNet's dataset terms, and finish the upload.
4. Open your Kaggle Notebook and select **Add Input**, then attach your private dataset.
5. In the Notebook settings cell, set:
   `ATTACHED_CACHE_PATH = "/kaggle/input/YOUR-DATASET-SLUG/physionet_mi_104subjects.h5"`
   (copy the actual path shown in Kaggle; the example is not a literal path).
6. If you ran `--sha256`, optionally paste the verified hash into `EXPECTED_CACHE_SHA256`; otherwise leave it empty.

Your dataset is available as a **read-only** `/kaggle/input/...` file in every Kaggle session to which you attach it. It is **not copied into RAM**.

Kaggle's official guide documents private Datasets and attaching them to Notebooks:
- https://www.kaggle.com/docs/datasets
- https://www.kaggle.com/docs/notebooks

For a large upload where the browser cannot finish, Kaggle also has an official `kaggle datasets create -p <folder>` CLI. It uses a `dataset-metadata.json` in the upload directory; do **not** use `--public` when uploading this research cache. Official CLI documentation:
https://github.com/Kaggle/kaggle-cli/blob/main/docs/datasets.md

## 4. Run pretrained and scratch experiments without re-downloading EEG

Import [run_kaggle.ipynb](../run_kaggle.ipynb), enable **GPU** and **Internet** (still needed for pinned model weights and setup), attach the private Dataset, then Run All.

The Notebook verifies the HDF5 schema and cohort, uses the lazy HDF5 training pipeline, and never executes the full PhysioNet downloader. Every training and validation epoch prints a progress bar with **completed/total batches, percentage, elapsed seconds and ETA** directly in the Kaggle output, while writing `train.log`.

The updated experiment code is pinned at
`74e511fb9c1ac4f0543b249b571e28e811ec92c3`.

**Do not combine training results from different source commits without an explicit reproducibility audit.** Use the same HDF5 file, `split_seed=42`, fixed preprocessing and controlled optimizer settings for all six comparisons (Pretrained vs Scratch for seeds 42, 123 and 2025). Changing the batch size after out-of-memory errors changes the optimization protocol and should be recorded.

## 5. Information to send after the first successful run

Download and share only the **results-only ZIP** from `/kaggle/working/neurodecode_outputs`, such as `results_only_pretrained_seed_42.zip`. It contains aggregate metrics and logs; it intentionally excludes raw EEG, the entire HDF5, weight files and per-subject predictions. The HDF5 remains yours for later reproducible runs.

If a session disconnects **during training**, the original script still cannot resume an interrupted optimizer step/epoch. A finished source-cache download is reusable; this does **not** imply the training optimizer has checkpoint-resume capability. If a GPU run stops, share `train.log` and the full last traceback.

## 6. Known limitations

- No empirical guarantee of Kaggle upload acceptance for an arbitrary large HDF5; inspect the upload limit and disk availability before uploading.
- The full file contains public subject recordings transformed into epochs. Follow PhysioNet's sharing/license requirements and avoid committing or publicly exposing the file.
- A complete HDF5 file does not prove model accuracy or absence of other operational issues.
- Sample-level random access can be slower than keeping arrays in RAM, but it substantially lowers peak host memory demands.
