# Download PhysioNet once on your own computer, use it for every Kaggle run

Kaggle can run out of RAM during bulk EEG preparation. The repository supports **one portable HDF5 file** containing all 104 eligible subjects with validated 64-channel, 160-Hz, 400-sample, microvolt EEG epochs. It downloads **one subject at a time**, writes completed subjects durably, and continues after interruptions.

## 1. Create the HDF5 cache locally

Install **Python 3.11**, Git, and sufficient free disk space on your computer. The local downloader does **not** need PyTorch, CUDA, a GPU or the full training requirements. From a terminal:

```bash
git clone https://github.com/mohamad679/labram-mi-project.git
cd labram-mi-project
python -m venv .venv
```

Activate on macOS/Linux with `source .venv/bin/activate` or on Windows PowerShell with `.venv\Scripts\Activate.ps1`. Install:

```bash
python -m pip install "numpy>=1.26,<3" "scikit-learn>=1.3,<2" "moabb==1.7.2" "mne==1.13.2" "h5py>=3.10,<4"
```

Now run:

```bash
python -m scripts.download_physionet_local --output physionet_mi_104subjects.h5
```

The script prints subject-level progress, per-subject duration, disk usage and a rough ETA. If the Internet connection drops, run **the same command again**. Completed subjects are skipped, and unfinished appends are repaired before resuming.

After all subjects are available, you should see `COMPLETE VERIFIED CACHE` with 104 subjects and a SHA-256 checksum. The companion `physionet_mi_104subjects.sha256.json` records validation information.

**Keep the complete H5 and checksum file locally as long-term backups.** Do not commit them to GitHub. Actual size and total download time depend on your computer and connection; ensure there is plenty of free space.

## 2. Upload into your own Kaggle account

In Kaggle, create a **private Dataset**, upload `physionet_mi_104subjects.h5` and attach it to the training Notebook via **Add Input**. The Notebook searches for the H5 file under `/kaggle/input`; you can also provide the exact path in `ATTACHED_CACHE_PATH`, e.g.:

```python
ATTACHED_CACHE_PATH = "/kaggle/input/my-private-physionet-cache/physionet_mi_104subjects.h5"
```

Use the updated project Notebook `run_kaggle.ipynb` from main. It validates the HDF5 cohort, and the model reads training batches **lazily from HDF5**, avoiding materializing the entire EEG array in Kaggle RAM. The same file can be used in all six model/seed experiments without re-downloading data.

This is not the original raw PhysioNet EDF archive; it is an **analysis-ready derived cache** with the scientifically checked preprocessing and metadata. Keep the Dataset private and respect PhysioNet terms.

## 3. Training progress

The training pipeline prints real-time **Train / Validation / Test batch progress bars**, percentages, elapsed time and approximate ETA. Epoch summaries include Train/Validation Accuracy and Macro-F1. Kaggle captures progress in `train.log`.

The full GPU benchmark is not yet completed merely by preparing the dataset. Preserve `split_seed=42`, model seeds and original source commit for a fair comparison. Run the selected model/seed, collect `results_only_*.zip`, and audit the outputs before publishing performance claims.
