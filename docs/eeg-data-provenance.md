# EEG channel, preprocessing and cache provenance

## Verified source definitions

- Dataset: [PhysioNet EEGMMIDB 1.0.0](https://physionet.org/content/eegmmidb/1.0.0/), 64 EEG channels at 160 Hz, with an ODC-By 1.0 data license.
- MOABB `PhysionetMI(imagined=True, executed=False)` describes the five labels `feet`, `hands`, `left_hand`, `rest`, `right_hand`.
- [MOABB paradigms](https://moabb.neurotechx.com/docs/generated/moabb.paradigms.base.BaseProcessing.html) can return `mne.Epochs` with channel names via `return_epochs=True`.
- [MNE EEGBCI](https://mne.tools/stable/generated/mne.datasets.eegbci.standardize.html) notes legacy EDF capitalization and trailing-period channel names.

## What the new loader verifies

`src.data.load_physionet_mi` requests MNE Epochs and checks **ordered EEG channel names** against `src.channels.PHYSIONET_64_CH_NAMES`, then checks **160-Hz sampling** and the alignment of epochs, labels and metadata.

The loader converts epochs with `epochs.get_data()` and crops the sample count down to a multiple of the LaBraM patch size (200). It saves the channel names, sampling rate and MNE EEG amplitude unit (`V`) in new local caches.

**Scientific migration warning:** The original version requested MOABB's default NumPy array output, whereas the new loader requests MNE Epochs to preserve montage metadata. Depending on MOABB preprocessing and output pipelines, array unit scaling and other transformations may differ. **A paired numerical comparison of the old and new outputs on the same subject must be performed before using this change to reproduce historical benchmarks.** Do not compare old scores to new ones without this control.

## Cache policy

New caches have `X`, `y`, `subject_ids`, `channel_names`, `sfreq`, and `input_unit`. Arrays without these provenance fields are rejected rather than assumed trustworthy.

An existing legacy cache may be used **only for non-publishable exploratory experiments**, by setting `ALLOW_UNVERIFIED_EEG_CACHE=1`. This bypass does **not** verify channel order or units; no scientific conclusions should rely on it.

Datasets, local EEG caches and checkpoints are gitignored. Never commit private or subject-level arrays into the repository.

## Known gaps

- No external GPU/data rerun or historical cache numerical-equivalence test has been performed for this migration.
- Exact MOABB/MNE preprocessing settings, bandpass and amplitude scaling must be frozen after a clean run.
- Exclusion of participants 88, 89, 92, 100, and 104 is inherited from the historical project and needs an independent data-quality re-audit.
- Spatial interpolation from 64 sensors to the LaBraM canonical 128 channels depends on sensor geometry and model/library revision. It is not evidence of medically meaningful channel importance.
