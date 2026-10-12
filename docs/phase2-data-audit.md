# Phase 2 — EEG data and preprocessing contract audit

**Status: OPEN — do not proceed to Phase 3 until scientific gate approval.**

## Audit scope and immutable scientific definitions

Source: [PhysioNet EEGMMIDB 1.0.0](https://physionet.org/content/eegmmidb/1.0.0/).
MOABB reference: [v1.7.2 dataset implementation](https://github.com/NeuroTechX/moabb/blob/v1.7.2/moabb/datasets/physionet_mi.py)
and [v1.7.2 MotorImagery defaults](https://github.com/NeuroTechX/moabb/blob/v1.7.2/moabb/paradigms/motor_imagery.py).

- Imagery-only six actual EDF runs: **4, 8, 12, 6, 10, 14**.
  MOABB's metadata `run` keys **0..5** represent those EDF runs **in this order**,
  not the actual EDF run numbers.
- Imagined unilateral hand EDF runs 4/8/12: T0=rest, T1=left_hand,
  T2=right_hand.
- Imagined bilateral hands/feet EDF runs 6/10/14: T0=rest,
  T1=hands, T2=feet.
- Executed-movement and eyes-open/eyes-closed baseline runs are excluded.
- Source EEG: 64 ordered channels, 160 Hz; standard MOABB `MotorImagery()`
  preprocessing uses an **8–32 Hz bandpass** by default.
- MOABB default epoch: 0..3.0 seconds, 481 inclusive samples at 160 Hz.
  The project retains the **first 400 samples (2.5 seconds)** to match
  LaBraM's 200-sample patch size. Last retained sample: 2.49375 seconds.
  This drops the remaining ~0.5 seconds rather than asserting the original
  task is only 2.5 seconds long.
- Data units: MOABB ndarray convention in microvolts; MNE Epochs return volts,
  so explicit `dataset.unit_factor=1e6` preserves numeric parity.
- Portable HDF5 schema 3 stores `float32` epochs and class/participant IDs.
  The Phase 2 diagnostic audits an **ephemeral two-subject cache**, NOT the
  complete 104-subject snapshot used by the project.

## Executable contract checks

`src/eeg_protocol.py` provides strictly checked source-run mapping,
per-run class constraints, original epoch times and cropped sample count.
`scripts/audit_phase2_data.py` compares labels from the real EDF original
`T0/T1/T2` annotations against the MOABB per-run output ordering, verifies
MOABB metadata and signal units, and checks exact float32 parity between
the MOABB arrays and temporary HDF5 readback. The report contains only
aggregate counts, no raw EEG or individual EEG samples.

`tests/test_phase2_protocol.py` adds negative controls for executed runs,
swapped hand/feet meanings, bad run keys, missing task classes, bad windows,
and metadata mixing participants.

## First real-data evidence (PR #21)

[Successful two-subject public EEG audit run](https://github.com/mohamad679/labram-mi-project/actions/runs/38313722098)
recorded 12/12 run-specific label checks, with original EDF annotations
30/run and retained 29 MOABB epochs/run for **each of subjects 1 and 2**.
Both subjects yielded **174 usable epochs**, distributed as:

| Subject | Rest | Left hand | Right hand | Both hands | Feet |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 84 | 23 | 22 | 21 | 24 |
| 2 | 84 | 23 | 22 | 24 | 21 |

Rest forms **84/174 ≈ 48.3%** of examples for both sampled
subjects; raw overall accuracy can reward a majority-rest classifier,
so subsequent model comparisons must emphasize balanced accuracy and
macro-F1. These are empirical class counts, **not model metrics**.

All **348 epochs** in the temporary partial HDF5 cache reproduced
the exact float32 tensor values, labels and subject identities from
the fresh MOABB preprocessing pipeline.

**Follow-up diagnostic:** A deliberately stronger assumption that any
missing raw annotations must occur **only at the end** of each run
failed on real subject 1/run 4. The code therefore does not assume
this behavior. The audited alignment instead enumerates all source
annotation indices that could have been omitted while preserving
the MOABB label sequence, and includes candidate onset times and
recording duration. This is an explicit investigation item, not proof
of a mislabeled epoch.

## Exclusion provenance from external research

The project excludes 88, 89, 92, 100 and 104. These exclusions
have **published precedents**, but the current dataset was not
independently re-audited for each participant:

- [MOABB v1.7.2 source](https://github.com/NeuroTechX/moabb/blob/v1.7.2/moabb/datasets/physionet_mi.py)
  explicitly warns of **128-Hz subject 88**.
- [The Effect of Processing Techniques on the Classification Accuracy of
  Brain-Computer Interface Systems (2024)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11674661/)
  reports exclusions 88, 92 and 100 for sampling/data-structure
  mismatches and 89 for reportedly incorrect labels.
- [Motor Imagery Analysis from Extensive EEG Data Representations
  Using Convolutional Neural Networks (2022)](https://pmc.ncbi.nlm.nih.gov/articles/PMC9414220/)
  documents incomplete annotations for 88, 92, 100 and 104.
- Other studies use different exclusion sets. These findings justify
  investigating the selected participants, **not assuming every
  participant excluded by another paper must be excluded here**.

The split/cohort definition remains unchanged during Phase 2.

## Required scientific acceptance gates

- [ ] CI and offline negative-control tests pass on final PR head.
- [x] Real public EDF annotations agree in ordered class sequence for subjects 1 and 2;
      exact reasons for omitted source annotations still under investigation.
- [ ] EEG event onset/timing and 481-to-400 cropping verified in seconds.
- [ ] 64 channels, 160 Hz, unit conversion, and MOABB metadata order verified.
- [ ] Temporary HDF5 cache is numerically equivalent to direct preprocessing.
- [x] External literature supporting the historical exclusions documented;
      **independent per-subject source-data validation remains pending**.
- [ ] Full 104-subject production HDF5 provenance verified OR explicitly
      restricted as unverified until the real production dataset is inspected.

## Limitations and reproducibility

The documented original MOABB dataset notes that subject 88 differs in
sampling frequency. **Other historical exclusions (89, 92, 100, 104) remain
hypotheses**, not independently documented exclusions. An audit of subjects
1 and 2 does not establish results for all subjects or validation of the
production cache. Peer-reviewed exclusion reports vary in selection and
therefore cannot substitute for direct source-data checks. No held-out test labels are used for model selection in
this audit.

Do not claim Phase 2 **PASS** or start model benchmarking solely because
the two-subject diagnostic passes. Keep those scientific limitations explicit.
