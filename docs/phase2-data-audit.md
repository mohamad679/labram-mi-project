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

## Required scientific acceptance gates

- [ ] CI and offline negative-control tests pass on final PR head.
- [ ] Real public EDF annotations match MOABB class mapping for multiple participants.
- [ ] EEG event onset/timing and 481-to-400 cropping verified in seconds.
- [ ] 64 channels, 160 Hz, unit conversion, and MOABB metadata order verified.
- [ ] Temporary HDF5 cache is numerically equivalent to direct preprocessing.
- [ ] A justified, recorded policy exists for exclusions beyond participant 88.
- [ ] Full 104-subject production HDF5 provenance verified OR explicitly
      restricted as unverified until the real production dataset is inspected.

## Limitations and reproducibility

The documented original MOABB dataset notes that subject 88 differs in
sampling frequency. **Other historical exclusions (89, 92, 100, 104) remain
hypotheses**, not independently documented exclusions. An audit of subjects
1 and 2 does not establish results for all subjects or validation of the
production cache. No held-out test labels are used for model selection in
this audit.

Do not claim Phase 2 **PASS** or start model benchmarking solely because
the two-subject diagnostic passes. Keep those scientific limitations explicit.
