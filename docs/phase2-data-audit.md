# Phase 2 — EEG data and preprocessing contract audit

**Status: Phase 2 PASS for the predeclared 104-subject imagery-only data contract, with explicit and material limitation for the excluded subject 89. Phase 3 requires separate user authorization.**

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
was falsified. The follow-up
[run 38314316708](https://github.com/mohamad679/labram-mi-project/actions/runs/38314316708)
found the **unique missing source event is index 0, onset 0.0 seconds,
T0 (rest)** in *all 12 runs* across the two audit participants.
The complete 29-epoch sequence in each run matches the remaining
29 raw annotations in source order.

This is consistent with the documented default
[`mne.find_events(initial_event=False)`](https://mne.tools/stable/generated/mne.find_events.html),
which does not emit a nonzero initial-stimulus event at sample zero.
The [MNE EDF issue #4760](https://github.com/mne-tools/mne-python/issues/4760)
specifically discusses the first t=0 annotation in `S001R04.edf`.
The contract now **fails closed** if the expected first-T0 omission
changes on either sampled subject. This validates an observed
preprocessing convention; it does not advocate restoring the missing
first rest epoch without a new dataset-versioned protocol.

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

## Full-cohort Kaggle validation — 2026-10-13

The project owner ran the pinned [Phase 2 Kaggle validation notebook](../results/validation/phase2_full_cohort_kaggle_20261013.json).
Its results-only archive was reviewed for internal consistency across the
individual JSON reports, notebook code, full logs, cohort summaries and
the fixed original builder SHA. The archive checksum was
`c57f7923c8037987584dd54c586ca9fddc1c5541bdb70bcbe62c25ba7a92866c`.
The 1.85-GB HDF5 itself was **not uploaded to GitHub**, so this is
an independent review **of the executed Kaggle evidence**, rather than a
second local raw-data rerun.

**Numerical input audit: PASS in the executed Kaggle report.**

- Full schema-3 cache: **104 subjects, 18,096 epochs, 64 channels, 400 samples**
  per epoch; complete 160-Hz microvolt dataset.
- Reported full-cache SHA-256:
  `f6d2c7824f6aaa522286ea8ee5253b0220602117fc69d7b49c3af6e6a1324f7a`,
  matching the earlier Kaggle cache builder report.
- Exactly **18,096 epoch-level reconstructions** from the official PhysioNet
  public S3 EDF source; **463,257,600 float32 signal values** compared with the
  final HDF5 and found equal by the executed audit.
- Aggregate class counts: feet 2,338; hands 2,342; left_hand 2,360;
  rest **8,736 (48.28%)**; right_hand 2,320.
- Repeated **seed-42 75/14/15 train/validation/test subject split** was
  identical and exhaustive: 13,050 / 2,436 / 2,610 epochs, respectively.
  No participant or epoch overlap; no held-out test evaluation was performed.
- 22 targeted offline regression tests passed in the Kaggle environment.

**Direct source EDF exclusion review: 30/30 runs inspected.**

| Participant | Observed EDF sampling rate | Scientific selection rationale |
| --- | ---: | --- |
| 88 | 128 Hz | Incompatible with fixed 160-Hz protocol |
| 92 | 128 Hz | Incompatible with fixed 160-Hz protocol |
| 100 | 128 Hz | Incompatible with fixed 160-Hz protocol; nonstandard annotation structure |
| 104 | 160 Hz | **Run 8 has only 26 annotations (106 seconds)**, whereas its other imagery runs have 30; objective completeness anomaly |
| 89 | 160 Hz | All six EDF files structurally passed the checks. A [peer-reviewed study](https://pmc.ncbi.nlm.nih.gov/articles/PMC11674661/) reports erroneous labels, which the original EDF cannot independently confirm |

A separate [published EEGMMIDB analysis](https://pmc.ncbi.nlm.nih.gov/articles/PMC9414220/)
also reports incomplete annotations for subject 104 among several excluded
participants. Other researchers choose different exclusion sets; this
**predeclared** 104-subject cohort must not be portrayed as universally optimal.
The automated notebook's `objective_flag_found=false` for 104 overlooks
the **26-versus-30 annotations**; the independent report review explicitly
corrects that interpretation without altering the original result archive.

**Supervisor acceptance policy:** Freeze and retain the five historical
exclusions *before* further model tuning; the existing, numerically verified
104-subject cache and deterministic held-out partitions remain unchanged.
Excluding subject 89 is a conservative, **literature-supported but
not independently proven semantic** decision. We do not claim otherwise.
If this assumption is tested in future, it is a **separate cohort-sensitivity
study on validation**, not a reason to modify the frozen test protocol.

**Gate: PASS with declared limitations** for the fixed scientific input
contract. This conclusion does not verify pretrained transfer effectiveness,
EEG decoding accuracy, cross-subject generalization, or the suitability of
this exclusion policy for every other study. Do not begin Phase 3 without
the user's separate approval.

## Required scientific acceptance gates — final disposition

- [x] CI and offline negative-control tests passed on PR #21.
- [x] Real EDF annotations and MOABB event mapping agreed for two subjects
      in GitHub CI and the full retained 104-subject cohort in Kaggle.
- [x] EEG original epoch (481 samples), model crop (400 samples), input
      channel order, 160-Hz target, 8–32 Hz filter and microvolt scaling verified.
- [x] All 104 participants and every retained EEG epoch are present in the
      complete reported production HDF5 and equal the separately reconstructed
      source EDF preprocessing output.
- [x] Actual full-cache deterministic 75/14/15 subject split proven disjoint
      and reproducible.
- [x] All 30 EDFs across five excluded participants independently inspected;
      the frozen exclusion policy documented with objective source anomalies
      for 88/92/100/104 and explicit peer-reviewed, **unverified semantic**
      rationale for subject 89.
- [x] The exclusion of participant 89 is explicitly classified as **a
      documented limitation**, not as independently verified EDF ground truth.

**Scientific decision:** PASS on the established, versioned 104-subject
cohort; retain the original cohort selection, record the unresolved semantic
uncertainty, and keep held-out Test locked.

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
