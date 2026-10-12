# Model card — research prototype

**Model:** NeuroDecode-LaBraM Motor Imagery Baseline

**Task:** Offline five-class EEG motor imagery classification.

**Architecture:** `InterpolatedLaBraM` via Braindecode, 64 input channels, 160 Hz, 400 time points, five class logits.

**Training dataset:** PhysioNet EEG Motor Movement/Imagery via MOABB. Historical reports describe a subject-disjoint 75/14/15 split.

**Initialization provenance:** Historical checkpoint provenance is **unverified**; default refactored code explicitly uses scratch initialization. Loading pretrained weights requires an explicit model source. No officially verified weights are redistributed here.

**Historical evaluation:** Representative archived head-only accuracy 0.357, macro-F1 0.193. Values have not been reproduced after this refactor and must not be presented as a validated pretrained foundation-model benchmark.

**Intended use:** Research, code review, educational demonstrations and nonclinical benchmarking.

**Out-of-scope uses:** Medical diagnosis, patient triage, clinical decisions, safety-critical assistive technology, or claims about a subject's neurological state.

**Known limitations:** Low macro-F1, variable class performance, unverified weights and preprocessing provenance, limited compute/seed coverage, and exploratory uncertainty and attention-based interpretation.

**Transparency:** Model cards, calibration checks, complete benchmark artifacts, dependency lock and independent validations are planned but not all complete. See [research audit](research-audit.md) and [reproducibility guide](reproducibility.md).
