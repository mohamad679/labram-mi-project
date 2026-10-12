# Scientific training strategy correction (2026-10-13)

The previous training entrypoint called a head-only freezing function for **both**
pretrained and scratch initialization. Consequently, `model_init: scratch`
meant frozen random backbone features plus a trained classifier; this is **not**
training LaBraM from scratch.

The training entrypoint now uses an explicit `training_strategy`:

| Initialization | Strategy | Backbone | Classification head |
| --- | --- | --- | --- |
| scratch | full_finetune | randomly initialized, trainable | trainable |
| pretrained | linear_probe | checkpoint initialized, frozen | trainable |
| pretrained | full_finetune | checkpoint initialized, trainable | trainable |

The scratch+linear_probe combination is rejected at configuration validation.
Parameter counts are printed and recorded in the experiment manifest.

**Scope of evidence:** Regression tests check parameter trainability with a
small PyTorch model. They do **not** establish a real LaBraM GPU run, successful
optimization, the identity of externally fetched pretrained weights, or a
valid improvement in EEG classification.

The historic scratch results cannot be relabeled as genuine scratch training.
The historical held-out test metrics remain exploratory. No new test-set
selection or comparison should occur before a predeclared validation protocol.

Remaining gates: real LaBraM head-structure integration check, actual gradient
flow, EDF/event-to-label audit, artifact uniqueness, and independent confirmatory
evaluation.
