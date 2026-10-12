# Phase 1 — Scientific Training Correctness: Acceptance Evidence

**Scope:** Verify model initialization/training semantics, real-EEG optimizer
operation, small-subset memorization, and checkpoint integrity. This is NOT
a benchmark or evidence of motor-imagery generalization.

## Source and immutable evidence

- Base main revision: `23902c11c3bf9c51ae990a544037bb03ebfc1c95`
- Real-EEG acceptance source revision: `c10d02b9c66f7394cccd2e84d09d03c35967e79c`
- [Successful real-EEG CPU workflow](https://github.com/mohamad679/labram-mi-project/actions/runs/38312699453)
- Saved workflow artifact: `phase1-real-eeg-acceptance` (checkpoint, SHA-256
  manifest, real-data diagnostic report, and S3 staging report; 7-day retention)
- Official pretrained reference:
  `braindecode/labram-pretrained@0563b6c626e7b40d9a36653b763715db94d945d7`
- [Prior independent source-tensor identity audit](https://github.com/mohamad679/labram-mi-project/actions/runs/38234001055)
  confirmed the 400-sample temporal prefix and backbone-tensor transfer.

## Verified training strategies

- `scratch + full_finetune`: all **5,818,341 / 5,818,341** model parameters
  trainable. A real two-epoch EEG training step produced finite gradients
  across **221** non-head gradient tensors and changed the backbone
  `cls_token` parameter after SGD. This validates a real optimizer path,
  NOT convergence of scratch training.
- `pretrained + linear_probe`: frozen pretrained backbone with trainable
  classification head. One genuine public-EEG subject-1 optimizer step and
  held-out subject-2 forward pass succeeded in the same workflow.
- The invalid random-frozen-backbone `scratch + linear_probe` configuration
  is explicitly rejected and covered by regression tests.

## Small-subset overfitting diagnostic

- **Input:** 10 genuine public subject-1 EEG epochs: two from each of five classes
- **Task:** fit classification head on frozen pretrained representations
- **Initial training loss:** 1.6093803644180298
- **Final training loss:** 0.0001989185984712094
- **Optimizer steps:** 20
- **Final training-set accuracy:** 1.0
- **Required threshold:** accuracy 1.0 and loss below 0.1
- **Result:** PASS

Important: this is training-set memorization, not held-out test accuracy.
No subject-2 labels were used to select or optimize these examples. The result
does not establish model generalization, five-class benchmark quality, or
pretraining superiority over EEGNet or a genuinely trained scratch model.

## Checkpoint provenance and anti-corruption tests

A complete model state dict was serialized, its SHA-256 recorded, the
manifest independently checked, and the state dict loaded back with
`weights_only=True`. Reloaded-model logits matched saved-model logits.

Checkpoint SHA-256 from CI:

`e4bf6274b1ea9cd680c460cb263c8950e504b1f55f076a8f2cc889dd2904e293`

Regression tests additionally cover:

- Modified checkpoint bytes must fail verification
- Modified prediction bytes must fail verification
- Participant overlap must fail verification
- Unsupported manifest schema must fail verification
- Manifest overwrites must be rejected
- Artifacts outside the expected folder must be rejected

The full-data training entrypoint now records both checkpoint and prediction
hashes in its manifest and rejects existing checkpoints to avoid untracked
overwrites. **Its complete multi-subject GPU training path is not validated by
this limited CPU acceptance diagnostic.**

## Download reliability

Direct PhysioNet downloads intermittently failed with a network timeout.
The final workflow stages the six official imagery runs for each of subjects
1 and 2 from the public PhysioNet S3 bucket before loading via MOABB/MNE.
The successful run confirmed all 12 EDF files.

## Scientific acceptance conclusion

**PASS for the limited Phase 1 scientific-correctness gate**, conditional on
successful lint, unit, and artifact-integrity workflows for the PR revision
that is ultimately merged.

This explicitly does NOT certify:

1. Held-out EEG motor-imagery decoding accuracy or generalization
2. Convergence of full scratch training beyond its single real optimizer step
3. A multi-seed benchmark or independent confirmatory holdout
4. Generalization of the temporal-prefix adapter versus alternatives
5. GPU stability of multi-hour training

These concerns belong to subsequent data, baseline, and optimization phases.
Historical held-out metrics are exploratory and must not be used for tuning.
