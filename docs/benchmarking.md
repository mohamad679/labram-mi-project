# Evidence-backed benchmark generation

This stage **does not claim a new model score**. It provides deterministic artifacts from the probabilities of a completed, subject-held-out test evaluation.

## Produce real predictions

```bash
python -m src.train --config configs/baseline.yaml
```

After training completes, the selected validation checkpoint is used for one final held-out evaluation. The script saves `checkpoints/heldout_predictions.npz` with:

- `y_true`: integer labels (N,)
- `y_prob`: softmax probability array (N,5)
- `subject_ids`: held-out participant identifiers (N,)
- `class_names`: the ordered five-class label vocabulary

**Do not fabricate or hand-edit this artifact.** It contains no raw EEG waveform samples, but subject identifiers and prediction labels still deserve privacy-aware handling.

## Generate figures

```bash
python -m scripts.build_benchmark \
  --predictions checkpoints/heldout_predictions.npz \
  --output-dir artifacts/evaluation
```

Expected products: `metrics.json`, `metrics.csv`, `confusion_matrix.png` and `reliability_diagram.png`. Outputs are local and gitignored. Numerical scores are not committed until a full run is verified.

## Scientific gates before publishing numbers

1. Resolve and verify the pinned pretrained LaBraM checkpoint; default `model_init: scratch` does **not** satisfy this.
2. Verify channel ordering, preprocessing and signal units using real PhysioNet/MOABB data.
3. Re-run with independent seeds and record frozen dependencies, dataset identifiers and checkpoint hashes.
4. Run the baseline and the model under the same subject-wise evaluation protocol.
5. Report confidence intervals at the **subject level**: EEG epochs within one subject are not independent observations. Do not describe epoch-level bootstrap intervals as independent-subject intervals.
6. Keep held-out test subjects untouched throughout model selection.

The results visualization script has offline unit tests, but no end-to-end GPU result has been produced during this repository upgrade.
