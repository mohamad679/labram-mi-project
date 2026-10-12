# Phase 3 — Experimental validation gates

## Evidence status

**Code and unit tests are NOT empirical EEG results.** The offline regression tests use synthetic arrays only. Do not update README benchmark scores or commit generated scientific figures until real held-out runs have completed and their run metadata have been reviewed.

### Gate A — EEG unit convention

MOABB default array output applies `dataset.unit_factor` (normally `1e6`) to MNE EEG values originally in volts. The new `return_epochs=True` route now explicitly applies that factor, records **uV** as the unit and writes cache schema **2**. Caches created by earlier refactors claiming unit `V` must be rebuilt.

`tests/test_moabb_equivalence.py` exercises unit scaling and crop consistency with a synthetic MNE-style mock. It does **not** validate the actual MOABB download, epochs processing or EDF channel ordering on a real recording.

**Pending experiment:** Compare `MotorImagery().get_data(..., return_epochs=False)` versus `return_epochs=True` for the **same PhysioNet subject**. Verify label/subject/run alignment, channel ordering, number of epochs and numerical equality after the unit-factor correction. Archive package versions and only aggregate numerical results, not raw EEG.

### Gate B — Pinned pretrained checkpoint

The official `braindecode/labram-pretrained` model is pinned to Hugging Face commit
`0563b6c626e7b40d9a36653b763715db94d945d7`.
This revision has a `model.safetensors` file. The official model was configured for 128 canonical channels and 3000 time samples, rather than the project's 64 x 400 input.

**Verified CPU run:** [GitHub Actions run 38234001055](https://github.com/mohamad679/labram-mi-project/actions/runs/38234001055) used Braindecode 1.5.1, torch 2.5.1+cpu, and the immutable checkpoint above. At 3000 samples, 221/221 compatible backbone tensors matched exactly. A direct 400-sample from_pretrained initially failed due to temporal_embedding shape [1,16,200] vs [1,3,200]. The adapter now loads 3000-sample pretrained weights and transfers 220/220 compatible 400-sample backbone tensors and an exact first-three-position temporal prefix. A synthetic (1,64,400) forward pass produced finite (1,5) logits. **This validates weight transfer and numerical execution, not EEG classification accuracy or historical results.**

### Gate C — Real evaluation plots

The training script saves `checkpoints/heldout_predictions.npz` (not committed) alongside the checkpoint and metadata JSON. It contains `y_true`, `y_prob`, class labels and test subject IDs.

After a successful real-data training run:

```bash
python -m scripts.render_evaluation \
  --predictions checkpoints/labram_head_finetuned_test_predictions.npz \
  --run-metadata checkpoints/labram_head_finetuned.json \
  --output artifacts/evaluation
```

The existing benchmark renderer from PR #5 validates probability shape and row sums, then generates `metrics.json`, `metrics.csv`, `confusion_matrix.png` and `reliability_diagram.png`. It does not fabricate data when predictions are missing. **Epoch-level accuracy is not a confidence interval over independent subjects.**

### Gate D — Final scientific benchmark (not done)

- Confirm n_times=400 pretrained model compatibility and appropriate task-head replacement.
- Verify EDF preprocessing against MOABB output for representative subjects.
- Repeat split-locked training across multiple seeds, compare against meaningful conventional EEG baselines, and report subject-level uncertainty intervals.
- Publish only a scientific figure with a linked provenance manifest and a checksum of its prediction artifact.
- Preserve all historic scores as explicitly unverified, never claim clinical readiness.
