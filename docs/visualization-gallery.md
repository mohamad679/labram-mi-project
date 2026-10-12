# Visualization gallery

This gallery uses **existing checked-in figures**, not fabricated results. They illustrate what the code and dashboard can display; they do not validate clinical suitability, physiological attribution or model performance.

## Transformer attention inspection

| Existing channel ranking | Existing temporal-patch ranking |
| --- | --- |
| ![Archived token/channel importance ranking for sample zero](../results/figures/phase3_attention_rollout/sample_0_top_channels.png) | ![Archived temporal patch importance for sample zero](../results/figures/phase3_attention_rollout/sample_0_time_patches.png) |

These are archived **attention rollout visualizations**. Their semantic interpretation depends on the token-to-channel mapping, temporal cropping and exact checkpoint. Attention magnitude is not a validated measurement of cortical activity or causal contribution. Other archived examples are included in the [Phase 3 report](../results/phase3_attention_rollout.md).

## Interactive demonstration

![Screenshot of archived Gradio classification demo](../results/figures/phase4_gradio_demo/gradio_ui_prediction.png)

[Watch the previously recorded dashboard video](../results/figures/phase4_gradio_demo/gradio_dashboard_demo.mp4).

The recording and screenshot are historical demonstration artifacts. The repository does **not** guarantee that a fresh user can run the application without supplying a compatible local checkpoint.

## What must be generated from a new real evaluation

| Figure | Input artifact | Acceptance rule |
| --- | --- | --- |
| Five-class confusion matrix | Real `y_true` and `y_prob` from test subjects | Each image links to a run ID, split manifest and prediction hash |
| Reliability diagram | Held-out class probabilities | State the ECE binning method and sample support |
| Training/validation curves | Per-epoch JSONL or CSV log | Must not report values inferred from README prose |
| Per-subject distributions and uncertainty | Subject-associated predictions | Use subject-aware intervals; avoid epoch-independence assumptions |
| Benchmark comparison | Multiple identical, split-locked protocols | Clearly distinguish scratch, pretrained, and external conventional baselines |

The [benchmark generator](benchmarking.md) already produces some plots from a real NPZ output. No new accuracy claim is made by this gallery.
