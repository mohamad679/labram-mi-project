"""
Phase 4 — Recordable Gradio dashboard for LaBraM Motor Imagery BCI.

Dashboard tabs:
  1. Classify    — deterministic prediction + probability chart
  2. Uncertainty — MC Dropout + entropy/BALD visualization
  3. Explain     — attention-rollout heatmap + channel/time importance

Input: one EEG epoch as .npy with shape (64, 400), (1, 64, 400), or
(400, 64). This is a research-engineering portfolio demo, not a clinical
BCI system.
"""

from __future__ import annotations

import copy
import os
import tempfile
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import gradio as gr
from braindecode.models.labram import LABRAM_CHANNEL_ORDER

from src.explain import attention_rollout, top_k_channels
from src.model import load_labram_model
from src.uncertainty import enable_mc_dropout, inject_mc_dropout, mc_dropout_predict


CLASS_NAMES = ["feet", "hands", "left_hand", "rest", "right_hand"]

N_CHANS = 64
N_TIMES = 400
N_OUTPUTS = 5
SFREQ = 160.0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CHECKPOINT_CANDIDATES = [
    os.environ.get("LABRAM_CHECKPOINT"),
    "/kaggle/working/labram_head_finetuned.pt",
    "checkpoints/labram_head_finetuned.pt",
    "labram_head_finetuned.pt",
]

_MODEL_CACHE: dict[str, tuple[nn.Module, str]] = {}


# ---------------------------------------------------------------------------
# General utilities
# ---------------------------------------------------------------------------
def _as_path(file_path: Any) -> str | None:
    """Normalize Gradio file outputs across versions."""
    if file_path is None:
        return None
    if isinstance(file_path, str):
        return file_path
    if isinstance(file_path, dict):
        return file_path.get("path") or file_path.get("name")
    if hasattr(file_path, "name"):
        return str(file_path.name)
    return str(file_path)


def find_checkpoint() -> str:
    """Find the first available Phase 1 checkpoint path."""
    for candidate in CHECKPOINT_CANDIDATES:
        if candidate and Path(candidate).exists():
            return str(candidate)

    searched = [str(p) for p in CHECKPOINT_CANDIDATES if p]
    raise FileNotFoundError(
        "No checkpoint found. Searched paths:\n"
        + "\n".join(f"- {p}" for p in searched)
        + "\n\nSet LABRAM_CHECKPOINT or place the checkpoint at "
        "checkpoints/labram_head_finetuned.pt."
    )


def load_state_dict_robust(checkpoint_path: str) -> dict[str, torch.Tensor]:
    """Load checkpoints saved either as a raw state_dict or a wrapper dict."""
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]
    if not isinstance(checkpoint, dict):
        raise TypeError(
            f"Unsupported checkpoint format at {checkpoint_path}: {type(checkpoint)}"
        )
    return checkpoint


def get_model(mode: str = "deterministic", dropout_p: float = 0.3) -> tuple[nn.Module, str]:
    """
    Load and cache a LaBraM model.

    mode='deterministic': eval model for normal prediction.
    mode='mc_dropout': separate model with MC Dropout injected and enabled.
    """
    checkpoint_path = find_checkpoint()

    if mode == "deterministic":
        cache_key = f"deterministic::{checkpoint_path}"
    elif mode == "mc_dropout":
        cache_key = f"mc_dropout::{checkpoint_path}::p={float(dropout_p):.4f}"
    else:
        raise ValueError(f"Unknown model mode: {mode}")

    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    model = load_labram_model(
        n_outputs=N_OUTPUTS,
        n_times=N_TIMES,
        n_chans=N_CHANS,
        sfreq=SFREQ,
    ).to(DEVICE)

    state_dict = load_state_dict_robust(checkpoint_path)
    model.load_state_dict(state_dict)

    if mode == "deterministic":
        model.eval()
    else:
        inject_mc_dropout(model, dropout_p=float(dropout_p))
        enable_mc_dropout(model)

    _MODEL_CACHE[cache_key] = (model, checkpoint_path)
    return model, checkpoint_path


def validate_eeg_array(array: np.ndarray) -> tuple[np.ndarray, str]:
    """Validate and normalize uploaded EEG epoch shape to (64, 400)."""
    array = np.asarray(array, dtype=np.float32)
    original_shape = array.shape

    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]

    if array.ndim != 2:
        raise ValueError(
            f"Expected a 2D EEG epoch, but got shape {original_shape}. "
            "Use shape (64, 400), (1, 64, 400), or (400, 64)."
        )

    transposed = False
    if array.shape == (N_CHANS, N_TIMES):
        normalized = array
    elif array.shape == (N_TIMES, N_CHANS):
        normalized = array.T
        transposed = True
    else:
        raise ValueError(
            f"Invalid EEG shape {original_shape}. "
            f"Expected ({N_CHANS}, {N_TIMES}) or ({N_TIMES}, {N_CHANS})."
        )

    replaced_nonfinite = False
    if not np.isfinite(normalized).all():
        normalized = np.nan_to_num(
            normalized,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        replaced_nonfinite = True

    info = (
        f"Original shape: {original_shape}\n"
        f"Model input shape: {normalized.shape}\n"
        f"Auto-transposed: {transposed}\n"
        f"Non-finite values replaced: {replaced_nonfinite}\n"
        f"Mean: {float(normalized.mean()):.6f}\n"
        f"Std: {float(normalized.std()):.6f}\n"
        f"Device: {DEVICE}"
    )

    return normalized.astype(np.float32, copy=False), info


def load_eeg_from_file(file_path: Any) -> tuple[np.ndarray, str]:
    """Load one .npy EEG epoch from Gradio file input."""
    path = _as_path(file_path)
    if path is None:
        raise ValueError("Please upload a .npy file first.")
    if not str(path).lower().endswith(".npy"):
        raise ValueError("Only .npy files are supported.")

    raw = np.load(path, allow_pickle=False)
    return validate_eeg_array(raw)


def tensor_from_epoch(eeg: np.ndarray) -> torch.Tensor:
    """Convert normalized EEG array to model tensor with batch dimension."""
    return torch.tensor(eeg[None, :, :], dtype=torch.float32, device=DEVICE)


def softmax_predict(model: nn.Module, x: torch.Tensor) -> tuple[np.ndarray, int, str]:
    """Run one deterministic forward pass."""
    model.eval()
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
    pred_idx = int(np.argmax(probs))
    pred_label = CLASS_NAMES[pred_idx]
    return probs, pred_idx, pred_label


def probability_dict(probs: np.ndarray) -> dict[str, float]:
    """Convert probability vector to Gradio Label dictionary."""
    return {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}


def prediction_summary(
    probs: np.ndarray,
    pred_label: str,
    true_label: str | None,
    checkpoint_path: str,
    shape_info: str,
) -> str:
    """Build a readable status block for the UI."""
    sorted_idx = np.argsort(probs)[::-1]
    confidence = float(probs[sorted_idx[0]])
    margin = float(probs[sorted_idx[0]] - probs[sorted_idx[1]])

    truth_line = "True label: unknown / not provided"
    if true_label and true_label != "unknown":
        is_correct = pred_label == true_label
        truth_line = f"True label: {true_label}\nPrediction correct: {is_correct}"

    lines = [
        "Prediction completed successfully.",
        "",
        f"Predicted class: {pred_label}",
        f"Top confidence: {confidence:.4f}",
        f"Top-1 vs Top-2 margin: {margin:.4f}",
        truth_line,
        "",
        f"Checkpoint: {checkpoint_path}",
        shape_info,
        "",
        "Scope note: research-engineering demo only; not a clinical BCI system.",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------
def make_probability_plot(probs: np.ndarray, title: str = "Class probabilities"):
    fig = plt.figure(figsize=(7, 4))
    plt.bar(CLASS_NAMES, probs)
    plt.ylim(0.0, 1.0)
    plt.ylabel("Probability")
    plt.title(title)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    return fig


def make_uncertainty_probability_plot(mean_probs: np.ndarray, std_probs: np.ndarray):
    fig = plt.figure(figsize=(7, 4))
    plt.bar(CLASS_NAMES, mean_probs, yerr=std_probs, capsize=4)
    plt.ylim(0.0, 1.0)
    plt.ylabel("Mean probability ± std")
    plt.title("MC Dropout probability dispersion")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    return fig


def make_bald_plot(pred_entropy: float, expected_ent: float, bald_score: float):
    fig = plt.figure(figsize=(6, 4))
    names = ["Predictive\nentropy", "Expected\nentropy", "BALD\nepistemic"]
    values = [pred_entropy, expected_ent, bald_score]
    plt.bar(names, values)
    plt.ylabel("Uncertainty value")
    plt.title("Entropy decomposition and BALD score")
    plt.tight_layout()
    return fig


def make_attention_heatmap(token_importance: np.ndarray, top_k: int):
    grid = token_importance.reshape(128, 2)
    channel_scores = grid.mean(axis=1)
    order = np.argsort(channel_scores)[::-1][:top_k]
    heat = grid[order]
    names = [LABRAM_CHANNEL_ORDER[i] for i in order]

    fig = plt.figure(figsize=(6, max(4, 0.28 * top_k)))
    plt.imshow(heat, aspect="auto")
    plt.colorbar(label="Attention rollout importance")
    plt.xticks([0, 1], ["patch_0", "patch_1"])
    plt.yticks(np.arange(len(names)), names)
    plt.xlabel("Temporal patch")
    plt.ylabel("Top canonical LaBraM channels")
    plt.title("Attention rollout heatmap")
    plt.tight_layout()
    return fig


def make_top_channel_plot(channel_scores: np.ndarray, top_k: int):
    top = top_k_channels(channel_scores[None, :], k=top_k)[0]
    names = [name for name, _ in top][::-1]
    scores = [score for _, score in top][::-1]

    fig = plt.figure(figsize=(7, max(4, 0.28 * top_k)))
    plt.barh(names, scores)
    plt.xlabel("Attention rollout importance")
    plt.ylabel("Canonical LaBraM channel")
    plt.title(f"Top-{top_k} channels")
    plt.tight_layout()
    return fig


def make_time_patch_plot(time_scores: np.ndarray):
    fig = plt.figure(figsize=(5, 4))
    plt.bar(["patch_0", "patch_1"], time_scores)
    plt.ylabel("Mean attention rollout importance")
    plt.title("Temporal patch importance")
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Demo sample helper
# ---------------------------------------------------------------------------
def generate_demo_sample() -> tuple[str, str]:
    """
    Generate a synthetic EEG-like .npy file for UI recording.

    This is only for testing the interface. It is not a real PhysioNet sample.
    """
    rng = np.random.default_rng(42)
    t = np.arange(N_TIMES, dtype=np.float32) / SFREQ
    eeg = 0.15 * rng.standard_normal((N_CHANS, N_TIMES)).astype(np.float32)

    # Add weak mu/beta-band rhythms over central channels so the signal is not
    # visually/randomly empty. Labels remain unknown because this is synthetic.
    central_channels = [7, 8, 9, 10, 11, 12, 13]
    for ch in central_channels:
        eeg[ch] += 0.35 * np.sin(2 * np.pi * 10.0 * t)
        eeg[ch] += 0.18 * np.sin(2 * np.pi * 20.0 * t)

    out_dir = Path(tempfile.gettempdir()) / "labram_gradio_demo"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "synthetic_eeg_epoch_64x400.npy"
    np.save(out_path, eeg.astype(np.float32))

    info = (
        "Synthetic demo sample generated.\n"
        f"Path: {out_path}\n"
        f"Shape: {eeg.shape}\n"
        "Use this only to record/test the dashboard UI; it is not a real EEG label."
    )
    return str(out_path), info


def describe_system() -> str:
    """Return a dashboard status panel."""
    try:
        checkpoint = find_checkpoint()
        checkpoint_status = f"✅ Checkpoint found: `{checkpoint}`"
    except Exception as exc:
        checkpoint_status = f"⚠️ Checkpoint not found: `{type(exc).__name__}: {exc}`"

    return f"""
### Runtime status

- Device: `{DEVICE}`
- Input shape: `({N_CHANS}, {N_TIMES})`
- Classes: `{', '.join(CLASS_NAMES)}`
- {checkpoint_status}

The app opens even without a checkpoint, but prediction/uncertainty/explain buttons need the Phase 1 `.pt` checkpoint.
"""


# ---------------------------------------------------------------------------
# Tab 1 — Classify
# ---------------------------------------------------------------------------
def run_classification(file_path: Any, true_label: str):
    try:
        eeg, shape_info = load_eeg_from_file(file_path)
        model, checkpoint_path = get_model(mode="deterministic")
        probs, _, pred_label = softmax_predict(model, tensor_from_epoch(eeg))

        return (
            probability_dict(probs),
            prediction_summary(probs, pred_label, true_label, checkpoint_path, shape_info),
            make_probability_plot(probs),
        )
    except Exception as exc:
        return None, f"Error: {type(exc).__name__}: {exc}", None


# ---------------------------------------------------------------------------
# Tab 2 — Uncertainty
# ---------------------------------------------------------------------------
def run_uncertainty(file_path: Any, n_samples: int, dropout_p: float, true_label: str):
    try:
        eeg, shape_info = load_eeg_from_file(file_path)
        model, checkpoint_path = get_model(mode="mc_dropout", dropout_p=float(dropout_p))
        x = tensor_from_epoch(eeg)

        results = mc_dropout_predict(model, x, n_samples=int(n_samples))

        mean_probs = results["mean_probs"][0]
        std_probs = results["std_probs"][0]
        pred_entropy = float(results["pred_entropy"][0])
        expected_ent = float(results["expected_ent"][0])
        bald_score = float(results["bald_score"][0])
        pred_idx = int(results["predictions"][0])
        pred_label = CLASS_NAMES[pred_idx]

        max_entropy = float(np.log(len(CLASS_NAMES)))
        confidence = float(mean_probs.max())
        mean_std = float(std_probs.mean())

        truth_line = "True label: unknown / not provided"
        if true_label and true_label != "unknown":
            truth_line = (
                f"True label: {true_label}\n"
                f"Prediction correct: {pred_label == true_label}"
            )

        status = "\n".join(
            [
                "MC Dropout uncertainty completed successfully.",
                "",
                f"Predicted class from mean probability: {pred_label}",
                f"Mean confidence: {confidence:.4f}",
                f"Predictive entropy: {pred_entropy:.4f} / max {max_entropy:.4f}",
                f"Expected entropy: {expected_ent:.4f}",
                f"BALD score: {bald_score:.4f}",
                f"Mean probability std: {mean_std:.4f}",
                f"MC stochastic passes: {int(n_samples)}",
                f"Dropout p: {float(dropout_p):.2f}",
                truth_line,
                "",
                f"Checkpoint: {checkpoint_path}",
                shape_info,
                "",
                "Interpretation: high entropy + low confidence means the baseline is uncertain; BALD estimates epistemic uncertainty from dropout variability.",
            ]
        )

        return (
            probability_dict(mean_probs),
            status,
            make_uncertainty_probability_plot(mean_probs, std_probs),
            make_bald_plot(pred_entropy, expected_ent, bald_score),
        )
    except Exception as exc:
        return None, f"Error: {type(exc).__name__}: {exc}", None, None


# ---------------------------------------------------------------------------
# Tab 3 — Explain
# ---------------------------------------------------------------------------
def run_explainability(file_path: Any, top_k: int, true_label: str):
    try:
        eeg, shape_info = load_eeg_from_file(file_path)
        model, checkpoint_path = get_model(mode="deterministic")
        x = tensor_from_epoch(eeg)

        probs, _, pred_label = softmax_predict(model, x)
        rollout_results = attention_rollout(model, x)

        token_importance = rollout_results["token_importance"][0]
        channel_scores = rollout_results["channel_importance_128"][0]
        time_scores = rollout_results["time_patch_importance"][0]

        top = top_k_channels(channel_scores[None, :], k=int(top_k))[0]
        top_text = "\n".join(
            f"{rank:02d}. {name:<10} {score:.6f}"
            for rank, (name, score) in enumerate(top, start=1)
        )

        truth_line = "True label: unknown / not provided"
        if true_label and true_label != "unknown":
            truth_line = (
                f"True label: {true_label}\n"
                f"Prediction correct: {pred_label == true_label}"
            )

        status = "\n".join(
            [
                "Attention rollout explainability completed successfully.",
                "",
                f"Predicted class: {pred_label}",
                f"Top confidence: {float(probs.max()):.4f}",
                truth_line,
                "",
                f"Token importance shape: {token_importance.shape}",
                f"Channel importance shape: {channel_scores.shape}",
                f"Time patch importance shape: {time_scores.shape}",
                "",
                f"Checkpoint: {checkpoint_path}",
                shape_info,
                "",
                "Top channels:",
                top_text,
                "",
                "Scope note: attention rollout is an inspection method, not a clinically validated EEG explanation.",
            ]
        )

        return (
            status,
            make_attention_heatmap(token_importance, top_k=int(top_k)),
            make_top_channel_plot(channel_scores, top_k=int(top_k)),
            make_time_patch_plot(time_scores),
        )
    except Exception as exc:
        return f"Error: {type(exc).__name__}: {exc}", None, None, None


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
#main-title {text-align: center;}
.metric-card {border: 1px solid #ddd; border-radius: 12px; padding: 12px;}
"""


def build_demo() -> gr.Blocks:
    with gr.Blocks(
        title="LaBraM Motor Imagery BCI Dashboard",
        css=CUSTOM_CSS,
    ) as demo:
        gr.Markdown(
            """
<div id="main-title">

# LaBraM Motor Imagery BCI Dashboard

**Phase 4 upgraded demo:** Classify → Uncertainty → Explainability

</div>

Upload one EEG epoch as `.npy`. Supported shapes: `(64, 400)`, `(1, 64, 400)`, or `(400, 64)`.
This dashboard is for research-engineering portfolio demonstration only; it is not a clinical BCI tool.
            """
        )

        with gr.Row():
            runtime_status = gr.Markdown(describe_system())
        refresh_status = gr.Button("Refresh runtime status")
        refresh_status.click(fn=describe_system, inputs=[], outputs=runtime_status)

        with gr.Accordion("Suggested recording flow", open=True):
            gr.Markdown(
                """
1. Open the dashboard and show the three tabs: **Classify**, **Uncertainty**, **Explain**.
2. In **Classify**, upload a real `.npy` epoch or generate a synthetic demo file, then click **Run prediction**.
3. In **Uncertainty**, run MC Dropout and show mean probability, probability std, entropy, and BALD score.
4. In **Explain**, run attention rollout and show the heatmap, top channels, and temporal patch plot.
5. End by saying the baseline is intentionally modest, but the pipeline is complete: model loading, inference, uncertainty, and explainability.
                """
            )

        with gr.Tabs():
            with gr.Tab("1. Classify"):
                gr.Markdown(
                    """
### Deterministic classification

This tab is the clean portfolio entry point: upload one EEG epoch, run one forward pass, and show the predicted motor-imagery class with a probability chart.
                    """
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        cls_file = gr.File(
                            label="Upload EEG epoch (.npy)",
                            file_types=[".npy"],
                            type="filepath",
                        )
                        cls_true = gr.Dropdown(
                            choices=["unknown"] + CLASS_NAMES,
                            value="unknown",
                            label="Optional true label",
                        )
                        cls_generate = gr.Button("Generate synthetic demo .npy")
                        cls_generated_info = gr.Textbox(
                            label="Generated sample info",
                            lines=4,
                        )
                        cls_run = gr.Button("Run prediction", variant="primary")
                    with gr.Column(scale=2):
                        cls_label = gr.Label(
                            label="Class probabilities",
                            num_top_classes=5,
                        )
                        cls_status = gr.Textbox(label="Prediction status", lines=14)
                        cls_plot = gr.Plot(label="Probability chart")

                cls_generate.click(
                    fn=generate_demo_sample,
                    inputs=[],
                    outputs=[cls_file, cls_generated_info],
                )
                cls_run.click(
                    fn=run_classification,
                    inputs=[cls_file, cls_true],
                    outputs=[cls_label, cls_status, cls_plot],
                )

            with gr.Tab("2. Uncertainty"):
                gr.Markdown(
                    """
### MC Dropout uncertainty

This tab makes Phase 2 visible for recording. It injects dropout into the verified LaBraM locations, runs stochastic forward passes, and visualizes mean probability, standard deviation, predictive entropy, expected entropy, and BALD.
                    """
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        unc_file = gr.File(
                            label="Upload EEG epoch (.npy)",
                            file_types=[".npy"],
                            type="filepath",
                        )
                        unc_true = gr.Dropdown(
                            choices=["unknown"] + CLASS_NAMES,
                            value="unknown",
                            label="Optional true label",
                        )
                        unc_samples = gr.Slider(
                            minimum=5,
                            maximum=100,
                            value=30,
                            step=5,
                            label="MC stochastic passes",
                        )
                        unc_dropout = gr.Slider(
                            minimum=0.05,
                            maximum=0.60,
                            value=0.30,
                            step=0.05,
                            label="Dropout probability",
                        )
                        unc_generate = gr.Button("Generate synthetic demo .npy")
                        unc_generated_info = gr.Textbox(
                            label="Generated sample info",
                            lines=4,
                        )
                        unc_run = gr.Button("Run MC Dropout", variant="primary")
                    with gr.Column(scale=2):
                        unc_label = gr.Label(
                            label="Mean class probabilities",
                            num_top_classes=5,
                        )
                        unc_status = gr.Textbox(label="Uncertainty status", lines=18)
                        unc_prob_plot = gr.Plot(label="Mean probability ± std")
                        unc_bald_plot = gr.Plot(label="BALD / entropy chart")

                unc_generate.click(
                    fn=generate_demo_sample,
                    inputs=[],
                    outputs=[unc_file, unc_generated_info],
                )
                unc_run.click(
                    fn=run_uncertainty,
                    inputs=[unc_file, unc_samples, unc_dropout, unc_true],
                    outputs=[unc_label, unc_status, unc_prob_plot, unc_bald_plot],
                )

            with gr.Tab("3. Explain"):
                gr.Markdown(
                    """
### Attention rollout explainability

This tab makes Phase 3 visible for recording. It computes CLS-token attention rollout and summarizes it as a top-channel heatmap, top-channel bar chart, and temporal patch importance.
                    """
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        exp_file = gr.File(
                            label="Upload EEG epoch (.npy)",
                            file_types=[".npy"],
                            type="filepath",
                        )
                        exp_true = gr.Dropdown(
                            choices=["unknown"] + CLASS_NAMES,
                            value="unknown",
                            label="Optional true label",
                        )
                        exp_top_k = gr.Slider(
                            minimum=5,
                            maximum=30,
                            value=16,
                            step=1,
                            label="Top channels shown",
                        )
                        exp_generate = gr.Button("Generate synthetic demo .npy")
                        exp_generated_info = gr.Textbox(
                            label="Generated sample info",
                            lines=4,
                        )
                        exp_run = gr.Button("Run attention rollout", variant="primary")
                    with gr.Column(scale=2):
                        exp_status = gr.Textbox(label="Explainability status", lines=22)
                        exp_heatmap = gr.Plot(label="Attention rollout heatmap")
                        exp_channel_plot = gr.Plot(label="Top-channel importance")
                        exp_time_plot = gr.Plot(label="Temporal patch importance")

                exp_generate.click(
                    fn=generate_demo_sample,
                    inputs=[],
                    outputs=[exp_file, exp_generated_info],
                )
                exp_run.click(
                    fn=run_explainability,
                    inputs=[exp_file, exp_top_k, exp_true],
                    outputs=[exp_status, exp_heatmap, exp_channel_plot, exp_time_plot],
                )

    return demo


if __name__ == "__main__":
    share = os.getenv("GRADIO_SHARE", "0").lower() in {"1", "true", "yes"}
    server_name = os.getenv("GRADIO_SERVER_NAME")
    server_port = os.getenv("GRADIO_SERVER_PORT")

    launch_kwargs: dict[str, Any] = {
        "share": share,
        "show_error": True,
    }
    if server_name:
        launch_kwargs["server_name"] = server_name
    if server_port:
        launch_kwargs["server_port"] = int(server_port)

    app = build_demo()
    app.queue()
    app.launch(**launch_kwargs)
