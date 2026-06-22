
"""
Phase 4 — Lightweight Gradio demo for LaBraM Motor Imagery BCI.

This demo accepts one EEG epoch as a .npy file with shape:
  - (64, 400), or
  - (1, 64, 400), or
  - (400, 64), which will be transposed automatically.

The demo loads the Phase 1 head-only checkpoint and returns:
  - predicted class
  - class probabilities
  - input-shape validation details

This is a research-engineering portfolio demo, not a clinical BCI system.
"""

import os
from pathlib import Path

import gradio as gr
import numpy as np
import torch

from src.model import load_labram_model


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

_MODEL_CACHE = None
_CHECKPOINT_PATH = None


def find_checkpoint() -> str:
    """Find the first available checkpoint path."""
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


def get_model():
    """Load and cache the LaBraM model."""
    global _MODEL_CACHE, _CHECKPOINT_PATH

    if _MODEL_CACHE is not None:
        return _MODEL_CACHE, _CHECKPOINT_PATH

    checkpoint_path = find_checkpoint()

    model = load_labram_model(
        n_outputs=N_OUTPUTS,
        n_times=N_TIMES,
        n_chans=N_CHANS,
        sfreq=SFREQ,
    ).to(DEVICE)

    state_dict = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()

    _MODEL_CACHE = model
    _CHECKPOINT_PATH = checkpoint_path

    return model, checkpoint_path


def validate_eeg_array(array: np.ndarray) -> tuple[np.ndarray, str]:
    """Validate and normalize uploaded EEG epoch shape."""
    array = np.asarray(array, dtype=np.float32)

    original_shape = array.shape

    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]

    if array.ndim != 2:
        raise ValueError(
            f"Expected a 2D EEG epoch, but got shape {original_shape}. "
            "Use shape (64, 400), (1, 64, 400), or (400, 64)."
        )

    if array.shape == (N_CHANS, N_TIMES):
        normalized = array

    elif array.shape == (N_TIMES, N_CHANS):
        normalized = array.T

    else:
        raise ValueError(
            f"Invalid EEG shape {original_shape}. "
            f"Expected ({N_CHANS}, {N_TIMES}) or ({N_TIMES}, {N_CHANS})."
        )

    if not np.isfinite(normalized).all():
        normalized = np.nan_to_num(
            normalized,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

    info = (
        f"Original shape: {original_shape}\n"
        f"Model input shape: {normalized.shape}\n"
        f"Device: {DEVICE}"
    )

    return normalized, info


def predict(file_path):
    """Run prediction for one uploaded .npy EEG epoch."""
    if file_path is None:
        return None, "Please upload a .npy file."

    try:
        eeg = np.load(file_path, allow_pickle=False)
        eeg, shape_info = validate_eeg_array(eeg)

        model, checkpoint_path = get_model()

        x = torch.tensor(eeg[None, :, :], dtype=torch.float32, device=DEVICE)

        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()

        pred_idx = int(np.argmax(probs))
        pred_label = CLASS_NAMES[pred_idx]

        prob_dict = {
            CLASS_NAMES[i]: float(probs[i])
            for i in range(len(CLASS_NAMES))
        }

        status = (
            "Prediction completed successfully.\n\n"
            f"Predicted class: {pred_label}\n"
            f"Checkpoint: {checkpoint_path}\n"
            f"{shape_info}\n\n"
            "Scope note: this is a research demo, not a clinical BCI system."
        )

        return prob_dict, status

    except Exception as exc:
        return None, f"Error: {type(exc).__name__}: {exc}"


def build_demo():
    with gr.Blocks(title="LaBraM Motor Imagery BCI Demo") as demo:
        gr.Markdown(
            """
            # LaBraM Motor Imagery BCI Demo

            Upload one EEG epoch as a `.npy` file.

            Expected shape:

            - `(64, 400)`
            - `(1, 64, 400)`
            - `(400, 64)` — automatically transposed

            Output classes:

            `feet`, `hands`, `left_hand`, `rest`, `right_hand`

            This is a research-engineering portfolio demo, not a clinical system.
            """
        )

        with gr.Row():
            file_input = gr.File(
                label="Upload EEG epoch (.npy)",
                file_types=[".npy"],
                type="filepath",
            )

        predict_button = gr.Button("Predict")

        probabilities = gr.Label(
            label="Class probabilities",
            num_top_classes=5,
        )

        status = gr.Textbox(
            label="Status",
            lines=10,
        )

        predict_button.click(
            fn=predict,
            inputs=file_input,
            outputs=[probabilities, status],
        )

    return demo


demo = build_demo()


if __name__ == "__main__":
    demo.launch()

