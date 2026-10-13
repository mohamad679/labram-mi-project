"""Versioned, explicit CBraMod input and trusted pretrained-weight contract.

Never use random weights as a substitute for a missing foundation checkpoint.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

CHECKPOINT_REPO = "braindecode/cbramod-pretrained"
CHECKPOINT_REVISION = "584cdc415913739a05d84bf0c1cb3db397764507"
CHECKPOINT_SHA256 = "a939ace9aa1e229f08391ad8bb2d197b507ae2c519a50addf087f0151b2df5c3"
SOURCE_SFREQ = 160
TARGET_SFREQ = 200
SOURCE_SAMPLES = 400
RESAMPLED_SAMPLES = 500
PATCH_SIZE = 200
MODEL_SAMPLES = 600
N_CHANNELS = 64
# From CBraMod pretraining: one model input unit represents 100 microvolts.
MICROVOLTS_PER_MODEL_UNIT = 100.0


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def adapt_cbramod_input(eeg_microvolts: np.ndarray) -> np.ndarray:
    """Preserve all original 2.5s and zero-pad 0.5s *after* 160→200Hz resampling.

    Shape (64,400) → (64,500) → (64,600). No learned statistics;
    every partition receives exactly the same predetermined operation.
    """
    x = np.asarray(eeg_microvolts)
    if x.shape != (N_CHANNELS, SOURCE_SAMPLES):
        raise ValueError(f"Expected (64,400) microvolt EEG, received {x.shape}")
    if not np.issubdtype(x.dtype, np.number) or not np.isfinite(x).all():
        raise ValueError("Non-numeric or non-finite EEG input")
    # 5/4, exact rational sample-rate conversion with polyphase anti-aliasing.
    resampled = resample_poly(x.astype(np.float64) / MICROVOLTS_PER_MODEL_UNIT,
                              5, 4, axis=-1)
    if resampled.shape != (N_CHANNELS, RESAMPLED_SAMPLES):
        raise RuntimeError("Resampling changed expected duration")
    out = np.pad(resampled, ((0, 0), (0, MODEL_SAMPLES-RESAMPLED_SAMPLES)),
                 mode="constant").astype(np.float32)
    if out.shape != (N_CHANNELS, MODEL_SAMPLES) or not np.isfinite(out).all():
        raise RuntimeError("Invalid CBraMod input adapter")
    if not np.array_equal(out[:, RESAMPLED_SAMPLES:],
                          np.zeros((N_CHANNELS, 100), dtype=np.float32)):
        raise RuntimeError("Nonzero synthetic padding")
    return out


def load_pinned_cbramod(checkpoint_path: str | Path, *, device="cpu"):
    """Load *only* exact public safetensors, check all backbone keys strictly.

    Official pretrained weights have no task head. A new downstream 5-class
    head is explicitly initialized; backbone must match every tensor exactly.
    """
    import torch
    from braindecode.models import CBraMod
    from safetensors.torch import load_file

    checkpoint_path = Path(checkpoint_path)
    found_sha = checksum(checkpoint_path)
    if found_sha != CHECKPOINT_SHA256:
        raise ValueError(f"Pretrained checkpoint SHA mismatch: {found_sha}")
    torch_model = CBraMod(n_outputs=5, n_chans=N_CHANNELS,
                          n_times=MODEL_SAMPLES, sfreq=TARGET_SFREQ,
                          patch_size=PATCH_SIZE, return_encoder_output=True)
    weights = load_file(str(checkpoint_path), device="cpu")
    expected = torch_model.state_dict()
    if set(weights) != set(expected):
        missing = sorted(set(expected)-set(weights))[:10]
        extra = sorted(set(weights)-set(expected))[:10]
        raise RuntimeError(f"Pretraining key mismatch: missing={missing}, extra={extra}")
    for key, parameter in expected.items():
        if weights[key].shape != parameter.shape:
            raise RuntimeError(f"Pretrained key {key} shape mismatch")
    torch_model.load_state_dict(weights, strict=True)

    class PinnedCBraModClassifier(torch.nn.Module):
        def __init__(self, pretrained):
            super().__init__()
            self.backbone = pretrained
            self.head = torch.nn.Sequential(torch.nn.LayerNorm(200),
                                            torch.nn.Linear(200, 5))

        def forward(self, x):
            features = self.backbone(x, return_features=True)["features"]
            if features.ndim != 4 or features.shape[1:] != (64, 3, 200):
                raise RuntimeError(f"Unexpected CBraMod features: {tuple(features.shape)}")
            return self.head(features.mean(dim=(1, 2)))

    model = PinnedCBraModClassifier(torch_model).to(device)
    return model, {"checkpoint_repo": CHECKPOINT_REPO,
                   "checkpoint_revision": CHECKPOINT_REVISION,
                   "checkpoint_sha256": found_sha,
                   "exact_backbone_key_count": len(weights),
                   "head_initialized_randomly": True,
                   "pretrained_backbone_strict_load": True}
