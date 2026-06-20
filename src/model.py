"""
Phase 0 — Load the pretrained LaBraM model via braindecode.

Uses InterpolatedLaBraM because our 64-channel PhysioNet montage
doesn't match LaBraM's canonical 128-channel pretraining layout.
"""

import torch
from braindecode.models import InterpolatedLaBraM


def load_labram_model(n_outputs: int = 2, n_times: int = None, n_chans: int = None, sfreq: float = 160.0, **kwargs):
    if n_times is None or n_chans is None:
        raise ValueError(
            "n_times and n_chans must be provided. Run src/data.py first, "
            "check X.shape, and pass X.shape[-1] as n_times and X.shape[1] as n_chans."
        )

    model = InterpolatedLaBraM(
        n_outputs=n_outputs, n_times=n_times, n_chans=n_chans, sfreq=sfreq, **kwargs
    )
    return model


if __name__ == "__main__":
    from src.data import load_physionet_mi

    X, y, metadata = load_physionet_mi(subjects=[1])
    n_times = X.shape[-1]
    n_chans = X.shape[1]
    print(f"Inferred n_times={n_times}, n_chans={n_chans} from data shape {X.shape}")

    model = load_labram_model(n_outputs=2, n_times=n_times, n_chans=n_chans, sfreq=160.0)

    # Real forward pass test on actual data — this IS the Phase 0
    # exit criteria, not printing the model summary.
    model.eval()
    dummy_batch = torch.tensor(X[:4], dtype=torch.float32)
    with torch.no_grad():
        out = model(dummy_batch)
    print("Forward pass succeeded. Output shape:", out.shape)
    print("Total parameters:", sum(p.numel() for p in model.parameters()))
