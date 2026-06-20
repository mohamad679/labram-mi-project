"""
Phase 0 — Load the pretrained LaBraM model via braindecode.

License: BSD-3-Clause (braindecode/labram-pretrained on HuggingFace).
"""

from braindecode.models import Labram


def load_labram_model(
    n_outputs: int = 2,
    n_times: int = None,
    n_chans: int = None,
    sfreq: float = 160.0,
    **kwargs,
):
    """
    Load LaBraM and prepare it for fine-tuning.

    Args:
        n_outputs: number of classes.
        n_times: number of time samples per epoch — from X.shape[-1].
        n_chans: number of EEG channels — from X.shape[1].
        sfreq: sampling frequency in Hz (PhysioNet EEGMMIDB = 160 Hz).
    """
    if n_times is None or n_chans is None:
        raise ValueError(
            "n_times and n_chans must be provided. Run src/data.py first, "
            "check X.shape, and pass X.shape[-1] as n_times and "
            "X.shape[1] as n_chans."
        )

    model = Labram(
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
    print(model)
    print("Total parameters:", sum(p.numel() for p in model.parameters()))
