"""
Phase 0 — Load the LaBraM model via braindecode.

This file creates an InterpolatedLaBraM model for the 64-channel
PhysioNet Motor Imagery EEG data.

Important:
Newer versions of braindecode require `chs_info` for InterpolatedLaBraM.
`chs_info` describes the EEG channel names and locations.
"""

import torch
import mne
from braindecode.models import InterpolatedLaBraM


from src.channels import PHYSIONET_64_CH_NAMES

def make_chs_info(n_chans: int, sfreq: float = 160.0):
    """
    Create MNE channel information for InterpolatedLaBraM.

    InterpolatedLaBraM needs chs_info, not only n_chans.
    chs_info is obtained from an MNE Info object via info["chs"].
    """
    if n_chans != len(PHYSIONET_64_CH_NAMES):
        raise ValueError(
            f"This project currently expects 64 EEG channels, but got n_chans={n_chans}. "
            "If you use another dataset, you must define the correct channel names."
        )

    info = mne.create_info(
        ch_names=PHYSIONET_64_CH_NAMES,
        sfreq=sfreq,
        ch_types="eeg",
    )

    # standard_1005 contains the standard EEG sensor positions.
    info.set_montage("standard_1005")

    return info["chs"]


def load_labram_model(
    n_outputs: int,
    n_times: int,
    n_chans: int,
    sfreq: float = 160.0,
    pretrained_repo_id: str | None = None,
    **kwargs,
):
    """
    Create an InterpolatedLaBraM model.

    Parameters
    ----------
    n_outputs:
        Number of classes.
    n_times:
        Number of time samples per EEG epoch.
    n_chans:
        Number of EEG channels.
    sfreq:
        Sampling frequency.
    """
    if n_outputs is None:
        raise ValueError("n_outputs must be provided.")

    if n_times is None or n_chans is None:
        raise ValueError(
            "n_times and n_chans must be provided. Run src/data.py first, "
            "check X.shape, and pass X.shape[-1] as n_times and X.shape[1] as n_chans."
        )

    chs_info = make_chs_info(n_chans=n_chans, sfreq=sfreq)

    # A constructor alone does NOT establish pretrained-weight provenance.
    # Only an explicit from_pretrained call activates transfer learning.
    if pretrained_repo_id is not None:
        return InterpolatedLaBraM.from_pretrained(
            pretrained_repo_id, chs_info=chs_info,
            n_outputs=n_outputs, n_times=n_times, n_chans=n_chans,
            sfreq=sfreq, strict=False, **kwargs,
        )

    model = InterpolatedLaBraM(
        chs_info=chs_info,
        n_outputs=n_outputs,
        n_times=n_times,
        n_chans=n_chans,
        sfreq=sfreq,
        **kwargs,
    )

    return model


if __name__ == "__main__":
    from src.data import load_physionet_mi

    X, y, metadata = load_physionet_mi(subjects=[1])

    n_times = X.shape[-1]
    n_chans = X.shape[1]
    n_outputs = len(set(y))

    print(f"Inferred n_times={n_times}, n_chans={n_chans} from data shape {X.shape}")
    print(f"Inferred n_outputs={n_outputs} from labels: {sorted(set(y))}")

    model = load_labram_model(
        n_outputs=n_outputs,
        n_times=n_times,
        n_chans=n_chans,
        sfreq=160.0,
    )

    # Real forward pass test on actual data.
    model.eval()
    dummy_batch = torch.tensor(X[:4], dtype=torch.float32)

    with torch.no_grad():
        out = model(dummy_batch)

    print("Forward pass succeeded. Output shape:", out.shape)
    print("Total parameters:", sum(p.numel() for p in model.parameters()))
