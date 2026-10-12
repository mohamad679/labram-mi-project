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
from src.provenance import validate_pretrained_reference

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
    pretrained_revision: str | None = None,
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
    # The official checkpoint is configured for 3,000 samples (15 patches).
    # Shorter windows need explicit time-position transfer; strict=False
    # alone does NOT resolve temporal_embedding size mismatches.
    if pretrained_repo_id is not None:
        validate_pretrained_reference(pretrained_repo_id, pretrained_revision)
        if pretrained_repo_id != "braindecode/labram-pretrained":
            raise ValueError("The audited temporal-prefix adapter supports only "
                             "braindecode/labram-pretrained.")
        if n_times > 3000 or n_times <= 0 or n_times % 200:
            raise ValueError("Pretrained LaBraM windows must be 200..3000 "
                             "samples and divisible by the patch size 200.")
        pretrained = InterpolatedLaBraM.from_pretrained(
            pretrained_repo_id, chs_info=chs_info,
            n_outputs=n_outputs, n_times=3000, n_chans=n_chans,
            sfreq=sfreq, revision=pretrained_revision, strict=False, **kwargs,
        )
        if n_times == 3000:
            return pretrained

        # Rebuild the architecture for the actual patch count. Transfer
        # every compatible backbone tensor, but never an old task head.
        target = InterpolatedLaBraM(
            chs_info=chs_info, n_outputs=n_outputs, n_times=n_times,
            n_chans=n_chans, sfreq=sfreq, **kwargs,
        )
        source_weights = pretrained.state_dict()
        target_weights = target.state_dict()
        weights_to_transfer = {}
        for name, expected in target_weights.items():
            if name.startswith("final_layer."):
                continue
            if name not in source_weights:
                raise RuntimeError(f"Pretrained checkpoint missing backbone tensor: {name}")
            source = source_weights[name]
            if name == "temporal_embedding":
                source = source[:, : expected.shape[1], :].clone()
            if source.shape != expected.shape:
                raise RuntimeError(
                    f"Unadapted pretrained weight mismatch for {name}: "
                    f"source={tuple(source.shape)}, target={tuple(expected.shape)}"
                )
            weights_to_transfer[name] = source
        missing, unexpected = target.load_state_dict(
            weights_to_transfer, strict=False
        )
        if unexpected or any(not name.startswith("final_layer.") for name in missing):
            raise RuntimeError(f"Unexpected transfer keys: missing={missing}, "
                               f"unexpected={unexpected}")
        return target

    if pretrained_revision is not None:
        raise ValueError('pretrained_revision requires pretrained_repo_id.')

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
