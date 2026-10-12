"""PhysioNet EEGMMIDB channel-axis contract.

The array axis must match the EDF channel order; do not infer sensor order
from the shape (N, 64, T). Source order is checked before model inference.
"""
from __future__ import annotations

from collections.abc import Sequence

PHYSIONET_64_CH_NAMES = (
    "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6",
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6",
    "Fp1", "Fpz", "Fp2",
    "AF7", "AF3", "AFz", "AF4", "AF8",
    "F7", "F5", "F3", "F1", "Fz", "F2", "F4", "F6", "F8",
    "FT7", "FT8",
    "T7", "T8", "T9", "T10",
    "TP7", "TP8",
    "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8",
    "PO7", "PO3", "POz", "PO4", "PO8",
    "O1", "Oz", "O2", "Iz",
)


def canonicalize_channel_name(name: str) -> str:
    """MNE's EEGBCI EDF channels may have trailing dots and mixed case."""
    return str(name).strip().strip(".").upper()


def verify_channel_order(names: Sequence[str], expected=PHYSIONET_64_CH_NAMES) -> None:
    """Fail on missing, duplicated, or permuted channel axes."""
    actual = tuple(canonicalize_channel_name(name) for name in names)
    reference = tuple(canonicalize_channel_name(name) for name in expected)
    if len(actual) != len(reference) or len(set(actual)) != len(actual):
        raise ValueError(
            f"Invalid channel list: {len(actual)} channels, "
            f"{len(set(actual))} unique; expected {len(reference)} unique."
        )
    if actual != reference:
        mismatches = [
            f"{i}: observed={a}, expected={b}"
            for i, (a, b) in enumerate(zip(actual, reference))
            if a != b
        ]
        raise ValueError(
            "EEG channel axis differs from the PhysioNet model montage. "
            "Do not train with reordered or unknown channels. "
            + "; ".join(mismatches[:5])
        )


def verify_sampling_rate(sfreq: float, expected_sfreq: float = 160.0) -> None:
    if abs(float(sfreq) - expected_sfreq) > 1e-4:
        raise ValueError(
            f"Unexpected EEG sampling frequency {sfreq} Hz; "
            f"expected {expected_sfreq} Hz."
        )
