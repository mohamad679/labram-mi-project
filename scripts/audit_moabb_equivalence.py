"""Compare MOABB ndarray and MNE-Epochs results on the SAME real subject.

No raw EEG or personally identifying sample-level data is written to artifacts.
Run after agreeing to the upstream PhysioNet dataset usage terms.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from src.channels import verify_channel_order, verify_sampling_rate
from src.data import LABRAM_PATCH_SIZE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--output", default="artifacts/moabb-equivalence.json")
    parser.add_argument("--atol-uv", type=float, default=1e-4)
    args = parser.parse_args()
    from moabb.datasets import PhysionetMI
    from moabb.paradigms import MotorImagery
    import moabb
    import mne

    dataset = PhysionetMI()
    paradigm = MotorImagery()
    arrays, labels_numpy, metadata_numpy = paradigm.get_data(
        dataset=dataset, subjects=[args.subject], return_epochs=False
    )
    epochs, labels_epochs, metadata_epochs = paradigm.get_data(
        dataset=dataset, subjects=[args.subject], return_epochs=True
    )
    verify_channel_order(epochs.ch_names)
    verify_sampling_rate(epochs.info["sfreq"])
    if list(labels_numpy) != list(labels_epochs):
        raise RuntimeError("Label/order mismatch between MOABB output modes.")
    index_fields = ["subject", "session", "run"]
    for field in index_fields:
        if list(metadata_numpy[field].astype(str)) != list(
            metadata_epochs[field].astype(str)
        ):
            raise RuntimeError(f"Metadata mismatch in '{field}' column.")
    if arrays.shape != epochs.get_data().shape:
        raise RuntimeError(
            f"Input array shape mismatch: {arrays.shape} vs {epochs.get_data().shape}"
        )
    factor = float(dataset.unit_factor)
    if not np.isclose(factor, 1e6):
        raise RuntimeError("This pipeline requires MOABB scaling by 1e6.")
    candidate = epochs.get_data() * factor
    samples = (candidate.shape[-1] // LABRAM_PATCH_SIZE) * LABRAM_PATCH_SIZE
    if samples < LABRAM_PATCH_SIZE:
        raise RuntimeError("Epoch shorter than one LaBraM patch.")
    diff = np.abs(np.asarray(arrays[..., :samples]) - candidate[..., :samples])
    maximum = float(np.max(diff))
    mean = float(np.mean(diff))
    same = bool(np.allclose(
        arrays[..., :samples], candidate[..., :samples], rtol=1e-6,
        atol=args.atol_uv, equal_nan=False,
    ))
    report = {
        "subject_id": args.subject,
        "epochs": int(len(labels_numpy)),
        "channels": int(arrays.shape[1]),
        "times_before_crop": int(arrays.shape[-1]),
        "times_after_crop": int(samples),
        "channel_order_verified": True,
        "metadata_order_verified": True,
        "unit_factor": factor,
        "unit": "uV",
        "max_abs_difference_uV": maximum,
        "mean_abs_difference_uV": mean,
        "array_equivalence_passed": same,
        "moabb_version": moabb.__version__,
        "mne_version": mne.__version__,
        "scope": "single public subject; not complete dataset reproducibility",
    }
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not same:
        raise RuntimeError("MOABB output paths did not agree numerically.")


if __name__ == "__main__":
    main()
