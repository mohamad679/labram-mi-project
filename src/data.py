"""
Phase 0 — Data loading for PhysioNet Motor Imagery via MOABB.
"""

from moabb.datasets import PhysionetMI
from moabb.paradigms import MotorImagery

LABRAM_PATCH_SIZE = 200  # LaBraM's internal patch size — n_times must be a multiple of this


def load_physionet_mi(subjects: list[int] | None = None):
    """
    Load PhysioNet Motor Imagery data via MOABB, cropped to a length
    compatible with LaBraM's patch size (200 samples/patch).
    """
    dataset = PhysionetMI()
    paradigm = MotorImagery()

    X, y, metadata = paradigm.get_data(dataset=dataset, subjects=subjects)

    n_times = X.shape[-1]
    usable_n_times = (n_times // LABRAM_PATCH_SIZE) * LABRAM_PATCH_SIZE
    if usable_n_times == 0:
        raise ValueError(
            f"Epoch length ({n_times}) is shorter than patch size ({LABRAM_PATCH_SIZE})."
        )
    if usable_n_times != n_times:
        print(f"Cropping n_times from {n_times} to {usable_n_times} (patch-size compatible)")
        X = X[..., :usable_n_times]

    return X, y, metadata


if __name__ == "__main__":
    X, y, metadata = load_physionet_mi(subjects=[1])
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("Unique labels:", set(y))
