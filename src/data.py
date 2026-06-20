"""
Phase 0 — Data loading for PhysioNet Motor Imagery via MOABB.

Reuses the same loading pattern from the thesis (MOABB-based pipeline),
now feeding into braindecode's LaBraM model instead of the classical
raw-EEG / PSD pipeline.
"""

from moabb.datasets import PhysionetMI
from moabb.paradigms import MotorImagery


def load_physionet_mi(subjects: list[int] | None = None):
    """
    Load PhysioNet Motor Imagery data via MOABB.

    Args:
        subjects: list of subject IDs to load. None = all subjects.
                  Start with a small subset (e.g. [1, 2, 3]) while
                  testing the pipeline in Phase 0 — do NOT load all
                  109 subjects until the forward pass test passes.

    Returns:
        X: EEG epochs array
        y: labels array
        metadata: MOABB metadata dataframe
    """
    dataset = PhysionetMI()
    paradigm = MotorImagery()

    X, y, metadata = paradigm.get_data(dataset=dataset, subjects=subjects)

    # TODO (Phase 0 exit criteria):
    # Print X.shape here and confirm channel count matches what
    # braindecode's Labram model expects. If it does not match,
    # this is the channel-mapping step flagged in the roadmap risk register.
    return X, y, metadata


if __name__ == "__main__":
    # Quick manual smoke test — run this first, on 1-2 subjects only.
    X, y, metadata = load_physionet_mi(subjects=[1])
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("Unique labels:", set(y))
