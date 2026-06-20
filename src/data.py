"""
Phase 0 — Data loading for PhysioNet Motor Imagery via MOABB.
"""
import numpy as np
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



from sklearn.model_selection import GroupShuffleSplit

RANDOM_STATE = 42

def subject_wise_split(X, y, subject_ids,
                        test_size_subjects=15,
                        val_size_subjects=14,
                        random_state=RANDOM_STATE):
    """
    Leakage-free subject-wise split.
    No subject appears in more than one split.
    Returns: train_idx, val_idx, test_idx (indices into X/y)
    """
    subject_ids = np.asarray(subject_ids)
    n_subjects = len(np.unique(subject_ids))

    gss_test = GroupShuffleSplit(n_splits=1, test_size=test_size_subjects,
                                  random_state=random_state)
    trainval_idx, test_idx = next(gss_test.split(X, y, groups=subject_ids))

    gss_val = GroupShuffleSplit(n_splits=1, test_size=val_size_subjects,
                                 random_state=random_state)
    train_rel, val_rel = next(gss_val.split(
        trainval_idx, groups=subject_ids[trainval_idx]))
    train_idx, val_idx = trainval_idx[train_rel], trainval_idx[val_rel]

    s_tr, s_val, s_te = (set(subject_ids[train_idx]),
                          set(subject_ids[val_idx]),
                          set(subject_ids[test_idx]))
    assert not (s_tr & s_val) and not (s_tr & s_te) and not (s_val & s_te), \
        "Subject leakage detected!"

    print(f"{len(s_tr)} train / {len(s_val)} val / {len(s_te)} test "
          f"subjects (of {n_subjects})")
    return train_idx, val_idx, test_idx

