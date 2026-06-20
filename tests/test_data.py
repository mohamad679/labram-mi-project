"""
Basic sanity tests — run these BEFORE moving past Phase 0.
"""

from src.data import load_physionet_mi


def test_data_shapes():
    """Smoke test: confirm data loads with expected dimensionality."""
    X, y, metadata = load_physionet_mi(subjects=[1])
    assert X.ndim == 3, f"Expected 3D array (epochs, channels, time), got shape {X.shape}"
    assert len(y) == X.shape[0], "Label count must match number of epochs"


def test_no_subject_leakage():
    """
    Placeholder for the leakage check that mattered so much in the
    thesis protocol — subject IDs in train and test splits must
    never overlap. Fill this in once train/test splitting is implemented
    in Phase 1.
    """
    pass  # TODO: implement once train.py has a real split function
