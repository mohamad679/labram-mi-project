"""Validate EEG array channel provenance without installing MOABB."""
import pytest

from src.channels import (
    PHYSIONET_64_CH_NAMES,
    verify_channel_order,
    verify_sampling_rate,
)


def test_expected_montage_contains_64_distinct_channels():
    assert len(PHYSIONET_64_CH_NAMES) == 64
    verify_channel_order(PHYSIONET_64_CH_NAMES)


def test_edf_case_and_trailing_periods_are_normalized():
    names = [name.lower() + "." for name in PHYSIONET_64_CH_NAMES]
    verify_channel_order(names)


def test_reordered_eeg_channels_are_rejected():
    names = list(PHYSIONET_64_CH_NAMES)
    names[0], names[1] = names[1], names[0]
    with pytest.raises(ValueError, match="channel axis differs"):
        verify_channel_order(names)


def test_duplicate_eeg_channels_are_rejected():
    names = list(PHYSIONET_64_CH_NAMES)
    names[-1] = names[0]
    with pytest.raises(ValueError, match="Invalid channel list"):
        verify_channel_order(names)


def test_wrong_sampling_frequency_is_rejected():
    verify_sampling_rate(160.0)
    with pytest.raises(ValueError, match="160"):
        verify_sampling_rate(128.0)
