"""Offline tests of MOABB-equivalent EEG scaling and provenance metadata."""
from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from src.channels import PHYSIONET_64_CH_NAMES
from src.data import load_physionet_mi


class FakeEpochs:
    def __init__(self, voltage):
        self._voltage = voltage
        self.ch_names = list(PHYSIONET_64_CH_NAMES)
        self.info = {"sfreq": 160.0}

    def get_data(self):
        return self._voltage.copy()


class FakeMetadata:
    def __init__(self, length=2):
        self.attrs = {}
        self.length = length

    def __len__(self):
        return self.length


def _install_mock_moabb(monkeypatch, voltage, factor=1e6, swap_channels=False):
    dataset_mod = types.ModuleType("moabb.datasets")
    paradigm_mod = types.ModuleType("moabb.paradigms")
    moabb_mod = types.ModuleType("moabb")

    class PhysionetMI:
        unit_factor = factor

    class MotorImagery:
        def get_data(self, dataset, subjects, return_epochs):
            assert return_epochs is True
            epochs = FakeEpochs(voltage)
            if swap_channels:
                epochs.ch_names[0], epochs.ch_names[1] = (
                    epochs.ch_names[1], epochs.ch_names[0]
                )
            return epochs, np.array(["rest", "feet"]), FakeMetadata()

    dataset_mod.PhysionetMI = PhysionetMI
    paradigm_mod.MotorImagery = MotorImagery
    moabb_mod.datasets = dataset_mod
    moabb_mod.paradigms = paradigm_mod
    monkeypatch.setitem(sys.modules, "moabb", moabb_mod)
    monkeypatch.setitem(sys.modules, "moabb.datasets", dataset_mod)
    monkeypatch.setitem(sys.modules, "moabb.paradigms", paradigm_mod)


def test_loader_matches_moabb_standard_array_scaling(monkeypatch):
    voltage = np.full((2, 64, 401), 0.00001, dtype=np.float64)
    _install_mock_moabb(monkeypatch, voltage)
    data, labels, metadata = load_physionet_mi(subjects=[1])
    assert data.shape == (2, 64, 400)
    np.testing.assert_allclose(data, voltage[..., :400] * 1e6, rtol=0, atol=1e-12)
    assert metadata.attrs["input_unit"] == "uV"
    assert metadata.attrs["cache_schema_version"] == 2
    assert metadata.attrs["scaling_factor_from_volts"] == 1e6
    assert list(labels) == ["rest", "feet"]


def test_loader_rejects_reordered_montage(monkeypatch):
    _install_mock_moabb(monkeypatch, np.zeros((2, 64, 400)), swap_channels=True)
    with pytest.raises(ValueError, match="channel axis differs"):
        load_physionet_mi(subjects=[1])


def test_loader_rejects_unexpected_unit_scaling(monkeypatch):
    _install_mock_moabb(monkeypatch, np.zeros((2, 64, 400)), factor=1)
    with pytest.raises(ValueError, match="scaling factor"):
        load_physionet_mi(subjects=[1])
