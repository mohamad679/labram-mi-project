"""Offline adversarial checks for strictly defined CBraMod input transfer."""
import numpy as np
import pytest

from src.phase5_cbramod import (
    CHECKPOINT_REVISION,
    CHECKPOINT_SHA256,
    MODEL_SAMPLES,
    RESAMPLED_SAMPLES,
    adapt_cbramod_input,
)


def test_exact_resampling_scale_and_padding():
    x = np.full((64, 400), 100., dtype=np.float32)
    out = adapt_cbramod_input(x)
    assert out.shape == (64, MODEL_SAMPLES)
    assert out.dtype == np.float32
    assert np.allclose(out[:, 100:400], 1., atol=1e-3)
    assert np.count_nonzero(out[:, RESAMPLED_SAMPLES:]) == 0


def test_no_untracked_channel_drop_or_reordering():
    x = np.zeros((64, 400), dtype=np.float32)
    x[20, 160] = 100.
    out = adapt_cbramod_input(x)
    assert np.allclose(out[np.arange(64) != 20], 0)
    assert np.any(out[20] != 0)


@pytest.mark.parametrize("shape", [(63, 400), (64, 399), (1, 64, 400), (64, 600)])
def test_wrong_shape_fails_closed(shape):
    with pytest.raises(ValueError, match="Expected"):
        adapt_cbramod_input(np.zeros(shape, dtype=np.float32))


def test_nonfinite_fails_closed():
    x = np.zeros((64, 400), dtype=np.float32)
    x[2, 3] = np.nan
    with pytest.raises(ValueError, match="finite"):
        adapt_cbramod_input(x)


def test_checkpoint_pinning_is_full_digest_and_revision():
    assert len(CHECKPOINT_REVISION) == 40
    assert len(CHECKPOINT_SHA256) == 64
