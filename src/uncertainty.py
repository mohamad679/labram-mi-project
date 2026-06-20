"""
Phase 2 — Uncertainty quantification (MC Dropout or Conformal Prediction).

NOT implemented yet — stub for Phase 2.
Scope reminder: pick ONE method, not both. Do not turn this into
a comparison study — that is out of scope for this 20-day project.
"""


def mc_dropout_predict(model, x, n_samples: int = 20):
    """TODO (Phase 2): run n_samples stochastic forward passes with
    dropout enabled, return mean prediction + variance as uncertainty."""
    raise NotImplementedError


def conformal_calibrate(model, calibration_set):
    """TODO (Phase 2, alternative to MC Dropout): compute conformal
    prediction intervals using a held-out calibration set."""
    raise NotImplementedError
