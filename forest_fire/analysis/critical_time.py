import numpy as np

# Exponential decay model fitted from your sweep output:
# D(f) = A * f^B
# time threshold t = ln(threshold_inverse) / D(f) = (ln(threshold_inverse)/A) * f^(-B)

A = 0.061399940272506136
B = 0.3081816633412444


def oscillatory_regime_end(f):
    """Frame index when oscillations have decayed to 10% of initial amplitude (ln(10))."""
    if not np.isfinite(f) or not 0 < f <= 1:
        raise ValueError('The empirical transient model requires 0 < f <= 1.')
    return int((np.log(10) / A) * (f ** (-B)))


def critical_regime_start(f):
    """Historical sampling heuristic, not a statistical test of criticality.

    Frame index when the fitted oscillation envelope reaches 0.1%.
    The constants are empirical and need recalibration outside the original sweep.
    """
    if not np.isfinite(f) or not 0 < f <= 1:
        raise ValueError('The empirical transient model requires 0 < f <= 1.')
    return int((np.log(1000) / A) * (f ** (-B)))


def simulation_length(f):
    """Total frames to run: 2 * critical_regime_start(f)."""
    return 2 * critical_regime_start(f)
