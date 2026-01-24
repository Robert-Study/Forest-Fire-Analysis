import numpy as np

# Exponential decay model fitted from your sweep output:
# D(f) = A * f^B
# time threshold t = ln(threshold_inverse) / D(f) = (ln(threshold_inverse)/A) * f^(-B)

A = 0.061399940272506136
B = 0.3081816633412444


def oscillatory_regime_end(f):
    """Frame index when oscillations have decayed to 10% of initial amplitude (ln(10))."""
    return int((np.log(10) / A) * (f ** (-B)))


def critical_regime_start(f):
    """Frame index when oscillations have decayed to 0.1% of initial amplitude (ln(1000))."""
    return int((np.log(1000) / A) * (f ** (-B)))


def simulation_length(f):
    """Total frames to run: 2 * critical_regime_start(f)."""
    return 2 * critical_regime_start(f)
