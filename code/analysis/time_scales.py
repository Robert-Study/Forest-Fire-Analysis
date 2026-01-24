import numpy as np

# Fitted decay constants from your analysis
A = 0.061399940272506136
B = 0.3081816633412444


def oscillatory_regime_end(f: float) -> int:
    """
    End of strong oscillations (10% of original amplitude).

    t = ln(10) / A * f^(-B)
    """
    return int((np.log(10) / A) * (f ** (-B)))


def critical_regime_start(f: float) -> int:
    """
    Start of critical regime (0.1% of original amplitude).

    t = ln(1000) / A * f^(-B)
    """
    return int((np.log(1000) / A) * (f ** (-B)))


def simulation_length(f: float) -> int:
    """
    Total simulation length policy.

    Defined as:
        2 × critical_regime_start(f)
    """
    return 2 * critical_regime_start(f)
