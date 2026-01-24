import numpy as np
from code.analysis.critical_time import A, B

def oscillatory_regime_end(f):
    """Alias for code.analysis.critical_time.oscillatory_regime_end."""
    return int((np.log(10) / A) * (f ** (-B)))
