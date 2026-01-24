import numpy as np
from code.analysis.critical_time import A, B

def critical_regime_start(f):
    return int((np.log(1000) / A) * (f ** (-B)))

def simulation_length(f):
    return 2 * critical_regime_start(f)
