import numpy as np
from scipy.optimize import curve_fit


def fit_truncated_powerlaw(s_fit, c_fit): 

    def model(s, A, alpha, s_c):                   # Truncated power law: N(s) = A s^(-alpha) exp(-s / s_c)
        return A * s**(-alpha) * np.exp(-s / s_c)  # Compared to previous model this includes the extra term exp(-s / s_c)

    parameters, covariance = curve_fit(model, s_fit, c_fit, bounds=([0, 0, 1], [np.inf, np.inf, np.inf]))
                                       # Lower bound has to be set to prevent s_c shooting off to negative inf

    A = parameters[0]                              
    alpha = parameters[1]                           
    s_c = parameters[2]                   # Best-fit cutoff scale (a fire size where linearity exponentially falls off / truncated, this is the value where truncation has resulted allready in a 1/e drop)

    A_err = np.sqrt(covariance[0, 0])               
    alpha_err = np.sqrt(covariance[1, 1])         
    s_c_err = np.sqrt(covariance[2, 2])             

    c_fit_model = model(s_fit, A, alpha, s_c)       
    r2 = 1 - np.sum((c_fit - c_fit_model)**2) / np.sum((c_fit - c_fit.mean())**2)

    return A, alpha, s_c, A_err, alpha_err, s_c_err, r2

