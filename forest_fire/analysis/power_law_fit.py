import numpy as np
from scipy.optimize import curve_fit


def model(s, A, alpha):
    # Power law model: N(s) = A s^(-alpha)
    return A * s**(-alpha)


def fit_powerlaw(s_fit, c_fit):  # s_fit = fire size values, c_fit = corresponding fire counts N(s)
    parameters, covariance = curve_fit(model, s_fit, c_fit, p0=[1, 1])  # p0: Initial guesses for A and alpha

    A = parameters[0]                                # Best-fit prefactor
    alpha = parameters[1]                            # Best-fit exponent

    A_err = np.sqrt(covariance[0, 0])                # Uncertainty in A from covariance
    alpha_err = np.sqrt(covariance[1, 1])            # Uncertainty in alpha

    # Compute R^2
    c_fit_model = model(s_fit, A, alpha)             # Model prediction at data points
    r2 = 1 - np.sum((c_fit - c_fit_model)**2) / np.sum((c_fit - c_fit.mean())**2)

    return A, alpha, A_err, alpha_err, r2

