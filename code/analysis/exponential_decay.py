import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def power_law(x, A, B):
    """Power-law model used throughout the notebook: y = A * x**B."""
    return A * x**B


def fit_power_law_with_errors(x, y, y_err):
    """
    Fit y = A * x**B in (x,y) space using curve_fit with sigma=y_err.
    Returns (A, B, A_err, B_err, r2_log10).
    R^2 is computed in log10-space, matching the notebook.
    """
    params, covariance = curve_fit(power_law, x, y, sigma=y_err, absolute_sigma=True)
    A, B = params
    A_err, B_err = np.sqrt(np.diag(covariance))

    log_y = np.log10(y)
    log_y_fit = np.log10(power_law(x, A, B))
    r2 = 1 - np.sum((log_y - log_y_fit) ** 2) / np.sum((log_y - np.mean(log_y)) ** 2)

    return A, B, A_err, B_err, r2


def loglog(mask, x_arr, y_arr, y_err_arr, x_label, y_label, title, model):
    """
    Notebook-style log-log plot + fit helper.
    Applies mask to (x_arr, y_arr, y_err_arr), fits a power law, plots data + fit,
    and returns (A, B).

    This is the function you used after the parameter sweep to make:
      - equilibrium coverage vs f
      - equilibrium coverage vs p
      - angular frequency vs p
      - exponential decay vs f
    """
    x = x_arr[mask]
    y = y_arr[mask]
    y_err = y_err_arr[mask]

    A, B, A_err, B_err, r2 = fit_power_law_with_errors(x, y, y_err)

    # Smooth curve for plotting
    log_x_fit = np.linspace(np.log10(x.min()), np.log10(x.max()), 300)
    x_fit = 10 ** log_x_fit
    y_fit = power_law(x_fit, A, B)

    plt.figure(figsize=(12, 6))
    plt.scatter(x, y, label="Simulation Data")
    plt.plot(x_fit, y_fit, 'k--', alpha=0.8, label=f"Fit: {model}")
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(f"{title}\n A={A:.4f}, B={B:.2f}, R²={r2:.2f}")

    # Only show ticks at the actual data points
    plt.xticks(x, labels=[f"{v:.3g}" for v in x])
    plt.yticks(y, labels=[f"{v:.3g}" for v in y])
    plt.minorticks_off()

    plt.grid(alpha=0.3, which="both", linestyle='--')
    plt.legend()
    plt.tight_layout()
    plt.show()

    return A, B
