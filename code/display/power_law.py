import numpy as np
import matplotlib.pyplot as plt

from code.analysis.power_law_fit import fit_powerlaw


def plot_powerlaw(sizes, L, p, f, regime):

    sizes = np.sort(sizes)  # Sort sizes into assending order (make debugging alot easier)
    
    # Generate logarithmic bins (for historgram, created linearly then converted to base 10 logarithm)
    num_bins = 300
    log_bins = np.linspace(np.log10(sizes.min()), np.log10(sizes.max()), num_bins)
    bins = 10**log_bins

    # Combine values into respective bins and find the center of each bin
    counts, edges = np.histogram(sizes, bins=bins)
    s_vals = np.sqrt(edges[:-1] * edges[1:])

    # Some bins (typically with high N, could have 0 fires so apply a mask to remove empty bins, else div 0 error)
    mask = counts > 0
    s_fit = s_vals[mask]
    c_fit = counts[mask]

    
    A, alpha, A_err, alpha_err, r2 = fit_powerlaw(s_fit, c_fit) # Fit pure power law using curve_fit

    plt.figure(figsize=(12, 5))
    plt.scatter(s_fit, c_fit, s=30, label="Simulation data") # Plot data

    s_ref = np.linspace(s_fit.min(), s_fit.max(), 500) # Recreate fit from parameters to display model
    c_ref = A * s_ref**(-alpha)
    plt.plot(s_ref, c_ref, '--', label="N(s) = A s^(-α)")

    plt.xlabel("Fire size (s)")
    plt.ylabel("Fire count [N(s)]")
    plt.title(f"{regime} Region (pure power law)\n L={L}, p={p:.4f}, f={f:.5f}, A={A:.0f}±{A_err:.0f}, α={alpha:.3f}±{alpha_err:.3f}, R^2={r2:.3f}")
    plt.xscale("log")
    plt.yscale("log")
    plt.grid(True, alpha=0.3)
    plt.legend()
    #plt.ylim(30, 30000)   # Fires sizes that have less than 30 entires have a very low sample size / inconsistent data
    #plt.xlim(1, 100)
    plt.tight_layout()
    plt.show()
