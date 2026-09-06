import numpy as np
import matplotlib.pyplot as plt

from forest_fire.analysis.truncated_power_law_fit import fit_truncated_powerlaw


def plot_truncated_powerlaw(sizes, L, p, f, regime):  # This code is identical to plot_powerlaw with correct variables / model

    sizes = np.sort(sizes)

    num_bins = 500
    log_bins = np.linspace(np.log10(sizes.min()), np.log10(sizes.max()), num_bins)
    bins = 10**log_bins

    counts, edges = np.histogram(sizes, bins=bins)
    s_vals = np.sqrt(edges[:-1] * edges[1:])

    mask = counts > 0
    s_fit = s_vals[mask]
    c_fit = counts[mask]

    # Fit truncated power law 
    A, alpha, s_c, A_err, alpha_err, s_c_err, r2 = fit_truncated_powerlaw(s_fit, c_fit)

    plt.figure(figsize=(10, 5))

    plt.scatter(s_fit, c_fit, s=30, label="Simulation data")

    s_ref = np.linspace(s_fit.min(), s_fit.max(), 500)
    plt.plot(s_ref, A * s_ref**(-alpha) * np.exp(-s_ref / s_c), '--', label="N(s) = A s^(-α) exp(-s / s_c)")

    plt.xlabel("Fire size s")
    plt.ylabel("Fire count N(s)")
    plt.title(f"{regime} regime (truncated power law)\n L={L}, p={p:.3f}, f={f:.3f}, α={alpha:.2f}±{alpha_err:.2f}, s_c={s_c:.1f}±{s_c_err:.1f}, R^2={r2:.3f}")
    plt.xscale("log")
    plt.yscale("log")
    plt.ylim(bottom=1)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

