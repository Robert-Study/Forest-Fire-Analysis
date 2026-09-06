import numpy as np
import matplotlib.pyplot as plt


def plot_pf_combinations(p_vals, f_vals, ratio_arr, p_arr, f_arr, mask, fixed_p, fixed_f, fixed_r, range_r):
    """
    Direct port of the notebook cell:
    'Visualisation of simulations initial parameters'

    Inputs:
      - p_vals, f_vals: the 1D arrays of unique p and f values (float probs)
      - p_arr, f_arr: the flattened n*n arrays of p and f used in the sweep
      - ratio_arr: flattened ratio array p/f (or precomputed ratios to annotate)
      - mask and the 4 line masks to highlight.

    This function only plots; it doesn't run any simulations.
    """
    x_log = np.log(p_arr[mask])
    y_log = np.log(f_arr[mask])

    plt.figure(figsize=(7, 7))
    plt.scatter(x_log, y_log, s=40, color='darkblue', label="Simulation params")

    for xl, yl, ratio in zip(x_log, y_log, ratio_arr[mask]):
        plt.text(xl * 1.01, yl * 0.99, f"1:{ratio:.0f}", fontsize=8)

    plt.title("All (p,f) combinations simulated shown on a log-log plot")
    plt.xlabel("Probability of Growth (p)")
    plt.ylabel("Probability of Lightning (f)")

    plt.xticks(np.log(p_vals), labels=[f"{p:.4f}" for p in p_vals])
    plt.yticks(np.log(f_vals), labels=[f"{f:.5f}" for f in f_vals])

    plt.grid(which="major", linestyle="--", alpha=0.5)

    plt.plot(np.log(p_arr[fixed_p]), np.log(f_arr[fixed_p]), 'g-', alpha=0.2, label="Fixed p")
    plt.plot(np.log(p_arr[fixed_f]), np.log(f_arr[fixed_f]), 'y-', alpha=0.3, label="Fixed f")
    plt.plot(np.log(p_arr[fixed_r]), np.log(f_arr[fixed_r]), 'r-', alpha=0.2, label="Fixed ratio")
    plt.plot(np.log(p_arr[range_r]), np.log(f_arr[range_r]), 'm-', alpha=0.2, label="Full ratio range")

    plt.legend(fontsize=8, labelspacing=0.3)
    plt.tight_layout()
    plt.show()

