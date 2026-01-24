import numpy as np
import pandas as pd
from datetime import datetime

from code.analysis.decaying_sine_fit import analyse_decaying_sine
from code.core_simulations.fast_simulation import run_simulation


def sweep(n, plots, filename, p_values, f_values, L):
    """
    Simulate all (p,f) combinations and fit decaying-sine parameters to tree coverage.

    This is a direct port of the notebook's sweep() function:
      - frames = int(56.063984 * f**(-0.398039))
      - optionally shows first and last fit plots if plots=True
      - appends results row-by-row to CSV

    Nothing is executed unless you call sweep().
    """
    total_runs = n ** 2
    run_index = 0

    columns = [
        "p", "f", "p/f", "frames",
        "C", "A", "d", "w", "phi",
        "C_err", "A_err", "d_err", "w_err", "phi_err",
        "r2_tree",
    ]
    pd.DataFrame(columns=columns).to_csv(filename, index=False)

    for p in p_values:
        for f in f_values:
            run_index += 1
            frames = int(56.063984 * f ** (-0.398039))
            print(
                f"Running: {run_index}/{total_runs}, p = {p:.6f}  f = {f:.6f},"
                f" f:p = 1:{p / f:.2f}, Frames = {frames}"
            )

            do_plot = bool(plots) and (run_index in {1, total_runs})

            tree_counts, _burn_counts = run_simulation(
                L=L, frames=frames, p=p, f=f, gif=False, save_2_file=False
            )

            params_tree, errors_tree, r2_tree = analyse_decaying_sine(
                tree_counts=tree_counts, burn_counts=None, L=L, show_plots=do_plot
            )

            C, A, d, w, phi = params_tree
            C_err, A_err, d_err, w_err, phi_err = errors_tree

            row = pd.DataFrame([{
                "p": p,
                "f": f,
                "p/f": p / f,
                "frames": frames,
                "C": C,
                "A": A,
                "d": d,
                "w": w,
                "phi": phi,
                "C_err": C_err,
                "A_err": A_err,
                "d_err": d_err,
                "w_err": w_err,
                "phi_err": phi_err,
                "r2_tree": r2_tree,
            }])
            row.to_csv(filename, mode="a", header=False, index=False)


def geometric_series(min_val, max_val, n):
    """Return n values in a geometric progression from min_val to max_val."""
    k = np.arange(n)
    r = (max_val / min_val) ** (1 / (n - 1))
    return min_val * (r ** k)


def generate_pf_combinations(n, p_min, p_max, ratio_min, ratio_max, base=65536):
    """
    Port of the notebook's 'Generation of p and f combinations' section.

    Returns:
      p_vals_prob, f_vals_prob, target_ratios, true_ratios, p_vals_int, f_vals_int
    """
    k = np.arange(n)

    # geometric series for integer p (scaled by base)
    r_p = (p_max / p_min) ** (1 / (n - 1))
    p = p_min * r_p ** k
    p_vals_int = np.round(p).astype(int)

    # geometric series for ratios (f:p = 1:x)
    x = (ratio_max / ratio_min) ** (1 / (n - 1))
    target_ratios = ratio_min * x ** k

    # derive f integers then convert to probs
    f = p_vals_int / target_ratios
    f_vals_int = np.round(f).astype(int)

    true_ratios = p_vals_int / f_vals_int

    # convert to true float probabilities
    p_vals_prob = p_vals_int / base
    f_vals_prob = f_vals_int / base

    return p_vals_prob, f_vals_prob, target_ratios, true_ratios, p_vals_int, f_vals_int


def estimate_sweep_total_frames(p_vals, f_vals):
    """Helper from the notebook: sum frames over all (p,f) combos."""
    frames = []
    for p in p_vals:
        for f in f_vals:
            frames.append(int(56.063984 * f ** (-0.398039)))
    return int(np.sum(frames))


def make_sweep_filename(L, prefix="Sweeping"):
    """Notebook-style filename helper (no writing, just returns a name)."""
    return f"{prefix}_L{L}_{datetime.now().strftime('%H%M%S')}.csv"


def mask_indices(n):
    """
    Returns the same index masks you built in the notebook for selecting subsets
    of the n*n sweep grid when doing log-log fits.
    """
    last = n * n
    fixed_p = np.arange(0, n)             # first row
    fixed_f = np.arange(n - 1, last, n)   # last column
    range_r = np.arange(0, last, n + 1)   # main diagonal
    fixed_r = np.arange(n - 1, last - 1, n - 1)  # anti-diagonal
    no_mask = np.arange(0, n * n)
    return {
        "fixed_p": fixed_p,
        "fixed_f": fixed_f,
        "range_r": range_r,
        "fixed_r": fixed_r,
        "all": no_mask,
    }
