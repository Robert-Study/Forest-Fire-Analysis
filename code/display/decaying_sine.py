import matplotlib.pyplot as plt

from code.analysis.decaying_sine_fit import analyse_decaying_sine


def plot_decaying_sine_fit(tree_counts, burn_counts, L):
    """Convenience wrapper: runs analyse_decaying_sine with show_plots=True."""
    return analyse_decaying_sine(tree_counts=tree_counts, burn_counts=burn_counts, L=L, show_plots=True)
