import pandas as pd

from code.analysis.equilibrium_coverage import extract_equilibrium_coverage, extract_growth_probability
from code.analysis.exponential_decay import loglog


def plot_equilibrium_coverage_vs_p(mask, sweep_csv, x_label="Probability of Growth (p)", y_label="Equilibrium Tree Coverage (C)"):
    """
    Convenience wrapper: loads sweep CSV, extracts equilibrium coverage (C) and errors,
    then makes the notebook-style log-log fit/plot using loglog().
    """
    df = pd.read_csv(sweep_csv)
    p = extract_growth_probability(df)
    C, C_err = extract_equilibrium_coverage(df)

    return loglog(mask, p, C, C_err, x_label, y_label, "Equilibrium coverage vs p", model="C = A p^B")
