import pandas as pd

from forest_fire.analysis.equilibrium_coverage import extract_equilibrium_coverage, extract_lightning_probability
from forest_fire.analysis.exponential_decay import loglog


def plot_equilibrium_coverage_vs_f(mask, sweep_csv, x_label="Probability of Lightning (f)", y_label="Equilibrium Tree Coverage (C)"):
    df = pd.read_csv(sweep_csv)
    f = extract_lightning_probability(df)
    C, C_err = extract_equilibrium_coverage(df)

    return loglog(mask, f, C, C_err, x_label, y_label, "Equilibrium coverage vs f", model="C = A f^B")

