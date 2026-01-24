import pandas as pd

from code.analysis.exponential_decay import loglog
from code.analysis.equilibrium_coverage import extract_lightning_probability


def plot_decay_rate_vs_f(mask, sweep_csv, x_label="Probability of Lightning (f)", y_label="Decay rate (d)"):
    """Decay rate is 'd' from the decaying-sine fit."""
    df = pd.read_csv(sweep_csv)
    f = extract_lightning_probability(df)
    d = df["d"].to_numpy(float)
    d_err = df["d_err"].to_numpy(float)

    return loglog(mask, f, d, d_err, x_label, y_label, "Exponential decay vs f", model="d = A f^B")
