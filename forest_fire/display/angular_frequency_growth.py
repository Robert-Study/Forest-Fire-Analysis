import pandas as pd

from forest_fire.analysis.angular_frequency import extract_angular_frequency, extract_growth_probability
from forest_fire.analysis.exponential_decay import loglog


def plot_angular_frequency_vs_p(mask, sweep_csv, x_label="Probability of Growth (p)", y_label="Angular Frequency (w)"):
    df = pd.read_csv(sweep_csv)
    p = extract_growth_probability(df)
    w, w_err = extract_angular_frequency(df)

    return loglog(mask, p, w, w_err, x_label, y_label, "Angular frequency vs p", model="w = A p^B")

