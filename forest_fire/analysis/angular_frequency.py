import numpy as np
import pandas as pd


def load_sweep_csv(filename):
    df = pd.read_csv(filename)
    if "r2_tree" not in df.columns and "r2" in df.columns:
        df = df.rename(columns={"r2": "r2_tree"})
    return df


def extract_angular_frequency(df):
    """Angular frequency is the fitted decaying-sine parameter 'w'."""
    return df["w"].to_numpy(float), df["w_err"].to_numpy(float)


def extract_growth_probability(df):
    return df["p"].to_numpy(float)

