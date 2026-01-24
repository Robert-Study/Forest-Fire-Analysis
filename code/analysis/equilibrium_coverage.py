import numpy as np
import pandas as pd


def load_sweep_csv(filename):
    """Load a sweep CSV written by analysis.parameter_sweep.sweep."""
    df = pd.read_csv(filename)
    # tolerate 'r2' vs 'r2_tree' naming differences
    if "r2_tree" not in df.columns and "r2" in df.columns:
        df = df.rename(columns={"r2": "r2_tree"})
    return df


def extract_equilibrium_coverage(df):
    """
    In the notebook, the equilibrium tree coverage is the fitted offset 'C'
    from the decaying-sine model.
    Returns (coverage, coverage_err).
    """
    return df["C"].to_numpy(float), df["C_err"].to_numpy(float)


def extract_growth_probability(df):
    return df["p"].to_numpy(float)


def extract_lightning_probability(df):
    return df["f"].to_numpy(float)
