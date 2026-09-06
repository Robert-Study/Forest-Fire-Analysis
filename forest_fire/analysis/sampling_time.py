"""Sampling policies used by the cluster-analysis simulations."""


def sample_when_fire_ends(previous_fire_cells, current_fire_cells):
    """Return True on the first frame after all active fire has extinguished."""
    return previous_fire_cells > 0 and current_fire_cells == 0

