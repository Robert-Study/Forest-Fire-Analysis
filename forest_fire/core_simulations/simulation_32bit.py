import numpy as np

from forest_fire.constants import EMPTY, TREE, FIRE
from forest_fire.analysis.critical_time import critical_regime_start, simulation_length
from forest_fire.analysis.cluster_area_perimeter import analyse_clusters
from forest_fire.analysis.sampling_time import sample_when_fire_ends


def update_forest_32(board, p, f, rng=None):
    """Compatibility wrapper for the shared 32-bit simulation engine."""
    from forest_fire.core_simulations.fast_simulation import update_forest
    return update_forest(board, p, f, rng=rng, probability_bits=32)


def run_simulation_clusters(L, p, f):
    """Run 32-bit simulation and sample tree clusters after each fire ends in the critical regime."""
    from tqdm import tqdm
    critical = critical_regime_start(f)
    total_frames = simulation_length(f)

    board = np.zeros((L, L), dtype=np.uint8)
    cluster_data = []
    prev_fire_cells = 0

    for t in tqdm(range(total_frames)):
        board = update_forest_32(board, p, f)

        fire_cells = int(np.sum(board == FIRE))

        if t >= critical and sample_when_fire_ends(prev_fire_cells, fire_cells):
            clusters = analyse_clusters(board, L)
            cluster_data.extend(clusters)

        prev_fire_cells = fire_cells

    return cluster_data

