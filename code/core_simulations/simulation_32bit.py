import numpy as np
from tqdm import tqdm

from code.constants import EMPTY, TREE, FIRE
from code.analysis.critical_time import critical_regime_start, simulation_length
from code.analysis.cluster_area_perimeter import analyse_clusters
from code.analysis.sampling_time import sample_when_fire_ends


def update_forest_32(board, p, f):
    """32-bit probability version of update_forest (uint32 random grid)."""
    new_board = board.copy()
    Empty_idx = (board == EMPTY)
    Tree_idx = (board == TREE)
    Burning_idx = (board == FIRE)

    random_grid = np.random.randint(0, 4294967296, size=board.shape, dtype=np.uint32)

    new_board[Burning_idx] = EMPTY

    up    = np.roll(Burning_idx, -1, axis=0)
    down  = np.roll(Burning_idx,  1, axis=0)
    left  = np.roll(Burning_idx, -1, axis=1)
    right = np.roll(Burning_idx,  1, axis=1)
    fire_neighbours = up | down | left | right
    fire_spread_idx = Tree_idx & fire_neighbours
    new_board[fire_spread_idx] = FIRE

    lightning_idx = Tree_idx & (random_grid < int(f * 4294967296))
    new_board[lightning_idx] = FIRE

    grow_idx = Empty_idx & (random_grid < int(p * 4294967296))
    new_board[grow_idx] = TREE

    return new_board


def run_simulation_clusters(L, p, f):
    """Run 32-bit simulation and sample tree clusters after each fire ends in the critical regime."""
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
