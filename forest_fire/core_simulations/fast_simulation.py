"""Synchronous forest-fire dynamics on a periodic square lattice."""
import csv
from datetime import datetime
from pathlib import Path

import numpy as np

from forest_fire.constants import EMPTY, TREE, FIRE, cmap, fps


def validate_probabilities(p, f, probability_bits=32):
    if probability_bits not in (16, 32):
        raise ValueError("probability_bits must be 16 or 32.")
    for name, value in [("p", p), ("f", f)]:
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{name} must be finite and lie in [0, 1].")
        if 0 < value < 2.0**(-probability_bits):
            raise ValueError(f"{name} is below the {probability_bits}-bit probability resolution.")


def random_grid_for(board, rng=None, probability_bits=32):
    dtype = np.uint16 if probability_bits == 16 else np.uint32
    if rng is None:
        return np.random.randint(0, 2**probability_bits, size=board.shape, dtype=dtype)
    return rng.integers(0, 2**probability_bits, size=board.shape, dtype=dtype)


def update_forest(board, p, f, rng=None, probability_bits=32):
    """Advance one synchronous step; never modify the input board.

    Four-neighbour fire spread wraps at both edges. Cells that burn this step
    cannot regrow until the following step. NumPy Boolean arrays use one BYTE
    per cell. Integer random draws represent floor(probability*2**bits)/2**bits.
    """
    validate_probabilities(p, f, probability_bits)
    if board.ndim != 2:
        raise ValueError("The board must be two-dimensional.")
    new_board = board.copy()
    empty, tree, burning = board == EMPTY, board == TREE, board == FIRE
    random_grid = random_grid_for(board, rng, probability_bits)
    neighbours = np.zeros_like(burning)
    for axis in (0, 1):
        for shift in (-1, 1):
            neighbours |= np.roll(burning, shift, axis=axis)
    scale = 2**probability_bits
    new_board[burning] = EMPTY
    new_board[tree & (neighbours | (random_grid < int(f * scale)))] = FIRE
    new_board[empty & (random_grid < int(p * scale))] = TREE
    return new_board


def run_simulation(L, frames, p, f, gif=False, save_2_file=False, *,
                   seed=None, probability_bits=32, output_dir=".", progress=False):
    """Return population counts; optionally stream CSV rows and GIF frames.

    A local Generator isolates simulation randomness from plotting and callers.
    The seed, NumPy version and probability precision define reproducibility.
    """
    if not isinstance(L, (int, np.integer)) or L < 1:
        raise ValueError("L must be a positive integer.")
    if not isinstance(frames, (int, np.integer)) or frames < 1:
        raise ValueError("frames must be a positive integer.")
    validate_probabilities(p, f, probability_bits)
    output = Path(output_dir)
    if gif or save_2_file:
        output.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    writer = None
    csv_file = None
    tree_counts, burn_counts = [], []
    rng = np.random.default_rng(seed)
    board = np.zeros((L, L), dtype=np.uint8)
    try:
        if gif:
            import imageio.v2 as imageio
            writer = imageio.get_writer(output / f"forest_fire_{stamp}.gif", mode="I", duration=1000/fps, loop=0)
        if save_2_file:
            csv_file = (output / f"population_{stamp}.csv").open("w", newline="")
            rows = csv.writer(csv_file)
            rows.writerow(["frame", "tree_counts", "burn_counts"])
        steps = range(frames)
        if progress:
            from tqdm import tqdm
            steps = tqdm(steps)
        for t in steps:
            board = update_forest(board, p, f, rng, probability_bits)
            trees, burning = int(np.count_nonzero(board == TREE)), int(np.count_nonzero(board == FIRE))
            tree_counts.append(trees)
            burn_counts.append(burning)
            if csv_file is not None:
                rows.writerow([t + 1, trees, burning])
                if (t + 1) % 100 == 0:
                    csv_file.flush()
            if writer is not None:
                writer.append_data(cmap[board])
    finally:
        if writer is not None:
            writer.close()
        if csv_file is not None:
            csv_file.close()
    return tree_counts, burn_counts
