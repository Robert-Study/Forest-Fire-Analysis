"""Run a seeded, modest-size simulation and save a plot, counts and metadata."""
import argparse
import csv
import json
from pathlib import Path
import platform
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from forest_fire.constants import TREE, FIRE
from forest_fire.core_simulations.fast_simulation import update_forest, validate_probabilities


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--frames", type=int, default=2000)
    parser.add_argument("--p", type=float, default=0.0153)
    parser.add_argument("--f", type=float, default=0.000153)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--probability-bits", type=int, choices=[16, 32], default=32)
    parser.add_argument("--output", type=Path, default=Path("outputs/demo"))
    args = parser.parse_args()
    if args.size < 2 or args.frames < 1:
        parser.error("--size must be at least 2 and --frames must be positive")
    validate_probabilities(args.p, args.f, args.probability_bits)
    args.output.mkdir(parents=True, exist_ok=True)
    board = np.zeros((args.size, args.size), dtype=np.uint8)
    rng = np.random.default_rng(args.seed)
    counts = np.empty((args.frames, 2), dtype=np.int64)
    started = time.perf_counter()
    with (args.output / "population.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["frame", "tree_count", "burn_count"])
        for frame in range(args.frames):
            board = update_forest(board, args.p, args.f, rng, args.probability_bits)
            counts[frame] = np.count_nonzero(board == TREE), np.count_nonzero(board == FIRE)
            writer.writerow([frame + 1, *counts[frame]])
    seconds = time.perf_counter() - started
    t = np.arange(1, args.frames + 1)
    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True, layout="constrained")
    for ax, column, colour, label in zip(axes, [0, 1], ["#16745b", "#c45032"], ["Tree coverage (%)", "Burning cells (%)"]):
        ax.plot(t, 100 * counts[:, column] / args.size**2, color=colour, linewidth=1)
        ax.set_ylabel(label)
        ax.grid(alpha=.2)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_title(f"Forest Fire Simulation | {args.size} × {args.size} cells | seed {args.seed}", loc="left", fontweight="bold")
    axes[1].set_xlabel("Simulation step")
    fig.savefig(args.output / "population.png", dpi=150)
    plt.close(fig)
    scale = 2**args.probability_bits
    metadata = {
        "seed": args.seed, "size": args.size, "frames": args.frames,
        "p_requested": args.p, "f_requested": args.f,
        "p_effective": int(args.p * scale)/scale, "f_effective": int(args.f * scale)/scale,
        "probability_bits": args.probability_bits, "python": platform.python_version(),
        "numpy": np.__version__, "platform": platform.platform(),
        "cell_time_updates": args.size**2 * args.frames,
        "simulation_and_csv_seconds": seconds,
        "cell_time_updates_per_second": args.size**2 * args.frames / seconds,
        "timing_scope": "Simulation, population counting and CSV output; excludes plotting."
    }
    (args.output / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
