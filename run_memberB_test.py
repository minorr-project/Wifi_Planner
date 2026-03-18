import numpy as np
import json

from signal_math import calibrate_cell_size, coverage_metrics
from ga_core import run_ga

from visualizationD2_integrated import plot_solution  # (we'll create this below)


def load_grid(grid_path="grid.npy", meta_path="grid_meta.json"):
    grid = np.load(grid_path)
    meta = None
    try:
        with open(meta_path, "r") as f:
            meta = json.load(f)
    except FileNotFoundError:
        pass
    return grid, meta


def main():
    # 1) Load grid produced by your DXF pipeline
    grid, meta = load_grid()

    # 2) Optional downsample for speed (you already did this idea)
    grid = grid[::8, ::8]

    # 3) Calibrate signal model to avoid “huge building” scaling issues
    cell_m = calibrate_cell_size(grid)
    print("CELL_SIZE_METERS =", cell_m)

    # 4) Run GA
    result = run_ga(
        grid,
        num_routers=2,
        population_size=20,
        generations=30,
        seed=42,
    )

    best = result["best_routers"]
    print("\nBest routers:", best)
    print("Best fitness:", result["best_fitness"])

    cov, avg = coverage_metrics(best, grid)
    print(f"Coverage%: {cov:.1f}, Avg signal (dBm): {avg:.1f}")

    # 5) Visualize (Member D integrated here)
    plot_solution(grid, best, out_path="outputs/images/final_solution.png")


if __name__ == "__main__":
    main()
