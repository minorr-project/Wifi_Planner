# -*- coding: utf-8 -*-
"""
WiFi Router Placement Optimization - Visualization Entry Point
Member D: Integrates GA results with full signal visualization.
"""

import os
import sys
import json
import random
import numpy as np

# Ensure project root is importable
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import matplotlib
matplotlib.use("Agg")  # headless - saves files without needing a display
import matplotlib.pyplot as plt

from member_A_genetic_Algorithm_core.ga_core import run_ga
from member_B_signal_simulation_engine.signal_math import (
    calibrate_cell_size,
    coverage_metrics,
    S_threshold,
)
from member_D_visualization.visualization_integrated import visualize_all


def load_grid(grid_path="grid.npy", meta_path="grid_meta.json"):
    if not os.path.exists(grid_path):
        raise FileNotFoundError(
            f"{grid_path} not found. Run dxf_pipeline_general.py first."
        )
    grid = np.load(grid_path).astype(np.uint8)
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)
    return grid, meta


def preprocess(grid, step=8):
    """Downsample for faster GA evaluation."""
    return grid[::step, ::step].astype(np.uint8)


def get_free_cells(grid):
    ys, xs = np.where(grid == 0)
    return list(zip(xs.tolist(), ys.tolist()))


def run_random_baseline(grid, num_routers=2, seed=42):
    rng = random.Random(seed)
    return rng.sample(get_free_cells(grid), num_routers)


def run_uniform_baseline(grid, num_routers=2):
    H, W = grid.shape
    candidates = [
        (W // 4, H // 4), (3 * W // 4, 3 * H // 4),
        (W // 4, 3 * H // 4), (3 * W // 4, H // 4), (W // 2, H // 2),
    ]
    routers = []
    for (x, y) in candidates:
        if 0 <= x < W and 0 <= y < H and grid[y, x] == 0:
            routers.append((x, y))
        if len(routers) == num_routers:
            return routers
    for c in get_free_cells(grid):
        if c not in routers:
            routers.append(c)
        if len(routers) == num_routers:
            return routers
    return routers[:num_routers]


def main():
    print("=" * 60)
    print(" WiFi Router Placement - Visualization Module (Member D)")
    print("=" * 60)

    # Load and downsample grid
    grid_full, meta = load_grid(
        grid_path=os.path.join(PROJECT_ROOT, "grid.npy"),
        meta_path=os.path.join(PROJECT_ROOT, "grid_meta.json"),
    )
    print(f"\nFull grid shape: {grid_full.shape}")

    grid = preprocess(grid_full, step=8)
    print(f"Downsampled grid shape: {grid.shape}")

    calibrate_cell_size(grid)

    NUM_ROUTERS = 2
    GENERATIONS = 20
    SEED = 42

    # Baselines
    random_routers = run_random_baseline(grid, num_routers=NUM_ROUTERS, seed=SEED)
    uniform_routers = run_uniform_baseline(grid, num_routers=NUM_ROUTERS)

    # GA
    print("\nRunning GA...")
    ga_result = run_ga(grid, num_routers=NUM_ROUTERS, generations=GENERATIONS, seed=SEED)
    ga_routers = ga_result["best_routers"]

    # Metrics
    print("\n=== METRICS ===")
    for name, routers in [("Random", random_routers), ("Uniform", uniform_routers), ("GA", ga_routers)]:
        cov, avg = coverage_metrics(routers, grid)
        print(f"  {name:>8} | routers={routers} | coverage={cov:.1f}% | avg_signal={avg:.1f} dBm")
    print(f"  Best GA fitness: {ga_result['best_fitness']:.2f}")

    # Visualization
    out_path = os.path.join(PROJECT_ROOT, "outputs", "images", "compare_memberD.png")
    visualize_all(
        grid,
        {
            "Random": random_routers,
            "Uniform": uniform_routers,
            "GA Optimized": ga_routers,
        },
        out_path=out_path,
    )

    print("\n" + "=" * 60)
    print(" DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()