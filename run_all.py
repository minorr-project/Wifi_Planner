# -*- coding: utf-8 -*-

import os
import sys
import json
import random
import numpy as np

from dxf_to_grid_render import dxf_to_grid_render, preview_grid


# ---- Force GUI backend (optional but helps on Windows) ----
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

# ---- Ensure project root is importable ----
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ---- Imports from your packages ----
from member_A_genetic_Algorithm_core.ga_core import run_ga
from member_B_signal_simulation_engine.signal_math import (
    calibrate_cell_size,
    coverage_metrics,
    best_signal,
    S_threshold,
)

# =========================
# Utility functions (no utils.py needed)
# =========================
def load_grid(grid_path="grid.npy", meta_path="grid_meta.json"):
    if not os.path.exists(grid_path):
        raise FileNotFoundError(
            f"{grid_path} not found. Run dxf_to_grid.py OR rasterize_to_grid.py first."
        )
    grid = np.load(grid_path)

    meta = None
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)

    return grid, meta


def preprocess(grid, step=8):
    # downsample for speed
    return grid[::step, ::step].astype(np.uint8)


def get_free_cells(grid):
    ys, xs = np.where(grid == 0)
    return list(zip(xs.tolist(), ys.tolist()))  # (x, y)


def run_random_baseline(grid, num_routers=2, seed=42):
    rng = random.Random(seed)
    candidates = get_free_cells(grid)
    if len(candidates) < num_routers:
        raise ValueError("Not enough free cells for router placement.")
    return rng.sample(candidates, num_routers)


def run_uniform_baseline(grid, num_routers=2):
    H, W = grid.shape
    candidates = [
        (W // 4, H // 4),
        (3 * W // 4, 3 * H // 4),
        (W // 4, 3 * H // 4),
        (3 * W // 4, H // 4),
        (W // 2, H // 2),
    ]

    routers = []
    for (x, y) in candidates:
        if 0 <= x < W and 0 <= y < H and grid[y, x] == 0:
            routers.append((x, y))
        if len(routers) == num_routers:
            return routers

    # fallback fill from free cells
    free_cells = get_free_cells(grid)
    for c in free_cells:
        if c not in routers:
            routers.append(c)
        if len(routers) == num_routers:
            return routers

    return routers[:num_routers]


def print_metrics(name, routers, grid):
    cov, avg = coverage_metrics(routers, grid)
    print(f"{name:>10} | routers={routers} | coverage={cov:.1f}% | avg={avg:.1f} dBm")


# =========================
# Visualization (3 methods in one figure)
# =========================
def make_dbm_heatmap(grid, routers):
    H, W = grid.shape
    heat = np.full((H, W), np.nan, dtype=float)
    for y in range(H):
        for x in range(W):
            if grid[y, x] == 1:
                continue
            heat[y, x] = best_signal((x, y), routers, grid)
    return heat


def visualize_all(grid, methods, out_path="outputs/images/compare.png"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    names = list(methods.keys())
    n = len(names)

    fig, axes = plt.subplots(2, n, figsize=(5 * n, 8))
    if n == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    last_im = None

    for i, name in enumerate(names):
        routers = methods[name]

        # top: floorplan + routers
        ax_top = axes[0, i]
        ax_top.imshow(grid, cmap="gray_r", origin="lower")
        for j, (x, y) in enumerate(routers):
            ax_top.plot(x, y, "ro", markersize=8, markeredgecolor="black")
            ax_top.text(
                x, y + 1, f"R{j+1}",
                ha="center", va="bottom",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8)
            )
        ax_top.set_title(f"{name} (Placement)")
        ax_top.set_xticks([])
        ax_top.set_yticks([])

        # bottom: signal heatmap (dBm)
        ax_bot = axes[1, i]
        heat = make_dbm_heatmap(grid, routers)
        heat = np.clip(heat, -95, -30)
        last_im = ax_bot.imshow(heat, origin="lower")
        for (x, y) in routers:
            ax_bot.plot(x, y, "ko", markersize=6, markeredgecolor="white")
        ax_bot.set_title(f"{name} (Signal dBm)\nThreshold = {S_threshold} dBm")
        ax_bot.set_xticks([])
        ax_bot.set_yticks([])

    if last_im is not None:
        cbar = fig.colorbar(last_im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
        cbar.set_label("Signal (dBm)")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"\nSaved plot: {out_path}")
    plt.show()


# =========================
# MAIN RUN
# =========================
def main():
    print(">>> run_all.py STARTED <<<")

    grid, meta = load_grid()
    print("Loaded grid shape:", grid.shape)

    grid = preprocess(grid, step=8)
    print("After preprocess shape:", grid.shape)

    calibrate_cell_size(grid)

    NUM_ROUTERS = 2
    GENERATIONS = 20
    SEED = 42

    # baselines
    random_routers = run_random_baseline(grid, num_routers=NUM_ROUTERS, seed=SEED)
    uniform_routers = run_uniform_baseline(grid, num_routers=NUM_ROUTERS)

    # GA
    print("\nRunning GA...")
    ga_result = run_ga(
        grid,
        num_routers=NUM_ROUTERS,
        generations=GENERATIONS,
        seed=SEED
    )
    ga_routers = ga_result["best_routers"]

    print("\n=== METRICS ===")
    print_metrics("Random", random_routers, grid)
    print_metrics("Uniform", uniform_routers, grid)
    print_metrics("GA", ga_routers, grid)
    print("Best GA fitness:", ga_result["best_fitness"])

    visualize_all(
        grid,
        {
            "Random": random_routers,
            "Uniform": uniform_routers,
            "GA Optimized": ga_routers,
        },
        out_path="outputs/images/compare.png"
    )


if __name__ == "__main__":
    main()
