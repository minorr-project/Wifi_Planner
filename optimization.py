"""
WiFi Router Placement Optimization module.
Supports three strategies: genetic algorithm (ga), random, and uniform.
"""

import os
import math
import json
import random as _random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from member_A_genetic_Algorithm_core.ga_core import run_ga, get_free_cells
from member_B_signal_simulation_engine.signal_math import (
    coverage_metrics,
    best_signal,
    S_threshold,
)

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(REPO_ROOT, "outputs")


def load_grids():
    grid_opt = np.load(os.path.join(REPO_ROOT, "grid.npy")).astype(np.uint8)
    grid_disp = np.load(os.path.join(REPO_ROOT, "grid_display.npy")).astype(np.uint8)
    meta = {}
    meta_path = os.path.join(REPO_ROOT, "grid_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
    return grid_opt, grid_disp, meta


def _scale_routers(routers_opt, grid_opt, grid_disp):
    h_opt, w_opt = grid_opt.shape
    h_disp, w_disp = grid_disp.shape
    sx = max(1, round(w_disp / w_opt))
    sy = max(1, round(h_disp / h_opt))
    return [(min(int(x * sx), w_disp - 1), min(int(y * sy), h_disp - 1))
            for x, y in routers_opt]


def ga_placement(grid, num_routers, generations=40, population_size=30, seed=42):
    result = run_ga(grid, num_routers=num_routers,
                    generations=generations,
                    population_size=population_size,
                    seed=seed)
    return result["best_routers"]


def random_placement(grid, num_routers, seed=None):
    rng = _random.Random(seed)
    free = get_free_cells(grid)
    if num_routers > len(free):
        raise ValueError("num_routers exceeds number of free cells")
    return rng.sample(free, num_routers)


def uniform_placement(grid, num_routers):
    """
    Divide free space into a grid of num_routers zones and pick the
    free cell nearest the centre of each zone.
    """
    free = get_free_cells(grid)
    if not free:
        raise ValueError("No free cells available")
    if num_routers > len(free):
        raise ValueError("num_routers exceeds number of free cells")

    xs = [x for x, y in free]
    ys = [y for x, y in free]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    cols = math.ceil(math.sqrt(num_routers))
    rows = math.ceil(num_routers / cols)

    free_set = set(free)
    routers = []
    zone_w = (max_x - min_x + 1) / cols
    zone_h = (max_y - min_y + 1) / rows

    for r in range(rows):
        for c in range(cols):
            if len(routers) >= num_routers:
                break
            cx = min_x + (c + 0.5) * zone_w
            cy = min_y + (r + 0.5) * zone_h
            best = min(free, key=lambda p: (p[0] - cx) ** 2 + (p[1] - cy) ** 2)
            if best not in routers:
                routers.append(best)
            else:
                candidates = sorted(
                    free,
                    key=lambda p: (p[0] - cx) ** 2 + (p[1] - cy) ** 2
                )
                for cand in candidates:
                    if cand not in routers:
                        routers.append(cand)
                        break

    return routers[:num_routers]


def make_heatmap(grid, routers):
    h, w = grid.shape
    heat = np.full((h, w), np.nan, dtype=float)
    for y in range(h):
        for x in range(w):
            if grid[y, x] == 0:
                heat[y, x] = best_signal((x, y), routers, grid)
    return heat


STRATEGY_LABELS = {
    "ga": "Genetic Algorithm",
    "random": "Random",
    "uniform": "Uniform Grid",
}


def visualize_and_save(grid_disp, routers_disp, strategy, output_dir=OUTPUT_DIR):
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    heat = make_heatmap(grid_disp, routers_disp)
    heat = np.clip(heat, -95, -30)

    label = STRATEGY_LABELS.get(strategy, strategy)

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    ax0 = axes[0]
    ax0.imshow(grid_disp, cmap="gray_r", origin="lower", interpolation="nearest")
    for i, (x, y) in enumerate(routers_disp, start=1):
        ax0.plot(x, y, "ro", markersize=8, markeredgecolor="black")
        ax0.text(x, y + 3, f"R{i}", ha="center", va="bottom",
                 bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85))
    ax0.set_title(f"{label} – Router Placement")
    ax0.set_xticks([])
    ax0.set_yticks([])

    ax1 = axes[1]
    im = ax1.imshow(heat, origin="lower", cmap="viridis", interpolation="nearest")
    wall_mask = np.ma.masked_where(grid_disp == 0, grid_disp)
    ax1.imshow(wall_mask, origin="lower", cmap="gray_r", alpha=0.9, interpolation="nearest")
    for i, (x, y) in enumerate(routers_disp, start=1):
        ax1.plot(x, y, "wo", markersize=6, markeredgecolor="black")
        ax1.text(x, y + 3, f"R{i}", ha="center", va="bottom",
                 bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85))
    ax1.set_title(f"Signal Heatmap (threshold = {S_threshold} dBm)")
    ax1.set_xticks([])
    ax1.set_yticks([])

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
    cbar.set_label("Signal (dBm)")

    plt.tight_layout()
    save_path = os.path.join(images_dir, f"{strategy}_placement.png")
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return save_path


def run_optimization(strategy, num_routers, seed=42):
    """
    Full optimization pipeline for a given strategy.
    Returns dict with routers, coverage_percent, average_signal_dBm, image_path.
    """
    grid_opt, grid_disp, meta = load_grids()

    if strategy == "ga":
        routers_opt = ga_placement(grid_opt, num_routers, seed=seed)
    elif strategy == "random":
        routers_opt = random_placement(grid_opt, num_routers, seed=seed)
    elif strategy == "uniform":
        routers_opt = uniform_placement(grid_opt, num_routers)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    coverage_pct, avg_signal = coverage_metrics(routers_opt, grid_opt)
    routers_disp = _scale_routers(routers_opt, grid_opt, grid_disp)
    image_path = visualize_and_save(grid_disp, routers_disp, strategy)

    result = {
        "strategy": strategy,
        "num_routers": num_routers,
        "routers_optimization_grid": [{"x": int(x), "y": int(y)} for x, y in routers_opt],
        "routers_display_grid": [{"x": int(x), "y": int(y)} for x, y in routers_disp],
        "coverage_percent": float(coverage_pct),
        "average_signal_dBm": float(avg_signal),
        "image_path": image_path,
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results_path = os.path.join(OUTPUT_DIR, f"{strategy}_results.json")
    with open(results_path, "w") as f:
        json.dump(result, f, indent=2)

    return result
