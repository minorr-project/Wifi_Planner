#!/usr/bin/env python3
"""
WiFi Router Placement Optimization - Full Pipeline Integration

Uses:
- grid.npy           -> optimization grid (smaller)
- grid_display.npy   -> display grid (higher resolution)

Outputs:
- outputs/optimization_results.json
- outputs/images/optimized_placement.png
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt

from member_A_genetic_Algorithm_core.ga_core import run_ga
from member_B_signal_simulation_engine.signal_math import (
    coverage_metrics,
    best_signal,
    S_threshold,
)


REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


def load_grid():
    """Load optimization grid and metadata."""
    grid_path = os.path.join(REPO_ROOT, "grid.npy")
    meta_path = os.path.join(REPO_ROOT, "grid_meta.json")

    if not os.path.exists(grid_path):
        raise FileNotFoundError(
            "grid.npy not found.\n"
            "Run:\n"
            "  python dxf_pipeline_general.py house.dxf ."
        )

    grid = np.load(grid_path).astype(np.uint8)

    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

    return grid, meta


def load_display_grid():
    """Load high-resolution display grid."""
    grid_path = os.path.join(REPO_ROOT, "grid_display.npy")

    if not os.path.exists(grid_path):
        raise FileNotFoundError(
            "grid_display.npy not found.\n"
            "Run:\n"
            "  python dxf_pipeline_general.py house.dxf ."
        )

    return np.load(grid_path).astype(np.uint8)


def compute_scale_factor(grid_opt, grid_display):
    """
    Estimate integer scale factor between optimization grid and display grid.
    """
    h_opt, w_opt = grid_opt.shape
    h_disp, w_disp = grid_display.shape

    scale_x = max(1, round(w_disp / w_opt))
    scale_y = max(1, round(h_disp / h_opt))

    # use separate scales if slightly different
    return scale_x, scale_y


def scale_routers_to_display(routers_opt, grid_display, scale_x, scale_y):
    """
    Convert router coordinates from optimization grid to display grid.
    Routers are stored as (x, y).
    """
    h_disp, w_disp = grid_display.shape
    routers_disp = []

    for x, y in routers_opt:
        xd = min(int(x * scale_x), w_disp - 1)
        yd = min(int(y * scale_y), h_disp - 1)
        routers_disp.append((xd, yd))

    return routers_disp


def make_dbm_heatmap(grid, routers):
    """
    Compute signal heatmap in dBm for free cells.
    Walls are left as NaN.
    """
    h, w = grid.shape
    heat = np.full((h, w), np.nan, dtype=float)

    for y in range(h):
        for x in range(w):
            if grid[y, x] == 1:
                continue
            heat[y, x] = best_signal((x, y), routers, grid)

    return heat


def visualize_results(grid_display, routers_display, output_dir="outputs"):
    """
    Save a clean placement + heatmap visualization.
    """
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    heat = make_dbm_heatmap(grid_display, routers_display)
    heat = np.clip(heat, -95, -30)

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    # Left: placement
    ax0 = axes[0]
    ax0.imshow(grid_display, cmap="gray_r", origin="lower", interpolation="nearest")
    for i, (x, y) in enumerate(routers_display, start=1):
        ax0.plot(x, y, "ro", markersize=8, markeredgecolor="black")
        ax0.text(
            x,
            y + 3,
            f"R{i}",
            ha="center",
            va="bottom",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85),
        )
    ax0.set_title("Optimized Router Placement")
    ax0.set_xticks([])
    ax0.set_yticks([])

    # Right: heatmap with wall overlay
    ax1 = axes[1]
    im = ax1.imshow(heat, origin="lower", cmap="viridis", interpolation="nearest")
    wall_mask = np.ma.masked_where(grid_display == 0, grid_display)
    ax1.imshow(wall_mask, origin="lower", cmap="gray_r", alpha=0.9, interpolation="nearest")

    for i, (x, y) in enumerate(routers_display, start=1):
        ax1.plot(x, y, "wo", markersize=6, markeredgecolor="black")
        ax1.text(
            x,
            y + 3,
            f"R{i}",
            ha="center",
            va="bottom",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85),
        )

    ax1.set_title(f"Signal Heatmap (Threshold = {S_threshold} dBm)")
    ax1.set_xticks([])
    ax1.set_yticks([])

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
    cbar.set_label("Signal (dBm)")

    plt.tight_layout()

    save_path = os.path.join(images_dir, "optimized_placement.png")
    plt.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    return save_path


def save_results(routers_opt, routers_display, coverage_pct, avg_signal, meta, output_dir="outputs"):
    """Save optimization results to JSON."""
    os.makedirs(output_dir, exist_ok=True)

    results = {
        "routers_optimization_grid": [{"x": int(x), "y": int(y)} for x, y in routers_opt],
        "routers_display_grid": [{"x": int(x), "y": int(y)} for x, y in routers_display],
        "coverage_percent": float(coverage_pct),
        "average_signal_dBm": float(avg_signal),
        "grid_meta": meta,
    }

    results_path = os.path.join(output_dir, "optimization_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results_path


def main():
    print("=" * 60)
    print(" WiFi Router Placement Optimization - Full Pipeline")
    print("=" * 60)

    # Configuration
    num_routers = 2
    generations = 40
    population_size = 30
    seed = 42
    output_dir = "outputs"

    # Step 1: load optimization grid
    print("\n[1/5] Loading optimization grid...")
    grid_opt, meta = load_grid()
    print(f"   • Optimization grid shape: {grid_opt.shape}")
    print(f"   • Wall cells: {int((grid_opt == 1).sum())}")
    print(f"   • Free cells: {int((grid_opt == 0).sum())}")

    # Step 2: run GA
    print("\n[2/5] Running GA...")
    ga_result = run_ga(
        grid_opt,
        num_routers=num_routers,
        generations=generations,
        population_size=population_size,
        seed=seed,
    )
    best_routers_opt = ga_result["best_routers"]
    best_fitness = ga_result["best_fitness"]

    print(f"   • Best routers (optimization grid): {best_routers_opt}")
    print(f"   • Best fitness: {best_fitness:.2f}")

    # Step 3: calculate signal metrics on optimization grid
    print("\n[3/5] Calculating coverage metrics...")
    coverage_pct, avg_signal = coverage_metrics(best_routers_opt, grid_opt)
    print(f"   • Coverage: {coverage_pct:.2f}%")
    print(f"   • Average signal: {avg_signal:.2f} dBm")

    # Step 4: load display grid and scale routers
    print("\n[4/5] Generating visualization...")
    grid_display = load_display_grid()
    scale_x, scale_y = compute_scale_factor(grid_opt, grid_display)
    best_routers_display = scale_routers_to_display(
        best_routers_opt,
        grid_display,
        scale_x,
        scale_y,
    )

    print(f"   • Display grid shape: {grid_display.shape}")
    print(f"   • Scale factor: x={scale_x}, y={scale_y}")
    print(f"   • Routers (display grid): {best_routers_display}")

    image_path = visualize_results(grid_display, best_routers_display, output_dir=output_dir)
    print(f"   • Saved image: {image_path}")

    # Step 5: save JSON results
    print("\n[5/5] Saving results...")
    results_path = save_results(
        best_routers_opt,
        best_routers_display,
        coverage_pct,
        avg_signal,
        meta,
        output_dir=output_dir,
    )
    print(f"   • Saved JSON: {results_path}")

    print("\n" + "=" * 60)
    print(" OPTIMIZATION COMPLETE")
    print("=" * 60)
    print(f" Routers (opt grid): {best_routers_opt}")
    print(f" Routers (display grid): {best_routers_display}")
    print(f" Coverage: {coverage_pct:.2f}%")
    print(f" Average Signal: {avg_signal:.2f} dBm")
    print(f" Best Fitness: {best_fitness:.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()