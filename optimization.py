"""
WiFi Router Placement Optimization module.
Supports three strategies: genetic algorithm (ga), random, and uniform.
Uses a fast vectorized signal model for speed.
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
from member_B_signal_simulation_engine.signal_math import S_threshold

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(REPO_ROOT, "outputs")

# Signal constants (mirror signal_math.py)
_S0 = -30.0
_D_LOSS_K = 2.0
_WALL_PENALTY = 8.0


# ── Fast vectorized signal helpers ─────────────────────────────────────────

def _cell_size_m(grid):
    """Estimate metres per cell from grid width (15 m reference building)."""
    return 15.0 / grid.shape[1]


def _vectorized_heatmap(grid, routers):
    """
    Fast numpy heatmap using distance-based signal (no wall-crossing loop).
    Returns a (H, W) array, NaN over walls.
    """
    h, w = grid.shape
    cs = _cell_size_m(grid)
    y_idx, x_idx = np.mgrid[0:h, 0:w]
    heat = np.full((h, w), -300.0)
    for rx, ry in routers:
        dist_m = np.sqrt((x_idx - rx) ** 2 + (y_idx - ry) ** 2) * cs
        sig = _S0 - _D_LOSS_K * dist_m
        heat = np.maximum(heat, sig)
    heat[grid != 0] = np.nan
    return heat


def _vectorized_coverage(routers, grid):
    """
    Fast numpy coverage metrics using distance-based signal (no wall loops).
    Returns (coverage_pct, avg_signal_dBm).
    """
    h, w = grid.shape
    cs = _cell_size_m(grid)
    y_idx, x_idx = np.mgrid[0:h, 0:w]
    heat = np.full((h, w), -300.0)
    for rx, ry in routers:
        dist_m = np.sqrt((x_idx - rx) ** 2 + (y_idx - ry) ** 2) * cs
        sig = _S0 - _D_LOSS_K * dist_m
        heat = np.maximum(heat, sig)
    free = heat[grid == 0]
    if len(free) == 0:
        return 0.0, 0.0
    return float((free >= S_threshold).mean() * 100), float(free.mean())


def _vectorized_fitness(routers, grid):
    cov, _ = _vectorized_coverage(routers, grid)
    return cov


# ── Grid helpers ────────────────────────────────────────────────────────────

def _max_pool(grid, factor):
    """Downsample grid by factor using max-pooling (wall wins in any block)."""
    h, w = grid.shape
    h2, w2 = h // factor, w // factor
    cropped = grid[:h2 * factor, :w2 * factor]
    return cropped.reshape(h2, factor, w2, factor).max(axis=(1, 3))


def load_grids():
    grid_opt = np.load(os.path.join(REPO_ROOT, "grid.npy")).astype(np.uint8)
    grid_disp = np.load(os.path.join(REPO_ROOT, "grid_display.npy")).astype(np.uint8)
    meta = {}
    meta_path = os.path.join(REPO_ROOT, "grid_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
    return grid_opt, grid_disp, meta


def _scale_routers(routers, from_grid, to_grid):
    """Scale router coords from one grid space to another."""
    fh, fw = from_grid.shape
    th, tw = to_grid.shape
    sx = tw / fw
    sy = th / fh
    return [(min(int(x * sx), tw - 1), min(int(y * sy), th - 1)) for x, y in routers]


# ── Custom fast GA that uses vectorized fitness ─────────────────────────────

def _run_fast_ga(grid, num_routers, population_size=20, generations=30, seed=42):
    """
    GA using a fast vectorized fitness function instead of the slow
    cell-by-cell signal_math version.
    """
    import random as rng_mod
    rng = rng_mod.Random(seed)

    free = get_free_cells(grid)
    if num_routers > len(free):
        raise ValueError("num_routers exceeds number of free cells")

    population = [rng.sample(free, num_routers) for _ in range(population_size)]
    best_routers = population[0][:]
    best_fit = float('-inf')

    for _ in range(generations):
        fits = [_vectorized_fitness(ind, grid) for ind in population]

        gen_best_idx = max(range(len(population)), key=lambda i: fits[i])
        if fits[gen_best_idx] > best_fit:
            best_fit = fits[gen_best_idx]
            best_routers = population[gen_best_idx][:]

        # Elitism + tournament selection + crossover + mutation
        elite = sorted(range(len(population)), key=lambda i: fits[i], reverse=True)[:2]
        new_pop = [population[i][:] for i in elite]

        while len(new_pop) < population_size:
            a = max(rng.sample(range(len(population)), 3), key=lambda i: fits[i])
            b = max(rng.sample(range(len(population)), 3), key=lambda i: fits[i])
            pa, pb = population[a], population[b]
            if len(pa) >= 2 and rng.random() < 0.8:
                cut = rng.randint(1, len(pa) - 1)
                child = pa[:cut] + pb[cut:]
            else:
                child = pa[:]
            child = [rng.choice(free) if rng.random() < 0.2 else g for g in child]
            # repair duplicates
            seen = set()
            repaired = []
            for c in child:
                if c not in seen:
                    repaired.append(c)
                    seen.add(c)
            while len(repaired) < num_routers:
                c = rng.choice(free)
                if c not in seen:
                    repaired.append(c)
                    seen.add(c)
            new_pop.append(repaired)

        population = new_pop

    return best_routers


# ── Placement strategies ─────────────────────────────────────────────────────

def ga_placement(grid, num_routers, seed=42):
    return _run_fast_ga(grid, num_routers, population_size=20, generations=30, seed=seed)


def random_placement(grid, num_routers, seed=None):
    rng = _random.Random(seed)
    free = get_free_cells(grid)
    if num_routers > len(free):
        raise ValueError("num_routers exceeds number of free cells")
    return rng.sample(free, num_routers)


def uniform_placement(grid, num_routers):
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
    zone_w = (max_x - min_x + 1) / cols
    zone_h = (max_y - min_y + 1) / rows

    routers = []
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
                for cand in sorted(free, key=lambda p: (p[0] - cx) ** 2 + (p[1] - cy) ** 2):
                    if cand not in routers:
                        routers.append(cand)
                        break

    return routers[:num_routers]


# ── Visualization ─────────────────────────────────────────────────────────

STRATEGY_LABELS = {
    "ga": "Genetic Algorithm",
    "random": "Random",
    "uniform": "Uniform Grid",
}


def visualize_and_save(grid_disp, routers_disp, strategy,
                       grid_opt=None, routers_opt=None, output_dir=OUTPUT_DIR):
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    if grid_opt is not None and routers_opt is not None:
        heat_small = _vectorized_heatmap(grid_opt, routers_opt)
        heat_small_clipped = np.clip(heat_small, -95, -30)
        nan_mask = np.isnan(heat_small_clipped)
        from scipy.ndimage import zoom as nd_zoom
        th, tw = grid_disp.shape
        zh = th / grid_opt.shape[0]
        zw = tw / grid_opt.shape[1]
        filled = np.where(nan_mask, -200.0, heat_small_clipped)
        heat = nd_zoom(filled, (zh, zw), order=1)
        mask_up = nd_zoom(nan_mask.astype(float), (zh, zw), order=0)
        heat = np.where(mask_up > 0.5, np.nan, heat)
    else:
        heat = _vectorized_heatmap(grid_disp, routers_disp)
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
    im = ax1.imshow(heat, origin="lower", cmap="viridis", interpolation="nearest",
                    vmin=-95, vmax=-30)
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


# ── Main entry point ─────────────────────────────────────────────────────────

def run_optimization(strategy, num_routers, seed=42):
    grid_opt, grid_disp, meta = load_grids()

    if strategy == "ga":
        routers_opt = ga_placement(grid_opt, num_routers, seed=seed)
    elif strategy == "random":
        routers_opt = random_placement(grid_opt, num_routers, seed=seed)
    elif strategy == "uniform":
        routers_opt = uniform_placement(grid_opt, num_routers)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    coverage_pct, avg_signal = _vectorized_coverage(routers_opt, grid_opt)
    routers_disp = _scale_routers(routers_opt, grid_opt, grid_disp)

    image_path = visualize_and_save(
        grid_disp, routers_disp, strategy,
        grid_opt=grid_opt, routers_opt=routers_opt,
    )

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
    with open(os.path.join(OUTPUT_DIR, f"{strategy}_results.json"), "w") as f:
        json.dump(result, f, indent=2)

    return result
