"""
WiFi Router Placement Optimization module.
Supports three strategies: genetic algorithm (ga), random, and uniform.

Signal model:
  - Distance attenuation: S0 - k * distance_m
  - Wall penalty: WALL_PENALTY dB per wall transition crossed (using vectorized
    ray sampling via scipy map_coordinates — fast, no Python loops)
"""

import os
import math
import json
import random as _random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates

from member_A_genetic_Algorithm_core.ga_core import get_free_cells
from member_B_signal_simulation_engine.signal_math import S_threshold

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(REPO_ROOT, "outputs")

# ── Signal constants (must match signal_math.py) ────────────────────────────
_S0 = -30.0
_D_LOSS_K = 2.0
_WALL_PENALTY = 8.0


# ── Core vectorized signal model ─────────────────────────────────────────────

def _cell_size_m(grid):
    """Metres per cell, calibrated to a ~15 m reference building width."""
    return 15.0 / grid.shape[1]


def _wall_crossings_map(grid, rx, ry, n_steps=50):
    """
    For a single router at (rx, ry), compute the number of wall-entry
    transitions (FREE→WALL) from the router to *every* cell in one
    vectorised pass using scipy map_coordinates.

    Returns an (H, W) float32 array of crossing counts.
    """
    h, w = grid.shape
    y_idx, x_idx = np.mgrid[0:h, 0:w]          # (H, W) each

    # Sample positions along the line router→cell: shape (n_steps, H, W)
    t = np.linspace(0, 1, n_steps, dtype=np.float32).reshape(-1, 1, 1)
    x_samp = (rx + t * (x_idx - rx)).reshape(-1)  # (n_steps*H*W,)
    y_samp = (ry + t * (y_idx - ry)).reshape(-1)

    # Sample grid values (0=free, 1=wall) at each interpolated point
    sampled = map_coordinates(
        grid.astype(np.float32),
        [y_samp, x_samp],
        order=0, mode='nearest',
    ).reshape(n_steps, h, w)

    # Count FREE→WALL transitions along each ray
    crossings = ((sampled[1:] == 1) & (sampled[:-1] == 0)).sum(axis=0)
    return crossings.astype(np.float32)


def _signal_map(grid, routers, n_steps=50):
    """
    Full signal heatmap (dBm) with distance attenuation + wall penalty.
    Returns (H, W) array; NaN over wall cells.
    """
    h, w = grid.shape
    cs = _cell_size_m(grid)
    y_idx, x_idx = np.mgrid[0:h, 0:w].astype(np.float32)
    heat = np.full((h, w), -300.0, dtype=np.float32)

    for rx, ry in routers:
        dist_m = np.sqrt((x_idx - rx) ** 2 + (y_idx - ry) ** 2) * cs
        crossings = _wall_crossings_map(grid, rx, ry, n_steps)
        sig = _S0 - _D_LOSS_K * dist_m - _WALL_PENALTY * crossings
        heat = np.maximum(heat, sig)

    heat[grid != 0] = np.nan
    return heat


def _coverage_metrics(routers, grid, n_steps=50):
    """
    Compute coverage % and average signal using the full wall-penalty model.
    """
    heat = _signal_map(grid, routers, n_steps)
    free_sig = heat[grid == 0]
    if len(free_sig) == 0:
        return 0.0, 0.0
    coverage = float((free_sig >= S_threshold).mean() * 100)
    avg_sig = float(free_sig.mean())
    return coverage, avg_sig


def _ga_fitness(routers, grid):
    """
    Wall-aware fitness for the GA using n_steps=6 ray sampling.
    Fast enough for hundreds of evaluations while genuinely accounting
    for walls — so the GA produces meaningfully better results than random.
    """
    h, w = grid.shape
    cs = _cell_size_m(grid)
    y_idx, x_idx = np.mgrid[0:h, 0:w].astype(np.float32)
    heat = np.full((h, w), -300.0, dtype=np.float32)
    for rx, ry in routers:
        dist_m = np.sqrt((x_idx - rx) ** 2 + (y_idx - ry) ** 2) * cs
        crossings = _wall_crossings_map(grid, rx, ry, n_steps=6)
        sig = _S0 - _D_LOSS_K * dist_m - _WALL_PENALTY * crossings
        heat = np.maximum(heat, sig)
    free = heat[grid == 0]
    if len(free) == 0:
        return 0.0
    return float((free >= S_threshold).mean() * 100)


# ── Grid I/O ─────────────────────────────────────────────────────────────────

def load_grids():
    grid_opt  = np.load(os.path.join(REPO_ROOT, "grid.npy")).astype(np.uint8)
    grid_disp = np.load(os.path.join(REPO_ROOT, "grid_display.npy")).astype(np.uint8)
    meta = {}
    meta_path = os.path.join(REPO_ROOT, "grid_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
    return grid_opt, grid_disp, meta


def _scale_routers(routers, from_grid, to_grid):
    fh, fw = from_grid.shape
    th, tw = to_grid.shape
    sx, sy = tw / fw, th / fh
    return [(min(int(x * sx), tw - 1), min(int(y * sy), th - 1))
            for x, y in routers]


# ── Placement strategies ──────────────────────────────────────────────────────

def _diverse_individual(free, num_routers, rng, n_sectors=None):
    """
    Create one individual by sampling from spatial sectors of the free space,
    ensuring candidates spread across the entire building rather than
    clustering in one corner.
    """
    if n_sectors is None:
        n_sectors = max(num_routers * 4, 20)

    # Partition sorted free cells into equal sectors
    sorted_free = sorted(free)
    chunk = max(1, len(sorted_free) // n_sectors)
    sectors = [sorted_free[i * chunk:(i + 1) * chunk]
               for i in range(n_sectors) if sorted_free[i * chunk:(i + 1) * chunk]]

    chosen_sectors = rng.sample(sectors, min(num_routers, len(sectors)))
    ind = []
    seen = set()
    for sector in chosen_sectors:
        c = rng.choice(sector)
        if c not in seen:
            ind.append(c)
            seen.add(c)
    # Fill any remaining slots randomly
    while len(ind) < num_routers:
        c = rng.choice(free)
        if c not in seen:
            ind.append(c)
            seen.add(c)
    return ind


def _run_ga(grid, num_routers, population_size=15, generations=16, seed=42):
    """
    Genetic Algorithm with wall-aware fitness and spatially diverse
    population initialization.

    Key improvements over purely random initialization:
    - Pop[0] is the uniform-grid placement (a strong, well-spread seed).
    - Remaining members are sampled from spatial sectors so every region
      of the building is represented from generation 0.
    - Mutation rate 0.3 (vs 0.2) encourages broader exploration.

    15 pop × 16 gen = 240 evaluations × ~15 ms each ≈ 3.6 s total.
    """
    rng = _random.Random(seed)
    free = get_free_cells(grid)
    if num_routers > len(free):
        raise ValueError("num_routers exceeds number of free cells")

    # Seed pop[0] with the uniform placement so we always start from a
    # known-good spatially-spread solution
    uniform_seed = uniform_placement(grid, num_routers)
    pop = [uniform_seed]

    # Fill remaining population with spatially diverse individuals
    while len(pop) < population_size:
        pop.append(_diverse_individual(free, num_routers, rng))

    best_ind = pop[0][:]
    best_fit = float('-inf')

    for _ in range(generations):
        fits = [_ga_fitness(ind, grid) for ind in pop]

        gi = max(range(len(pop)), key=lambda i: fits[i])
        if fits[gi] > best_fit:
            best_fit = fits[gi]
            best_ind = pop[gi][:]

        elite_idx = sorted(range(len(pop)), key=lambda i: fits[i], reverse=True)[:2]
        new_pop = [pop[i][:] for i in elite_idx]

        while len(new_pop) < population_size:
            a = max(rng.sample(range(len(pop)), 3), key=lambda i: fits[i])
            b = max(rng.sample(range(len(pop)), 3), key=lambda i: fits[i])
            pa, pb = pop[a], pop[b]
            if len(pa) >= 2 and rng.random() < 0.8:
                cut = rng.randint(1, len(pa) - 1)
                child = pa[:cut] + pb[cut:]
            else:
                child = pa[:]
            # Higher mutation rate (0.3) for broader exploration
            child = [rng.choice(free) if rng.random() < 0.3 else g for g in child]
            seen, repaired = set(), []
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

        pop = new_pop

    return best_ind


def ga_placement(grid, num_routers, seed=42):
    return _run_ga(grid, num_routers, seed=seed)


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
            for cand in sorted(free, key=lambda p: (p[0]-cx)**2 + (p[1]-cy)**2):
                if cand not in routers:
                    routers.append(cand)
                    break
    return routers[:num_routers]


# ── Visualization ─────────────────────────────────────────────────────────────

STRATEGY_LABELS = {
    "ga":      "Genetic Algorithm",
    "random":  "Random",
    "uniform": "Uniform Grid",
}


def visualize_and_save(grid_disp, routers_disp, strategy,
                       grid_opt=None, routers_opt=None, output_dir=OUTPUT_DIR):
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    # Compute heatmap on opt grid (small) then upscale — avoids the 2M-cell loop
    if grid_opt is not None and routers_opt is not None:
        heat_small = _signal_map(grid_opt, routers_opt, n_steps=50)
        heat_small = np.clip(heat_small, -95, -30)
        nan_mask = np.isnan(heat_small)
        th, tw = grid_disp.shape
        zh, zw = th / grid_opt.shape[0], tw / grid_opt.shape[1]
        from scipy.ndimage import zoom as nd_zoom
        filled = np.where(nan_mask, -200.0, heat_small)
        heat = nd_zoom(filled, (zh, zw), order=1)
        mask_up = nd_zoom(nan_mask.astype(float), (zh, zw), order=0)
        heat = np.where(mask_up > 0.5, np.nan, heat)
    else:
        heat = _signal_map(grid_disp, routers_disp, n_steps=50)
        heat = np.clip(heat, -95, -30)

    label = STRATEGY_LABELS.get(strategy, strategy)
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    # Left: placement on floor plan
    ax0 = axes[0]
    ax0.imshow(grid_disp, cmap="gray_r", origin="lower", interpolation="nearest")
    for i, (x, y) in enumerate(routers_disp, start=1):
        ax0.plot(x, y, "ro", markersize=8, markeredgecolor="black")
        ax0.text(x, y + 3, f"R{i}", ha="center", va="bottom",
                 bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85))
    ax0.set_title(f"{label} – Router Placement")
    ax0.set_xticks([]); ax0.set_yticks([])

    # Right: signal heatmap with wall overlay
    ax1 = axes[1]
    im = ax1.imshow(heat, origin="lower", cmap="viridis",
                    interpolation="nearest", vmin=-95, vmax=-30)
    wall_mask = np.ma.masked_where(grid_disp == 0, grid_disp)
    ax1.imshow(wall_mask, origin="lower", cmap="gray_r",
               alpha=0.9, interpolation="nearest")
    for i, (x, y) in enumerate(routers_disp, start=1):
        ax1.plot(x, y, "wo", markersize=6, markeredgecolor="black")
        ax1.text(x, y + 3, f"R{i}", ha="center", va="bottom",
                 bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85))
    ax1.set_title(f"Signal Heatmap (threshold = {S_threshold} dBm)")
    ax1.set_xticks([]); ax1.set_yticks([])

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
    cbar.set_label("Signal (dBm)")

    plt.tight_layout()
    save_path = os.path.join(images_dir, f"{strategy}_placement.png")
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return save_path


# ── Main entry point ──────────────────────────────────────────────────────────

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

    # Accurate coverage metrics using the full wall-penalty model
    coverage_pct, avg_signal = _coverage_metrics(routers_opt, grid_opt, n_steps=50)
    routers_disp = _scale_routers(routers_opt, grid_opt, grid_disp)

    image_path = visualize_and_save(
        grid_disp, routers_disp, strategy,
        grid_opt=grid_opt, routers_opt=routers_opt,
    )

    result = {
        "strategy": strategy,
        "num_routers": num_routers,
        "routers_optimization_grid": [{"x": int(x), "y": int(y)} for x, y in routers_opt],
        "routers_display_grid":      [{"x": int(x), "y": int(y)} for x, y in routers_disp],
        "coverage_percent":          float(coverage_pct),
        "average_signal_dBm":        float(avg_signal),
        "image_path":                image_path,
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, f"{strategy}_results.json"), "w") as f:
        json.dump(result, f, indent=2)

    return result
