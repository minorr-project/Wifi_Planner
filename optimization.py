"""
WiFi Router Placement Optimization module.
Supports three strategies: genetic algorithm (ga), random, and uniform.

Signal model:
  - Distance attenuation: S0 - k * distance_m
  - Wall penalty: WALL_PENALTY dB per wall transition crossed

GA fitness uses n_steps=50 ray sampling on a fixed 10K-cell sample,
giving the same accuracy as the final metric but 13x faster (500K lookups
vs 6.56M), enabling 400+ wall-accurate GA evaluations in ~10 seconds.
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

_S0          = -30.0
_D_LOSS_K    =   2.0
_WALL_PENALTY =  8.0
_N_STEPS_FULL = 50       # ray steps for final metrics / heatmap
_N_STEPS_GA   = 50       # same accuracy for GA fitness (fast via subsampling)
_GA_SAMPLE    = 10000    # fixed free-cell sample used inside GA evaluations
_GA_SAMPLE_SEED = 123    # reproducible sample selection


def _cell_size_m(grid):
    return 15.0 / grid.shape[1]


# ── Precomputed sample for GA fitness ────────────────────────────────────────

_SAMPLE_CACHE: dict = {}   # keyed by grid id


def _get_sample(grid):
    """
    Return (sy, sx) arrays — a fixed random sample of _GA_SAMPLE free cells.
    Cached so it is only computed once per grid shape.
    """
    key = grid.shape
    if key not in _SAMPLE_CACHE:
        free_yx = np.argwhere(grid == 0)          # (N, 2)
        rng = np.random.RandomState(_GA_SAMPLE_SEED)
        n   = min(_GA_SAMPLE, len(free_yx))
        idx = rng.choice(len(free_yx), n, replace=False)
        samp = free_yx[idx]                        # (n, 2)
        _SAMPLE_CACHE[key] = (samp[:, 0].astype(np.float32),
                              samp[:, 1].astype(np.float32))
    return _SAMPLE_CACHE[key]


# ── Core signal helpers ───────────────────────────────────────────────────────

def _wall_crossings_map(grid, rx, ry, n_steps=50):
    """
    Vectorised (H, W) wall-crossing count from router (rx, ry) to every cell.
    Uses scipy map_coordinates for fast batch ray sampling.
    """
    h, w = grid.shape
    y_idx, x_idx = np.mgrid[0:h, 0:w]
    t = np.linspace(0, 1, n_steps, dtype=np.float32).reshape(-1, 1, 1)
    x_samp = (rx + t * (x_idx - rx)).reshape(-1)
    y_samp = (ry + t * (y_idx - ry)).reshape(-1)
    sampled = map_coordinates(
        grid.astype(np.float32), [y_samp, x_samp],
        order=0, mode='nearest',
    ).reshape(n_steps, h, w)
    crossings = ((sampled[1:] == 1) & (sampled[:-1] == 0)).sum(axis=0)
    return crossings.astype(np.float32)


def _signal_map(grid, routers, n_steps=50):
    """Full (H, W) signal heatmap with wall penalty; NaN over walls."""
    h, w = grid.shape
    cs = _cell_size_m(grid)
    y_idx, x_idx = np.mgrid[0:h, 0:w].astype(np.float32)
    heat = np.full((h, w), -300.0, dtype=np.float32)
    for rx, ry in routers:
        dist_m   = np.sqrt((x_idx - rx)**2 + (y_idx - ry)**2) * cs
        crossings = _wall_crossings_map(grid, rx, ry, n_steps)
        heat = np.maximum(heat, _S0 - _D_LOSS_K * dist_m - _WALL_PENALTY * crossings)
    heat[grid != 0] = np.nan
    return heat


def _coverage_metrics(routers, grid, n_steps=50):
    """Coverage % and average signal using the full wall-penalty model."""
    heat     = _signal_map(grid, routers, n_steps)
    free_sig = heat[grid == 0]
    if len(free_sig) == 0:
        return 0.0, 0.0
    return (float((free_sig >= S_threshold).mean() * 100),
            float(free_sig.mean()))


# ── GA fitness (sampled — same accuracy, 13× faster) ─────────────────────────

def _ga_fitness(routers, grid):
    """
    Wall-accurate coverage on a fixed 10K-cell sample.

    Instead of tracing rays to all 126K free cells (6.56M map_coordinates
    lookups), we trace to a representative 10K sample (500K lookups) —
    13× faster while keeping ±0.3% statistical accuracy.  The sample is
    fixed (deterministic seed) so the GA sees a consistent landscape.
    """
    sy, sx = _get_sample(grid)
    cs      = _cell_size_m(grid)
    n_samp  = len(sy)
    best    = np.full(n_samp, -300.0, dtype=np.float32)

    for rx, ry in routers:
        dist_m = np.sqrt((sx - rx)**2 + (sy - ry)**2) * cs

        # Trace n_steps sample points along each of the n_samp rays
        t       = np.linspace(0, 1, _N_STEPS_GA, dtype=np.float32)   # (n_steps,)
        x_samp  = rx + t[:, None] * (sx - rx)                         # (n_steps, n_samp)
        y_samp  = ry + t[:, None] * (sy - ry)
        coords  = np.array([y_samp.reshape(-1), x_samp.reshape(-1)])

        wall_v  = map_coordinates(
            grid.astype(np.float32), coords, order=0, mode='nearest',
        ).reshape(_N_STEPS_GA, n_samp)

        crossings = ((wall_v[1:] == 1) & (wall_v[:-1] == 0)).sum(axis=0)
        sig = (_S0 - _D_LOSS_K * dist_m
               - _WALL_PENALTY * crossings.astype(np.float32))
        best = np.maximum(best, sig)

    return float((best >= S_threshold).mean() * 100)


# ── Grid I/O ──────────────────────────────────────────────────────────────────

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
    """One GA individual using spatial-sector sampling for diversity."""
    if n_sectors is None:
        n_sectors = max(num_routers * 4, 20)
    sorted_free = sorted(free)
    chunk   = max(1, len(sorted_free) // n_sectors)
    sectors = [sorted_free[i * chunk:(i + 1) * chunk]
               for i in range(n_sectors) if sorted_free[i * chunk:(i + 1) * chunk]]
    chosen  = rng.sample(sectors, min(num_routers, len(sectors)))
    ind, seen = [], set()
    for s in chosen:
        c = rng.choice(s)
        if c not in seen:
            ind.append(c); seen.add(c)
    while len(ind) < num_routers:
        c = rng.choice(free)
        if c not in seen:
            ind.append(c); seen.add(c)
    return ind


def _run_ga(grid, num_routers, population_size=20, generations=20, seed=42):
    """
    Genetic Algorithm with wall-accurate sampled fitness.

    The fitness is computed on a fixed 10K-cell sample with n_steps=50,
    giving the same wall-detection accuracy as the final metric but running
    in ~14 ms per evaluation vs 189 ms full-grid.

    20 pop × 20 gen = 400 evaluations × ~14 ms ≈ 5.6 s for the GA.
    Final full-grid coverage metrics are computed once afterwards.
    """
    rng  = _random.Random(seed)
    free = get_free_cells(grid)
    if num_routers > len(free):
        raise ValueError("num_routers exceeds number of free cells")

    # Warm up the sample cache before timing-sensitive evolution
    _get_sample(grid)

    # Diverse initialisation — covers every region of the building
    pop = [_diverse_individual(free, num_routers, rng)
           for _ in range(population_size)]

    best_ind = pop[0][:]
    best_fit = float('-inf')

    for gen in range(generations):
        fits = [_ga_fitness(ind, grid) for ind in pop]

        gi = max(range(len(pop)), key=lambda i: fits[i])
        if fits[gi] > best_fit:
            best_fit = fits[gi]
            best_ind = pop[gi][:]

        # Elitism: carry top-2 unchanged
        elite_idx = sorted(range(len(pop)),
                           key=lambda i: fits[i], reverse=True)[:2]
        new_pop   = [pop[i][:] for i in elite_idx]

        while len(new_pop) < population_size:
            # Tournament selection (size 3)
            a  = max(rng.sample(range(len(pop)), 3), key=lambda i: fits[i])
            b  = max(rng.sample(range(len(pop)), 3), key=lambda i: fits[i])
            pa, pb = pop[a], pop[b]

            # Crossover
            if len(pa) >= 2 and rng.random() < 0.8:
                cut   = rng.randint(1, len(pa) - 1)
                child = pa[:cut] + pb[cut:]
            else:
                child = pa[:]

            # Mutation (30%)
            child = [rng.choice(free) if rng.random() < 0.3 else g
                     for g in child]

            # Repair duplicates
            seen, repaired = set(), []
            for c in child:
                if c not in seen:
                    repaired.append(c); seen.add(c)
            while len(repaired) < num_routers:
                c = rng.choice(free)
                if c not in seen:
                    repaired.append(c); seen.add(c)
            new_pop.append(repaired)

        pop = new_pop

    return best_ind


def ga_placement(grid, num_routers, seed=42):
    return _run_ga(grid, num_routers, seed=seed)


def random_placement(grid, num_routers, seed=None):
    rng  = _random.Random(seed)
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

    xs = [x for x, y in free];  ys = [y for x, y in free]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    cols   = math.ceil(math.sqrt(num_routers))
    rows   = math.ceil(num_routers / cols)
    zone_w = (max_x - min_x + 1) / cols
    zone_h = (max_y - min_y + 1) / rows

    routers = []
    for r in range(rows):
        for c in range(cols):
            if len(routers) >= num_routers:
                break
            cx = min_x + (c + 0.5) * zone_w
            cy = min_y + (r + 0.5) * zone_h
            for cand in sorted(free,
                               key=lambda p: (p[0]-cx)**2 + (p[1]-cy)**2):
                if cand not in routers:
                    routers.append(cand); break
    return routers[:num_routers]


# ── Visualization ─────────────────────────────────────────────────────────────

STRATEGY_LABELS = {
    "ga":      "Genetic Algorithm",
    "random":  "Random",
    "uniform": "Uniform Grid",
}


def visualize_and_save(grid_disp, routers_disp, strategy,
                       grid_opt=None, routers_opt=None,
                       output_dir=OUTPUT_DIR):
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    # Compute heatmap on opt grid then upscale (avoids 2M-cell loop)
    if grid_opt is not None and routers_opt is not None:
        heat_s    = np.clip(_signal_map(grid_opt, routers_opt, _N_STEPS_FULL),
                            -95, -30)
        nan_mask  = np.isnan(heat_s)
        th, tw    = grid_disp.shape
        zh, zw    = th / grid_opt.shape[0], tw / grid_opt.shape[1]
        from scipy.ndimage import zoom as nd_zoom
        filled    = np.where(nan_mask, -200.0, heat_s)
        heat      = nd_zoom(filled, (zh, zw), order=1)
        mask_up   = nd_zoom(nan_mask.astype(float), (zh, zw), order=0)
        heat      = np.where(mask_up > 0.5, np.nan, heat)
    else:
        heat = np.clip(_signal_map(grid_disp, routers_disp, _N_STEPS_FULL),
                       -95, -30)

    label = STRATEGY_LABELS.get(strategy, strategy)
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    ax0 = axes[0]
    ax0.imshow(grid_disp, cmap="gray_r", origin="lower", interpolation="nearest")
    for i, (x, y) in enumerate(routers_disp, start=1):
        ax0.plot(x, y, "ro", markersize=8, markeredgecolor="black")
        ax0.text(x, y + 3, f"R{i}", ha="center", va="bottom",
                 bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85))
    ax0.set_title(f"{label} – Router Placement")
    ax0.set_xticks([]); ax0.set_yticks([])

    ax1 = axes[1]
    im  = ax1.imshow(heat, origin="lower", cmap="viridis",
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
        # No fixed seed — each run gives a genuinely different random placement
        routers_opt = random_placement(grid_opt, num_routers, seed=None)
    elif strategy == "uniform":
        routers_opt = uniform_placement(grid_opt, num_routers)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    coverage_pct, avg_signal = _coverage_metrics(
        routers_opt, grid_opt, n_steps=_N_STEPS_FULL)
    routers_disp = _scale_routers(routers_opt, grid_opt, grid_disp)

    image_path = visualize_and_save(
        grid_disp, routers_disp, strategy,
        grid_opt=grid_opt, routers_opt=routers_opt,
    )

    result = {
        "strategy":                    strategy,
        "num_routers":                 num_routers,
        "routers_optimization_grid":   [{"x": int(x), "y": int(y)}
                                        for x, y in routers_opt],
        "routers_display_grid":        [{"x": int(x), "y": int(y)}
                                        for x, y in routers_disp],
        "coverage_percent":            float(coverage_pct),
        "average_signal_dBm":         float(avg_signal),
        "image_path":                  image_path,
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, f"{strategy}_results.json"), "w") as f:
        json.dump(result, f, indent=2)

    return result
