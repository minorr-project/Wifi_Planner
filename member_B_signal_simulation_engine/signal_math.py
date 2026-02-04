"""
Signal model + coverage fitness for WiFi router placement.
Works with Member C grid:
  grid[y][x] == 0 -> FREE
  grid[y][x] == 1 -> WALL
"""

import math
import numpy as np

# -----------------------------
# CALIBRATION (IMPORTANT)
# -----------------------------
# Set an approximate real-world building width (meters).
# Typical small house width ~ 10–20m. Pick one.
TARGET_BUILDING_WIDTH_M = 15.0

# We'll compute CELL_SIZE_METERS dynamically from grid width:
# CELL_SIZE_METERS = TARGET_BUILDING_WIDTH_M / grid_width_cells
CELL_SIZE_METERS = None  # computed at runtime


def calibrate_cell_size(grid, target_width_m=TARGET_BUILDING_WIDTH_M):
    """
    Sets global CELL_SIZE_METERS based on the grid width.
    This prevents the "500m house" problem.
    """
    global CELL_SIZE_METERS
    H, W = grid.shape
    CELL_SIZE_METERS = float(target_width_m) / float(W)
    return CELL_SIZE_METERS


# -----------------------------
# CONSTANTS AND PARAMETERS
# -----------------------------
S0 = -30.0          # reference signal at router (dBm)
S_threshold = -75.0 # usable threshold (dBm)  (tune -70 to -80)
d_loss_k = 2.0      # distance attenuation (dB per meter)
WALL_PENALTY = 8.0  # penalty per wall crossing (dB)


# -----------------------------
# GEOMETRY
# -----------------------------
def distance_cells(router, cell):
    x_r, y_r = router
    x_p, y_p = cell
    return math.hypot(x_p - x_r, y_p - y_r)


def distance_loss(distance_m, k=d_loss_k):
    return k * distance_m


# -----------------------------
# OBSTACLE LOSS (stable version)
# -----------------------------
def obstacle_loss(router, cell, grid, wall_penalty=WALL_PENALTY):
    """
    Count wall crossings along a line (FREE->WALL transitions),
    not "wall pixels touched". This avoids crazy -1000 dBm values.
    """
    x_r, y_r = router
    x_p, y_p = cell

    steps = int(max(abs(x_p - x_r), abs(y_p - y_r)))
    if steps == 0:
        return 0.0

    wall_crossings = 0
    prev = 0  # 0 free, 1 wall

    for step in range(1, steps + 1):
        t = step / steps
        x = int(round(x_r + t * (x_p - x_r)))
        y = int(round(y_r + t * (y_p - y_r)))

        if 0 <= y < grid.shape[0] and 0 <= x < grid.shape[1]:
            cur = 1 if grid[y, x] == 1 else 0
            if cur == 1 and prev == 0:
                wall_crossings += 1
            prev = cur

    return wall_crossings * wall_penalty


# -----------------------------
# SIGNAL
# -----------------------------
def signal_strength(router, cell, grid):
    global CELL_SIZE_METERS
    if CELL_SIZE_METERS is None:
        calibrate_cell_size(grid)

    d_cells = distance_cells(router, cell)
    d_m = d_cells * CELL_SIZE_METERS

    loss_d = distance_loss(d_m)
    loss_o = obstacle_loss(router, cell, grid)

    return S0 - loss_d - loss_o


def best_signal(cell, routers, grid):
    return max(signal_strength(r, cell, grid) for r in routers)


def is_cell_covered(signal, thres=S_threshold):
    return signal >= thres


# -----------------------------
# COVERAGE METRICS
# -----------------------------
def coverage_metrics(routers, grid, max_range_m=30.0):
    """
    Evaluate coverage only within max_range_m of at least one router.
    This makes fitness stable and realistic (and faster).
    """
    global CELL_SIZE_METERS
    if CELL_SIZE_METERS is None:
        calibrate_cell_size(grid)

    H, W = grid.shape
    total = 0
    covered = 0
    sig_sum = 0.0

    max_range_cells = max_range_m / CELL_SIZE_METERS

    for y in range(H):
        for x in range(W):
            if grid[y, x] != 0:  # only FREE cells
                continue

            # skip cells far from all routers
            in_range = False
            for rx, ry in routers:
                if distance_cells((rx, ry), (x, y)) <= max_range_cells:
                    in_range = True
                    break
            if not in_range:
                continue

            total += 1
            s = best_signal((x, y), routers, grid)
            sig_sum += s
            if is_cell_covered(s):
                covered += 1

    if total == 0:
        return 0.0, 0.0

    coverage_percentage = (covered / total) * 100.0
    average_signal = sig_sum / total
    return coverage_percentage, average_signal


# -----------------------------
# FITNESS (for GA)
# -----------------------------
def fitness_function(routers, grid, beta=0.0):
    coverage, _avg = coverage_metrics(routers, grid)

    # optional overlap penalty later
    overlap_penalty = 0.0

    return coverage - beta * overlap_penalty
