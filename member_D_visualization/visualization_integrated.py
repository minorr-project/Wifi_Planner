import os
import numpy as np
import matplotlib.pyplot as plt

from signal_math import best_signal, S_threshold


def make_dbm_heatmap(grid, routers):
    H, W = grid.shape
    heat = np.full((H, W), np.nan, dtype=float)

    for y in range(H):
        for x in range(W):
            if grid[y, x] == 1:
                continue  # wall stays NaN
            heat[y, x] = best_signal((x, y), routers, grid)

    return heat


def plot_solution(grid, routers, out_path=None):
    """
    routers: List[(x, y)]
    Produces:
      - left: floorplan + routers
      - right: signal heatmap in dBm with threshold marker
    """
    os.makedirs(os.path.dirname(out_path or "outputs/images/"), exist_ok=True)

    heat = make_dbm_heatmap(grid, routers)

    # For display range: clip to reasonable WiFi values
    display = np.clip(heat, -95, -30)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # ---- LEFT: floor plan ----
    ax1.imshow(grid, cmap="gray_r", origin="lower")
    for i, (x, y) in enumerate(routers):
        ax1.plot(x, y, "ro", markersize=8, markeredgecolor="black")
        ax1.text(x, y + 1, f"R{i+1}", ha="center", va="bottom",
                 bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
    ax1.set_title("Floorplan + Router Placement")
    ax1.set_xticks([])
    ax1.set_yticks([])

    # ---- RIGHT: signal heatmap (dBm) ----
    im = ax2.imshow(display, origin="lower", cmap="viridis")
    wall_mask = np.ma.masked_where(grid == 0, grid)
    ax2.imshow(wall_mask, origin="lower", cmap="gray_r", alpha=0.9)
    for (x, y) in routers:
        ax2.plot(x, y, "ko", markersize=6, markeredgecolor="white")
    ax2.set_title(f"Signal Heatmap (dBm)\nThreshold = {S_threshold} dBm")
    ax2.set_xticks([])
    ax2.set_yticks([])

    cbar = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label("Signal (dBm)")

    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        print("Saved:", out_path)

    plt.show()
    return fig
