import os
import numpy as np
import matplotlib.pyplot as plt

from member_B_signal_simulation_engine.signal_math import best_signal, S_threshold


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
    plt.close(fig)
    return fig


def visualize_all(grid, methods, out_path="outputs/images/compare.png"):
    """
    Compare multiple router placements side-by-side.
    methods: dict of {name: [(x, y), ...]}
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    names = list(methods.keys())
    n = len(names)

    fig, axes = plt.subplots(2, n, figsize=(5 * n, 8))
    if n == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    last_im = None

    for i, name in enumerate(names):
        routers = methods[name]

        # Top: floorplan + routers
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

        # Bottom: signal heatmap (dBm)
        ax_bot = axes[1, i]
        heat = make_dbm_heatmap(grid, routers)
        heat = np.clip(heat, -95, -30)
        last_im = ax_bot.imshow(heat, origin="lower", cmap="viridis")
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
    print(f"Saved: {out_path}")
    plt.close(fig)
    return fig
