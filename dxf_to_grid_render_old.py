# dxf_to_grid_render.py
from __future__ import annotations

import numpy as np
import ezdxf
import matplotlib.pyplot as plt

from ezdxf.addons.drawing import RenderContext, Frontend
from ezdxf.addons.drawing.matplotlib import MatplotlibBackend


def render_dxf_to_rgb(dxf_path: str, dpi: int = 300) -> np.ndarray:
    """
    Render DXF using ezdxf drawing add-on (matplotlib backend).
    Returns RGB image as uint8 array (H, W, 3).
    """
    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()

    fig = plt.figure(figsize=(8, 8), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()

    ctx = RenderContext(doc)
    out = MatplotlibBackend(ax)
    Frontend(ctx, out).draw_layout(msp, finalize=True)

    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
    plt.close(fig)
    return img


def _resize_nearest(mask: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    y_idx = (np.linspace(0, mask.shape[0] - 1, out_h)).astype(int)
    x_idx = (np.linspace(0, mask.shape[1] - 1, out_w)).astype(int)
    return mask[np.ix_(y_idx, x_idx)]


def dilate_binary(mask: np.ndarray, r: int = 2) -> np.ndarray:
    """
    Simple dilation without scipy. r=2 or 3 usually makes walls look like walls.
    """
    mask = (mask > 0).astype(np.uint8)
    h, w = mask.shape
    out = mask.copy()
    ys, xs = np.where(mask > 0)
    for y, x in zip(ys, xs):
        y0, y1 = max(0, y - r), min(h, y + r + 1)
        x0, x1 = max(0, x - r), min(w, x + r + 1)
        out[y0:y1, x0:x1] = 1
    return out


def dxf_to_grid_render(
    dxf_path: str,
    out_size: int = 700,
    thresh: int = 245,
    wall_dilate: int = 2,
    dpi: int = 300,
) -> np.ndarray:
    """
    Main: DXF -> rendered image -> threshold -> resized -> dilated wall grid.
    Returns grid where 1=wall, 0=free.
    """
    img = render_dxf_to_rgb(dxf_path, dpi=dpi)
    gray = img.mean(axis=2)

    # wall pixels are darker than background
    wall = (gray < thresh).astype(np.uint8)

    # resize to grid size
    wall = _resize_nearest(wall, out_size, out_size)

    # thicken walls
    if wall_dilate > 0:
        wall = dilate_binary(wall, r=wall_dilate)

    return wall


def preview_grid(grid: np.ndarray, title: str = "Grid Preview"):
    plt.figure(figsize=(8, 8))
    plt.imshow(grid, origin="lower")
    plt.title(title)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    g = dxf_to_grid_render("house.dxf", out_size=700, thresh=245, wall_dilate=2, dpi=300)
    np.save("grid.npy", g)
    preview_grid(g, "Rendered DXF -> Grid")
