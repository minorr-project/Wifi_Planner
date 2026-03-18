from __future__ import annotations

import json
import numpy as np
import ezdxf
import matplotlib.pyplot as plt

from ezdxf.addons.drawing import RenderContext, Frontend
from ezdxf.addons.drawing.matplotlib import MatplotlibBackend


def get_dxf_bounds(dxf_path: str):
    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()

    xs = []
    ys = []

    for e in msp:
        etype = e.dxftype()

        try:
            if etype == "LINE":
                xs.extend([e.dxf.start.x, e.dxf.end.x])
                ys.extend([e.dxf.start.y, e.dxf.end.y])

            elif etype == "LWPOLYLINE":
                pts = list(e.get_points())
                xs.extend([p[0] for p in pts])
                ys.extend([p[1] for p in pts])

            elif etype == "POLYLINE":
                pts = [(v.dxf.location.x, v.dxf.location.y) for v in e.vertices]
                xs.extend([p[0] for p in pts])
                ys.extend([p[1] for p in pts])

            elif etype == "3DFACE":
                xs.extend([e.dxf.vtx0.x, e.dxf.vtx1.x, e.dxf.vtx2.x, e.dxf.vtx3.x])
                ys.extend([e.dxf.vtx0.y, e.dxf.vtx1.y, e.dxf.vtx2.y, e.dxf.vtx3.y])
        except Exception:
            pass

    if not xs or not ys:
        raise RuntimeError("Could not compute DXF bounds.")

    return min(xs), min(ys), max(xs), max(ys)


def render_dxf_to_rgb(dxf_path: str, width_px: int = 1600, dpi: int = 200) -> np.ndarray:
    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()

    min_x, min_y, max_x, max_y = get_dxf_bounds(dxf_path)
    width = max_x - min_x
    height = max_y - min_y
    aspect = height / width if width > 0 else 1.0

    fig_w = width_px / dpi
    fig_h = max(4, int(width_px * aspect) / dpi)

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
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


def dilate_binary(mask: np.ndarray, r: int = 1) -> np.ndarray:
    mask = (mask > 0).astype(np.uint8)
    h, w = mask.shape
    out = mask.copy()
    ys, xs = np.where(mask > 0)

    for y, x in zip(ys, xs):
        y0, y1 = max(0, y - r), min(h, y + r + 1)
        x0, x1 = max(0, x - r), min(w, x + r + 1)
        out[y0:y1, x0:x1] = 1

    return out


def remove_small_components(mask: np.ndarray, min_size: int = 30) -> np.ndarray:
    from collections import deque

    h, w = mask.shape
    visited = np.zeros((h, w), dtype=bool)
    out = np.zeros_like(mask)

    dirs = [(-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)]

    for r in range(h):
        for c in range(w):
            if mask[r, c] == 0 or visited[r, c]:
                continue

            q = deque([(r, c)])
            visited[r, c] = True
            comp = [(r, c)]

            while q:
                cr, cc = q.popleft()
                for dr, dc in dirs:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w:
                        if mask[nr, nc] == 1 and not visited[nr, nc]:
                            visited[nr, nc] = True
                            q.append((nr, nc))
                            comp.append((nr, nc))

            if len(comp) >= min_size:
                for rr, cc in comp:
                    out[rr, cc] = 1

    return out


def crop_to_walls(grid: np.ndarray, margin: int = 10):
    ys, xs = np.where(grid > 0)
    if len(xs) == 0:
        return grid

    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()

    y_min = max(0, y_min - margin)
    y_max = min(grid.shape[0] - 1, y_max + margin)
    x_min = max(0, x_min - margin)
    x_max = min(grid.shape[1] - 1, x_max + margin)

    return grid[y_min:y_max + 1, x_min:x_max + 1]


def dxf_to_grid_render(
    dxf_path: str,
    out_width: int = 1200,
    thresh: int = 245,
    wall_dilate: int = 1,
    dpi: int = 200,
) -> np.ndarray:
    img = render_dxf_to_rgb(dxf_path, width_px=out_width, dpi=dpi)
    gray = img.mean(axis=2)

    wall = (gray < thresh).astype(np.uint8)

    if wall_dilate > 0:
        wall = dilate_binary(wall, r=wall_dilate)

    wall = remove_small_components(wall, min_size=30)
    wall = crop_to_walls(wall, margin=10)

    return wall


def preview_grid(grid: np.ndarray, title: str = "Grid Preview"):
    plt.figure(figsize=(8, 8))
    plt.imshow(grid, origin="lower", cmap="viridis")
    plt.title(title)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    g = dxf_to_grid_render("house.dxf", out_width=1200, thresh=245, wall_dilate=1, dpi=200)

    np.save("grid_render.npy", g)

    with open("grid_render_meta.json", "w") as f:
        json.dump({
            "source": "RENDERED_DXF",
            "codes": {"FREE": 0, "WALL": 1},
            "grid_shape": [int(g.shape[0]), int(g.shape[1])]
        }, f, indent=2)

    plt.figure(figsize=(8, 8))
    plt.imshow(g, origin="lower", cmap="viridis")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("grid_render_preview.png", dpi=200, bbox_inches="tight")
    plt.close()

    print("Saved grid_render.npy and grid_render_preview.png")