import os
import sys
import json
import math
from collections import Counter, deque

import ezdxf
import numpy as np


# ============================================================
# Utility functions
# ============================================================

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_json(path: str, data: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def save_preview(grid: np.ndarray, path: str, title: str = "Grid Preview"):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 8))
    plt.imshow(grid, origin="lower", cmap="gray")
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def downsample_max(grid: np.ndarray, factor: int = 4) -> np.ndarray:
    h, w = grid.shape
    new_h = h // factor
    new_w = w // factor
    out = np.zeros((new_h, new_w), dtype=np.uint8)

    for y in range(new_h):
        for x in range(new_w):
            block = grid[y * factor:(y + 1) * factor, x * factor:(x + 1) * factor]
            out[y, x] = 1 if np.any(block == 1) else 0

    return out


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


def remove_small_components(mask: np.ndarray, min_size: int = 20) -> np.ndarray:
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
        return grid, {
            "cropped": False,
            "reason": "no wall cells"
        }

    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()

    y_min = max(0, y_min - margin)
    y_max = min(grid.shape[0] - 1, y_max + margin)
    x_min = max(0, x_min - margin)
    x_max = min(grid.shape[1] - 1, x_max + margin)

    cropped = grid[y_min:y_max + 1, x_min:x_max + 1]

    info = {
        "cropped": True,
        "crop_box": {
            "y_min": int(y_min),
            "y_max": int(y_max),
            "x_min": int(x_min),
            "x_max": int(x_max),
        },
        "original_shape": [int(grid.shape[0]), int(grid.shape[1])],
        "cropped_shape": [int(cropped.shape[0]), int(cropped.shape[1])],
    }
    return cropped, info


# ============================================================
# DXF inspection
# ============================================================

def inspect_entities(doc):
    msp = doc.modelspace()
    counts = Counter(e.dxftype() for e in msp)
    return counts


def choose_strategy(counts: Counter) -> str:
    if counts.get("INSERT", 0) > 0:
        return "render_fallback"

    if counts.get("3DFACE", 0) > 0 and counts.get("LWPOLYLINE", 0) == 0 and counts.get("LINE", 0) == 0:
        return "render_fallback"

    if counts.get("LINE", 0) + counts.get("LWPOLYLINE", 0) + counts.get("POLYLINE", 0) + counts.get("ARC", 0) + counts.get("CIRCLE", 0) > 0:
        return "vector"

    return "render_fallback"


# ============================================================
# Vector extraction
# ============================================================

def add_line(lines, p1, p2):
    x1, y1 = float(p1[0]), float(p1[1])
    x2, y2 = float(p2[0]), float(p2[1])

    if math.isclose(x1, x2) and math.isclose(y1, y2):
        return

    lines.append(((x1, y1), (x2, y2)))


def arc_to_lines(center, radius, start_deg, end_deg, segments=36):
    cx, cy = center
    if end_deg < start_deg:
        end_deg += 360.0

    pts = []
    for i in range(segments + 1):
        t = i / segments
        ang = math.radians(start_deg + t * (end_deg - start_deg))
        x = cx + radius * math.cos(ang)
        y = cy + radius * math.sin(ang)
        pts.append((x, y))

    lines = []
    for i in range(len(pts) - 1):
        lines.append((pts[i], pts[i + 1]))
    return lines


def circle_to_lines(center, radius, segments=48):
    return arc_to_lines(center, radius, 0.0, 360.0, segments=segments)


def collect_vector_geometry(doc):
    msp = doc.modelspace()
    lines = []

    for e in msp:
        etype = e.dxftype()

        try:
            if etype == "LINE":
                add_line(lines, (e.dxf.start.x, e.dxf.start.y), (e.dxf.end.x, e.dxf.end.y))

            elif etype == "LWPOLYLINE":
                points = list(e.get_points())
                if len(points) >= 2:
                    for i in range(len(points) - 1):
                        add_line(lines, (points[i][0], points[i][1]), (points[i + 1][0], points[i + 1][1]))

                    is_closed = bool(getattr(e, "closed", False))
                    if is_closed:
                        add_line(lines, (points[-1][0], points[-1][1]), (points[0][0], points[0][1]))

            elif etype == "POLYLINE":
                pts = []
                try:
                    for v in e.vertices:
                        pts.append((v.dxf.location.x, v.dxf.location.y))
                except Exception:
                    pts = []

                if len(pts) >= 2:
                    for i in range(len(pts) - 1):
                        add_line(lines, pts[i], pts[i + 1])

                    is_closed = bool(getattr(e, "is_closed", False))
                    if is_closed:
                        add_line(lines, pts[-1], pts[0])

            elif etype == "ARC":
                center = (e.dxf.center.x, e.dxf.center.y)
                radius = float(e.dxf.radius)
                start_deg = float(e.dxf.start_angle)
                end_deg = float(e.dxf.end_angle)

                for p1, p2 in arc_to_lines(center, radius, start_deg, end_deg, segments=36):
                    add_line(lines, p1, p2)

            elif etype == "CIRCLE":
                center = (e.dxf.center.x, e.dxf.center.y)
                radius = float(e.dxf.radius)

                # Ignore extremely tiny circles, usually symbols/noise
                if radius >= 0.05:
                    for p1, p2 in circle_to_lines(center, radius, segments=48):
                        add_line(lines, p1, p2)

            # TEXT/MTEXT/INSERT/HATCH etc intentionally ignored here

        except Exception:
            continue

    return lines


def compute_bounds(lines):
    xs = [p[0] for line in lines for p in line]
    ys = [p[1] for line in lines for p in line]
    return min(xs), min(ys), max(xs), max(ys)


def mark_line(grid, p1, p2, min_x, min_y, cell_size, line_sample_step):
    x1, y1 = p1
    x2, y2 = p2

    dx = x2 - x1
    dy = y2 - y1
    dist = math.hypot(dx, dy)

    steps = max(1, int(dist / line_sample_step))

    for i in range(steps + 1):
        t = i / steps

        x = x1 + t * dx
        y = y1 + t * dy

        gx = int((x - min_x) / cell_size)
        gy = int((y - min_y) / cell_size)

        if 0 <= gx < grid.shape[1] and 0 <= gy < grid.shape[0]:
            grid[gy, gx] = 1
def vector_to_grid(doc, target_display_long_side=1600, min_cell_size=0.2, line_sample_factor=0.5):
    lines = collect_vector_geometry(doc)
    if len(lines) == 0:
        raise RuntimeError("No usable vector geometry found.")

    min_x, min_y, max_x, max_y = compute_bounds(lines)
    width = max_x - min_x
    height = max_y - min_y
    long_side = max(width, height)

    # Auto-select cell size so the longer side becomes about target_display_long_side cells
    display_cell_size = max(min_cell_size, long_side / target_display_long_side)

    # Sample points along each line at a fraction of one cell
    line_sample_step = display_cell_size * 0.25

    print("\nDXF bounds:")
    print("min_x:", min_x, "max_x:", max_x)
    print("min_y:", min_y, "max_y:", max_y)
    print("width:", width)
    print("height:", height)
    print("auto display_cell_size:", display_cell_size)
    print("auto line_sample_step:", line_sample_step)

    grid_w = int(math.ceil(width / display_cell_size)) + 1
    grid_h = int(math.ceil(height / display_cell_size)) + 1

    print("grid_w:", grid_w)
    print("grid_h:", grid_h)

    grid = np.zeros((grid_h, grid_w), dtype=np.uint8)

    for p1, p2 in lines:
        mark_line(grid, p1, p2, min_x, min_y, display_cell_size, line_sample_step)

    #grid = dilate_binary(grid, r=1)
    #grid = remove_small_components(grid, min_size=20)
    grid, crop_info = crop_to_walls(grid, margin=10)

    meta = {
        "source": "VECTOR_DXF",
        "cell_size": float(display_cell_size),
        "line_sample_step": float(line_sample_step),
        "bounds": {
            "min_x": float(min_x),
            "min_y": float(min_y),
            "max_x": float(max_x),
            "max_y": float(max_y),
        },
        "grid_shape": [int(grid.shape[0]), int(grid.shape[1])],
        "codes": {"FREE": 0, "WALL": 1},
        "crop": crop_info,
        "notes": "Generated from LINE/LWPOLYLINE/POLYLINE/ARC/CIRCLE entities using adaptive cell size."
    }

    return grid, meta

# ============================================================
# Render fallback
# ============================================================

def render_fallback_to_grid(dxf_path: str, out_width=1400, thresh=245):
    import matplotlib.pyplot as plt
    from ezdxf.addons.drawing import RenderContext, Frontend
    from ezdxf.addons.drawing.matplotlib import MatplotlibBackend

    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()

    fig = plt.figure(figsize=(10, 10), dpi=140)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()

    ctx = RenderContext(doc)
    out = MatplotlibBackend(ax)
    Frontend(ctx, out).draw_layout(msp, finalize=True)

    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
    plt.close(fig)

    gray = img.mean(axis=2)
    wall = (gray < thresh).astype(np.uint8)

    wall = dilate_binary(wall, r=1)
    wall = remove_small_components(wall, min_size=30)
    wall, crop_info = crop_to_walls(wall, margin=10)

    meta = {
        "source": "RENDER_FALLBACK",
        "grid_shape": [int(wall.shape[0]), int(wall.shape[1])],
        "codes": {"FREE": 0, "WALL": 1},
        "crop": crop_info,
        "notes": "Rendered DXF fallback. Can capture annotations as walls."
    }

    return wall, meta


# ============================================================
# Main processing
# ============================================================

def process_dxf(dxf_path: str, output_dir: str = ".", optimization_downsample: int = 4):
    ensure_dir(output_dir)

    doc = ezdxf.readfile(dxf_path)
    counts = inspect_entities(doc)
    strategy = choose_strategy(counts)

    if strategy == "vector":
        grid_display, meta = vector_to_grid(
    doc,
    target_display_long_side=1600,
    min_cell_size=0.2,
    line_sample_factor=0.5
)
    else:
        grid_display, meta = render_fallback_to_grid(dxf_path)

    grid_opt = downsample_max(grid_display, factor=optimization_downsample)

    meta["entity_counts"] = dict(counts)
    meta["strategy"] = strategy
    meta["input_dxf"] = os.path.abspath(dxf_path)

    meta_opt = dict(meta)
    meta_opt["derived_from"] = "grid_display.npy"
    meta_opt["downsample_factor_for_optimization"] = optimization_downsample
    meta_opt["grid_shape"] = [int(grid_opt.shape[0]), int(grid_opt.shape[1])]

    np.save(os.path.join(output_dir, "grid_display.npy"), grid_display)
    save_json(os.path.join(output_dir, "grid_display_meta.json"), meta)
    save_preview(grid_display, os.path.join(output_dir, "grid_display_preview.png"), title="Display Grid")

    np.save(os.path.join(output_dir, "grid.npy"), grid_opt)
    save_json(os.path.join(output_dir, "grid_meta.json"), meta_opt)
    save_preview(grid_opt, os.path.join(output_dir, "grid_preview.png"), title="Optimization Grid")

    summary = {
        "input_dxf": os.path.abspath(dxf_path),
        "strategy": strategy,
        "entity_counts": dict(counts),
        "display_grid_shape": list(grid_display.shape),
        "optimization_grid_shape": list(grid_opt.shape),
        "output_dir": os.path.abspath(output_dir),
    }
    save_json(os.path.join(output_dir, "dxf_processing_summary.json"), summary)

    print("\nDXF processing completed.")
    print("Input DXF:", dxf_path)
    print("Strategy:", strategy)
    print("Display grid shape:", grid_display.shape)
    print("Optimization grid shape:", grid_opt.shape)
    print("Saved files in:", output_dir)


if __name__ == "__main__":
    dxf_path = sys.argv[1] if len(sys.argv) > 1 else "house.dxf"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "."
    process_dxf(dxf_path, output_dir=output_dir, optimization_downsample=4)