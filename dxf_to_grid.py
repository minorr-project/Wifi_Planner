import ezdxf
import numpy as np
import math
import json

DXF_FILE = "house.dxf"

# High-resolution vector grid
CELL_SIZE = 0.05   # meters per cell for display-quality wall extraction
LINE_SAMPLE_STEP = 0.01  # sampling resolution along line segments


def add_line(lines, p1, p2):
    x1, y1 = float(p1[0]), float(p1[1])
    x2, y2 = float(p2[0]), float(p2[1])
    if not (math.isclose(x1, x2) and math.isclose(y1, y2)):
        lines.append(((x1, y1), (x2, y2)))


def collect_lines_from_dxf(dxf_path):
    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()

    lines = []

    for e in msp:
        etype = e.dxftype()

        if etype == "LINE":
            add_line(lines, (e.dxf.start.x, e.dxf.start.y), (e.dxf.end.x, e.dxf.end.y))

        elif etype == "LWPOLYLINE":
            points = list(e.get_points())
            if len(points) >= 2:
                for i in range(len(points) - 1):
                    add_line(lines, (points[i][0], points[i][1]), (points[i + 1][0], points[i + 1][1]))

                # IMPORTANT: close the polyline if closed
                try:
                    is_closed = bool(e.closed)
                except Exception:
                    is_closed = False

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

                try:
                    is_closed = bool(e.is_closed)
                except Exception:
                    is_closed = False

                if is_closed:
                    add_line(lines, pts[-1], pts[0])

    return lines


def compute_bounds(lines):
    xs = [p[0] for line in lines for p in line]
    ys = [p[1] for line in lines for p in line]
    return min(xs), min(ys), max(xs), max(ys)


def mark_line(grid, p1, p2, min_x, min_y, cell_size):
    x1, y1 = p1
    x2, y2 = p2

    dx = x2 - x1
    dy = y2 - y1
    dist = math.hypot(dx, dy)

    steps = max(1, int(dist / LINE_SAMPLE_STEP))

    for i in range(steps + 1):
        t = i / steps
        x = x1 + t * dx
        y = y1 + t * dy

        gx = int((x - min_x) / cell_size)
        gy = int((y - min_y) / cell_size)

        if 0 <= gy < grid.shape[0] and 0 <= gx < grid.shape[1]:
            grid[gy, gx] = 1


def dilate_binary(mask, r=1):
    mask = (mask > 0).astype(np.uint8)
    h, w = mask.shape
    out = mask.copy()
    ys, xs = np.where(mask > 0)

    for y, x in zip(ys, xs):
        y0, y1 = max(0, y - r), min(h, y + r + 1)
        x0, x1 = max(0, x - r), min(w, x + r + 1)
        out[y0:y1, x0:x1] = 1

    return out


def crop_to_walls(grid, margin=10):
    ys, xs = np.where(grid > 0)
    if len(xs) == 0:
        return grid, {"cropped": False, "reason": "no wall cells"}

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


def save_preview(grid, path="grid_preview.png"):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 8))
    plt.imshow(grid, origin="lower", cmap="viridis")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def main():
    lines = collect_lines_from_dxf(DXF_FILE)
    print(f"Collected {len(lines)} line segments")

    if len(lines) == 0:
        raise RuntimeError("No LINE/LWPOLYLINE/POLYLINE geometry found.")

    min_x, min_y, max_x, max_y = compute_bounds(lines)
    width = max_x - min_x
    height = max_y - min_y

    grid_w = int(math.ceil(width / CELL_SIZE)) + 1
    grid_h = int(math.ceil(height / CELL_SIZE)) + 1

    grid = np.zeros((grid_h, grid_w), dtype=np.uint8)

    for p1, p2 in lines:
        mark_line(grid, p1, p2, min_x, min_y, CELL_SIZE)

    # very light thickening, only for continuity
    grid = dilate_binary(grid, r=1)

    grid, crop_info = crop_to_walls(grid, margin=10)

    meta = {
        "source": "VECTOR_DXF",
        "cell_size": CELL_SIZE,
        "bounds": {
            "min_x": float(min_x),
            "min_y": float(min_y),
            "max_x": float(max_x),
            "max_y": float(max_y),
        },
        "grid_shape": [int(grid.shape[0]), int(grid.shape[1])],
        "codes": {"FREE": 0, "WALL": 1},
        "crop": crop_info,
        "notes": "High-resolution vector-derived wall grid from LINE/LWPOLYLINE/POLYLINE."
    }

    np.save("grid_display.npy", grid)
    with open("grid_display_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # also save as grid.npy for compatibility
    np.save("grid.npy", grid)
    with open("grid_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    save_preview(grid, "grid_preview.png")

    print("Saved:")
    print("  grid_display.npy")
    print("  grid_display_meta.json")
    print("  grid.npy")
    print("  grid_meta.json")
    print("  grid_preview.png")
    print("Grid shape:", grid.shape)
    print("Unique values:", np.unique(grid))


if __name__ == "__main__":
    main()