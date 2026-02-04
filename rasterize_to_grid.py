import os
import numpy as np
import json
import math
import matplotlib.pyplot as plt

WALL_FACES_FILE = "wall_faces_xy.npy"
GRID_FILE_FALLBACK = "grid.npy"  # produced by dxf_to_grid.py


def crop_grid_to_walls(grid, margin=20):
    ys, xs = np.where(grid > 0)
    if len(xs) == 0 or len(ys) == 0:
        return grid, {"cropped": False, "reason": "no wall cells found"}

    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()

    y_min = max(0, y_min - margin)
    y_max = min(grid.shape[0] - 1, y_max + margin)
    x_min = max(0, x_min - margin)
    x_max = min(grid.shape[1] - 1, x_max + margin)

    cropped = grid[y_min:y_max + 1, x_min:x_max + 1]

    info = {
        "cropped": True,
        "crop_box": {"y_min": int(y_min), "y_max": int(y_max), "x_min": int(x_min), "x_max": int(x_max)},
        "original_shape": [int(grid.shape[0]), int(grid.shape[1])],
        "cropped_shape": [int(cropped.shape[0]), int(cropped.shape[1])]
    }
    return cropped, info


def load_faces_if_valid(path: str):
    """Return faces array if it exists and contains at least 1 face; else return None."""
    if not os.path.exists(path):
        return None

    faces = np.load(path, allow_pickle=True)

    # empty file case
    if faces is None or faces.size == 0 or len(faces) == 0:
        return None

    # Basic shape sanity: expect list-like of polygons
    try:
        _ = faces[0]
    except Exception:
        return None

    return faces


def compute_bounds(faces):
    xs = []
    ys = []
    for poly in faces:
        for (x, y) in poly:
            xs.append(float(x))
            ys.append(float(y))
    if not xs or not ys:
        raise ValueError("compute_bounds() got empty faces.")
    return min(xs), min(ys), max(xs), max(ys)


def rasterize_faces_to_grid(faces, cell_size=0.25, face_sample_step=0.05):
    minx, miny, maxx, maxy = compute_bounds(faces)

    width = maxx - minx
    height = maxy - miny

    gw = int(math.ceil(width / cell_size)) + 1
    gh = int(math.ceil(height / cell_size)) + 1

    grid = np.zeros((gh, gw), dtype=np.uint8)

    def mark_point(x, y):
        gx = int((x - minx) / cell_size)
        gy = int((y - miny) / cell_size)
        if 0 <= gx < gw and 0 <= gy < gh:
            grid[gy, gx] = 1

    for poly in faces:
        pts = [(float(x), float(y)) for (x, y) in poly]

        # Use all edges in the polygon (not assuming exactly 4 points)
        n = len(pts)
        if n < 2:
            continue

        for i in range(n):
            x1, y1 = pts[i]
            x2, y2 = pts[(i + 1) % n]

            dx = x2 - x1
            dy = y2 - y1
            dist = math.hypot(dx, dy)

            steps = max(1, int(dist / face_sample_step))
            for s in range(steps + 1):
                t = s / steps
                x = x1 + t * dx
                y = y1 + t * dy
                mark_point(x, y)

    meta = {
        "source": "3DFACE",
        "cell_size": cell_size,
        "bounds": {"minx": float(minx), "miny": float(miny), "maxx": float(maxx), "maxy": float(maxy)},
        "grid_shape": [int(gh), int(gw)],
        "codes": {"FREE": 0, "WALL": 1},
        "notes": "Grid derived from 3DFACE edges (2D projection)."
    }

    return grid, meta


def keep_largest_wall_component(grid):
    from collections import deque

    H, W = grid.shape
    wall = (grid > 0)

    visited = np.zeros((H, W), dtype=bool)
    best_count = 0
    best_cells = None

    dirs = [(-1,-1), (-1,0), (-1,1),
            (0,-1),          (0,1),
            (1,-1),  (1,0),  (1,1)]

    for r in range(H):
        for c in range(W):
            if wall[r, c] and not visited[r, c]:
                q = deque([(r, c)])
                visited[r, c] = True
                cells = [(r, c)]
                cnt = 1

                while q:
                    cr, cc = q.popleft()
                    for dr, dc in dirs:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < H and 0 <= nc < W:
                            if wall[nr, nc] and not visited[nr, nc]:
                                visited[nr, nc] = True
                                q.append((nr, nc))
                                cells.append((nr, nc))
                                cnt += 1

                if cnt > best_count:
                    best_count = cnt
                    best_cells = cells

    cleaned = np.zeros_like(grid)
    if best_cells is None:
        return cleaned

    for r, c in best_cells:
        cleaned[r, c] = 1
    return cleaned


def save_outputs(grid, meta):
    np.save("grid.npy", grid)
    with open("grid_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    plt.figure(figsize=(8, 8))
    plt.imshow(grid == 1, origin="lower")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("grid_preview.png", dpi=200)
    plt.close()

    print("Saved grid.npy, grid_meta.json, grid_preview.png")
    print("Grid shape:", grid.shape)


if __name__ == "__main__":
    # 1) Try 3DFACE path
    faces = load_faces_if_valid(WALL_FACES_FILE)

    if faces is not None:
        print("Loaded faces:", len(faces))
        grid, meta = rasterize_faces_to_grid(
            faces,
            cell_size=1.5,
            face_sample_step=0.3
        )
    else:
        # 2) Fallback: use 2D grid generated by dxf_to_grid.py
        if not os.path.exists(GRID_FILE_FALLBACK):
            raise RuntimeError(
                "No valid 3DFACE walls found (wall_faces_xy.npy missing/empty) AND grid.npy not found.\n"
                "Run dxf_to_grid.py first to generate grid.npy for 2D DXF floorplans."
            )

        grid = np.load(GRID_FILE_FALLBACK).astype(np.uint8)
        meta = {
            "source": "2D_GRID_FALLBACK",
            "cell_size": None,
            "codes": {"FREE": 0, "WALL": 1},
            "notes": "Loaded existing grid.npy generated from LINE/LWPOLYLINE DXF extraction."
        }
        print("Using fallback grid.npy:", grid.shape)

    print("Grid shape BEFORE cleanup:", grid.shape)
    print("Wall cells BEFORE cleanup:", int(grid.sum()))

    grid_clean = keep_largest_wall_component(grid)

    print("Wall cells AFTER cleanup:", int(grid_clean.sum()))

    grid_cropped, crop_info = crop_grid_to_walls(grid_clean, margin=20)

    print("Grid shape AFTER crop:", grid_cropped.shape)
    print("Wall cells AFTER crop:", int(grid_cropped.sum()))

    meta["crop"] = crop_info
    meta["cleanup"] = {"method": "largest_connected_component"}

    save_outputs(grid_cropped, meta)
    print("Saved CLEANED+CROPPED: grid.npy, grid_meta.json, grid_preview.png")
