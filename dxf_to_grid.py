import ezdxf
import numpy as np
import math

DXF_FILE = "house.dxf"
CELL_SIZE = 1.0  # meters

# 1) Load DXF
doc = ezdxf.readfile(DXF_FILE)
msp = doc.modelspace()

# 2) Collect wall segments from LINE and LWPOLYLINE
lines = []

for e in msp:
    etype = e.dxftype()

    # Case 1: Simple LINE
    if etype == "LINE":
        x1, y1 = e.dxf.start.x, e.dxf.start.y
        x2, y2 = e.dxf.end.x, e.dxf.end.y
        lines.append(((x1, y1), (x2, y2)))

    # Case 2: LWPOLYLINE (very common from SketchUp)
    elif etype == "LWPOLYLINE":
        points = list(e.get_points())  # [(x, y, ...), ...]
        for i in range(len(points) - 1):
            x1, y1 = points[i][0], points[i][1]
            x2, y2 = points[i + 1][0], points[i + 1][1]
            lines.append(((x1, y1), (x2, y2)))

print(f"Collected {len(lines)} wall segments")

if len(lines) == 0:
    print("No wall geometry found at all. Check DXF export.")
    raise SystemExit


# 3) Find bounds
xs = [p[0] for line in lines for p in line]
ys = [p[1] for line in lines for p in line]

min_x, max_x = min(xs), max(xs)
min_y, max_y = min(ys), max(ys)

width = max_x - min_x
height = max_y - min_y

grid_w = math.ceil(width / CELL_SIZE) + 1
grid_h = math.ceil(height / CELL_SIZE) + 1

grid = np.zeros((grid_h, grid_w), dtype=np.uint8)  # 0 = free, 1 = wall

# 4) Mark lines onto grid by sampling points along each line
def mark_line(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    steps = max(2, int(dist / (CELL_SIZE * 0.25)))  # more samples -> better wall continuity

    for i in range(steps + 1):
        t = i / steps
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)

        gx = int((x - min_x) / CELL_SIZE)
        gy = int((y - min_y) / CELL_SIZE)

        if 0 <= gx < grid_w and 0 <= gy < grid_h:
            grid[gy, gx] = 1

for p1, p2 in lines:
    mark_line(p1, p2)

np.save("grid.npy", grid)
print("Saved grid.npy with shape:", grid.shape)
print("Unique values in grid:", np.unique(grid))