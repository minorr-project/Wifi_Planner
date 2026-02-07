# WiFi Planner - Grid Generation

## Overview
Python CLI project that converts architectural floor plans (DXF/SketchUp export) into simulation-ready grids for indoor Wi-Fi planning.

## Pipeline
1. `flatten_3dface.py` - Extracts 3DFACE polygons from DXF and projects to 2D, saves `wall_faces_xy.npy`
2. `rasterize_to_grid.py` - Rasterizes wall faces into a binary grid, cleans noise, crops, saves `grid.npy`, `grid_meta.json`, `grid_preview.png`
3. `dxf_to_grid.py` - Alternative simpler approach using LINE/LWPOLYLINE entities
4. `inspect_dxf.py` - Utility to inspect DXF file structure

## Key Files
- `house.dxf` - Input floor plan
- `grid.npy` - Output binary grid (1=wall, 0=free)
- `grid_meta.json` - Grid metadata (bounds, cell size, crop info)
- `grid_preview.png` - Visual preview of the grid

## Dependencies
- Python 3.12
- ezdxf, numpy, matplotlib, scipy, scikit-learn, pillow
- Listed in `requirements.txt`

## Running
```bash
python flatten_3dface.py && python rasterize_to_grid.py
```
