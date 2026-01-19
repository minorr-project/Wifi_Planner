# WiFi Planner – Grid Generation

This module converts an architectural floor plan (DXF / SketchUp export)
into a simulation-ready grid for indoor Wi-Fi planning.

## Output
- `grid.npy`  
  Binary grid where:
  - 1 = wall / obstacle
  - 0 = free space

- `grid_meta.json`  
  Metadata including grid scale and crop information.

## Pipeline
1. Extract wall faces from DXF
2. Flatten 3D faces to 2D
3. Rasterize walls into a grid
4. Remove noise (largest connected component)
5. Crop grid to building bounds

## Usage
```bash
python rasterize_to_grid.py
