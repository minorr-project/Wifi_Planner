# AGENTS.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project overview

This repository contains a small toolkit for converting an architectural floor plan (DXF / SketchUp export) into a simulation-ready occupancy grid for indoor Wi‑Fi planning. The main deliverables are:
- `grid.npy`: binary grid where 1 = wall/obstacle, 0 = free space
- `grid_meta.json`: metadata describing grid scale, bounds, and crop information
- `grid_preview.png`: visual preview of the wall mask for quick validation

Core logic lives in a few standalone Python scripts in the repo root; there is no package layout, test suite, or build system.

## Dependencies and environment

- Python project using external libraries such as `ezdxf`, `numpy`, and `matplotlib` (see `requirements.txt` for the authoritative list).
- Before running any scripts, install dependencies from the repository root:
  ```bash
  pip install -r requirements.txt
  ```

## Key scripts and data flow

All scripts assume the working directory is the repository root and operate on files in-place.

### 1. DXF inspection helper (`inspect_dxf.py`)

Purpose:
- Quickly inspect the contents of a DXF floor plan, focusing on entity counts and types in modelspace/paperspace and listing defined blocks.

Typical command:
- Inspect a DXF (defaults to `house.dxf`):
  ```bash
  python inspect_dxf.py
  ```

### 2. 3DFACE-based pipeline (`flatten_3dface.py` → `rasterize_to_grid.py`)

High-level flow:
1. `flatten_3dface.py`
   - Reads a DXF file (default `house.dxf`) using `ezdxf`.
   - Extracts all `3DFACE` entities and projects them into 2D `(x, y)` wall polygons.
   - Writes the resulting list of 4-point polygons to `wall_faces_xy.npy` for downstream processing.

   Command:
   ```bash
   python flatten_3dface.py
   ```

2. `rasterize_to_grid.py` (main pipeline script referenced by the README)
   - Loads `wall_faces_xy.npy` via `load_faces`.
   - Computes global bounds over all wall vertices and allocates a 2D grid based on `cell_size`.
   - Rasterizes walls by sampling along 3DFACE polygon edges (`rasterize_faces_to_grid`).
   - Cleans the grid by keeping only the largest 8-connected wall component (`keep_largest_wall_component`), removing speck noise.
   - Crops the grid tightly around remaining walls with a configurable margin (`crop_grid_to_walls`).
   - Saves:
     - `grid.npy` (binary occupancy grid)
     - `grid_meta.json` (cell size, bounds, crop metadata, cleanup notes)
     - `grid_preview.png` (matplotlib visualization of wall cells)

   Command (end-to-end from existing `wall_faces_xy.npy`):
   ```bash
   python rasterize_to_grid.py
   ```

This two-stage pipeline (`flatten_3dface.py` then `rasterize_to_grid.py`) is the main, more robust route for generating Wi‑Fi planning grids from 3D wall geometry.

### 3. LINE/LWPOLYLINE-based alternative (`dxf_to_grid.py`)

Purpose:
- A simpler, single-script pipeline that works directly from DXF `LINE` and `LWPOLYLINE` entities instead of 3DFACE polygons.

High-level behavior:
- Reads `house.dxf` using `ezdxf` and iterates over modelspace entities.
- Collects wall segments from:
  - `LINE` entities (start/end points)
  - `LWPOLYLINE` entities (consecutive vertex pairs)
- Computes XY bounds, creates a grid at a given `CELL_SIZE`, and samples along each segment to mark wall cells.
- Writes the resulting binary grid to `grid.npy`.

Command:
```bash
python dxf_to_grid.py
```

### 4. Shared algorithmic building blocks

Across the scripts, the main architectural ideas are:
- **DXF parsing layer** (using `ezdxf`): isolates CAD-specific concerns (entity types, modelspace vs. paperspace, blocks).
- **Geometry-to-grid projection**:
  - For 3DFACE: `flatten_3dface.py` flattens 3D wall faces to 2D polygons, and `rasterize_to_grid.py` samples polygon edges into grid indices.
  - For LINE/LWPOLYLINE: `dxf_to_grid.py` samples along line segments directly into the grid.
- **Post-processing** (in `rasterize_to_grid.py`): connected-component analysis to keep only the main building structure, plus cropping to a tight bounding box with margin.

When extending the system (e.g., to handle new DXF entity types or multi-floor layouts), keep the separation between DXF parsing, geometric interpretation, and grid post-processing.

## Running and validating the pipeline

Common development commands from the repo root:
- Inspect input DXF structure:
  ```bash
  python inspect_dxf.py
  ```
- 3DFACE-based pipeline (recommended):
  ```bash
  python flatten_3dface.py
  python rasterize_to_grid.py
  ```
- LINE/LWPOLYLINE-based alternative (single-step):
  ```bash
  python dxf_to_grid.py
  ```

After running a pipeline, validate results by examining `grid_preview.png` and checking `grid_meta.json` for reasonable bounds and cell size.

## Testing and linting status

- There is currently **no automated test suite** or `tests/` directory in this repository, and no configured test runner.
- There is no project-level configuration for linters/formatters (e.g., no `pyproject.toml`, `setup.cfg`, or linter config files).

If tests or linting are added in the future, update this section with the exact commands (including how to run a single test) so Warp can use them automatically.
