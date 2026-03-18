# WiFi Planner – Grid Generator

A web-based tool that converts architectural floor plans (DXF/SketchUp exports) into simulation-ready binary wall grids for indoor Wi-Fi planning.

## Architecture

- **Backend**: Flask web server (`app.py`) on port 5000
- **Frontend**: Single-page HTML/JS UI (`templates/index.html`)
- **Pipeline scripts**:
  - `flatten_3dface.py` – Extracts 3DFACE entities from DXF and flattens them to 2D polygons → `wall_faces_xy.npy`
  - `rasterize_to_grid.py` – Rasterizes the 2D polygons into a binary wall grid, cleans noise, crops → `grid.npy`, `grid_meta.json`, `grid_preview.png`
  - `dxf_to_grid.py` – Alternative: processes LINE/LWPOLYLINE entities (simple DXF)
  - `inspect_dxf.py` – Utility to inspect entity types in a DXF file

## Web UI Features

- Upload any `.dxf` floor plan file (SketchUp / AutoCAD)
- Run the full pipeline (extract → flatten → rasterize → clean → crop)
- Live status + progress bar during pipeline execution
- Visual preview of the generated wall grid
- Download `grid.npy` and `grid_meta.json`
- Grid metadata display (size, cell size, bounds)

## Pipeline Output

| File | Description |
|------|-------------|
| `grid.npy` | Binary NumPy array: 1=wall, 0=free space |
| `grid_meta.json` | Cell size, bounds, grid shape, crop info |
| `grid_preview.png` | Visual PNG of the wall grid |
| `wall_faces_xy.npy` | Intermediate: 2D polygon list extracted from 3DFACE entities |

## Running

Start the web server workflow "Start Web App" (port 5000).

## Dependencies

- Flask
- ezdxf
- numpy
- matplotlib
- scikit-learn
- scipy
- pillow
