# WiFi Planner – Grid Generation

This module converts an architectural floor plan (DXF / SketchUp export)
into a simulation-ready occupancy grid for indoor Wi-Fi planning.

## Outputs

| File | Description |
|---|---|
| `grid_display.npy` | High-resolution binary wall grid |
| `grid_display_meta.json` | Display grid scale, bounds & crop info |
| `grid.npy` | Optimization grid (4× downsampled from display grid) |
| `grid_meta.json` | Optimization grid metadata |
| `grid_display_preview.png` | Visual preview of the display grid |
| `grid_preview.png` | Visual preview of the optimization grid |

In all grids: `1` = wall / obstacle, `0` = free space.

## Setup

Requires Python 3. Install dependencies from the repo root:

```bash
pip3 install -r requirements.txt
```

## Running via the Web UI (recommended)

1. Start the Flask server:
   ```bash
   python3 app.py
   ```
2. Open **http://localhost:5000** in your browser.
3. Drag and drop your `.dxf` floor plan onto the upload zone (or click to browse).
4. Click **▶ Run Pipeline**. Progress is shown in real time.
5. Once complete:
   - The **preview panel** shows the wall grid. Toggle between **Display** (high-res) and **Optimization (4×↓)** views.
   - The **Download Outputs** section provides all four output files.
   - **Grid Metadata** shows grid dimensions and cell size.

## Running via the command line

The general pipeline script auto-detects DXF entity types and picks the best strategy:

```bash
python3 dxf_pipeline_general.py <your_file.dxf> .
```

Example:

```bash
python3 dxf_pipeline_general.py floorplan.dxf .
```

Outputs are written to the current directory.

## Pipeline internals

`dxf_pipeline_gener`dxf_pipeline_gener`dxf_pipeline_gener`dxf_pipelinategies:

- **Vector** — used when the file contains `LINE`, `LWPOLYLINE`, `POLYLINE`, `ARC`, or `CIRCLE` entities without `INSERT` blocks. Samples geometry directly into the grid at an auto-calculated cell size.
- **Render fallback** — used when `INSERT` (block reference) entities are present. Renders the full DXF via the ezdxf matplotlib backend and thresholds the resulting image into a binary grid.

In both cases the pipeline then:
1. Removes small disconnected noise components
2. Crops the grid tightly to the building bounds with a small margin
3. Saves a high-resolution display grid and a 4× downsampled optimization grid

## Inspecting a DXF file

```bash
python3 inspect_dxf.py <your_file.dxf>
```

Prints entity type counts and suggests which pipeline strategy will be used.
