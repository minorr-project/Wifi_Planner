import ezdxf
import numpy as np

DXF_FILE = "house.dxf"

def extract_3dface_polygons(dxf_path):
    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()

    polygons = []

    for entity in msp:
        if entity.dxftype() == "3DFACE":
            pts = [
                (entity.dxf.vtx0.x, entity.dxf.vtx0.y),
                (entity.dxf.vtx1.x, entity.dxf.vtx1.y),
                (entity.dxf.vtx2.x, entity.dxf.vtx2.y),
                (entity.dxf.vtx3.x, entity.dxf.vtx3.y),
            ]
            polygons.append(pts)

    return polygons


if __name__ == "__main__":
    polys = extract_3dface_polygons(DXF_FILE)
    print(f"Extracted {len(polys)} wall faces")

    if len(polys) == 0:
        print("No 3DFACE entities found. Not saving wall_faces_xy.npy.")
    else:
        np.save("wall_faces_xy.npy", polys)
        print("Saved wall_faces_xy.npy")
