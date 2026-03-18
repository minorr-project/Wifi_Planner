import sys
import ezdxf
from collections import Counter


def inspect_dxf(dxf_path: str):
    doc = ezdxf.readfile(dxf_path)

    print("DXF loaded.")
    print("File:", dxf_path)
    print("Modelspace entities:", len(doc.modelspace()))
    print("Paperspace entities:", len(doc.paperspace()))

    msp = doc.modelspace()
    types_msp = Counter(e.dxftype() for e in msp)
    print("\n--- MODELSPACE entity types ---")
    for k, v in types_msp.most_common():
        print(f"{k}: {v}")

    psp = doc.paperspace()
    types_psp = Counter(e.dxftype() for e in psp)
    print("\n--- PAPERSPACE entity types ---")
    for k, v in types_psp.most_common():
        print(f"{k}: {v}")

    print("\n--- BLOCKS (names) ---")
    block_names = [b.name for b in doc.blocks]
    print("Total blocks:", len(block_names))
    print("Some block names:", block_names[:20])

    print("\n--- SIMPLE STRATEGY HINT ---")
    if types_msp.get("INSERT", 0) > 0:
        print("This DXF contains INSERT entities. A block-aware or render fallback may be needed.")
    elif types_msp.get("3DFACE", 0) > 0:
        print("This DXF contains 3DFACE geometry. 3DFACE extraction may help.")
    elif types_msp.get("LWPOLYLINE", 0) + types_msp.get("LINE", 0) + types_msp.get("POLYLINE", 0) > 0:
        print("Vector wall extraction from lines/polylines is likely appropriate.")
    else:
        print("Use render-based fallback.")

    return types_msp


if __name__ == "__main__":
    dxf_path = sys.argv[1] if len(sys.argv) > 1 else "house.dxf"
    inspect_dxf(dxf_path)