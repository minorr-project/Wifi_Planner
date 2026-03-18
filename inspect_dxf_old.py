import ezdxf
from collections import Counter

DXF_FILE = "house.dxf"

doc = ezdxf.readfile(DXF_FILE)

print("DXF loaded.")
print("Modelspace entities:", len(doc.modelspace()))
print("Paperspace entities:", len(doc.paperspace()))

# Count types in modelspace
msp = doc.modelspace()
types_msp = Counter(e.dxftype() for e in msp)
print("\n--- MODELSPACE entity types ---")
for k, v in types_msp.most_common():
    print(f"{k}: {v}")

# Count types in paperspace
psp = doc.paperspace()
types_psp = Counter(e.dxftype() for e in psp)
print("\n--- PAPERSPACE entity types ---")
for k, v in types_psp.most_common():
    print(f"{k}: {v}")

# Show blocks (common when geometry is stored as INSERT)
print("\n--- BLOCKS (names) ---")
block_names = [b.name for b in doc.blocks]
print("Total blocks:", len(block_names))
print("Some block names:", block_names[:20])