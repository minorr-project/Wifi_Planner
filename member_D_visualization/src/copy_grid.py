import shutil
import os

# Person C's folder path
person_c_folder = r"C:\Users\ACER\OneDrive\Documents\Python_clone(Mridul)\Wifi_Planner_PD-main"

# Your folder path
your_folder = r"C:\Users\ACER\OneDrive\Documents\Python_MP"

# Files to copy
files = ["grid.npy", "grid_meta.json"]

print("=" * 50)
print("COPYING PERSON C'S GRID DATA")
print("=" * 50)

for file in files:
    source = os.path.join(person_c_folder, file)
    destination = os.path.join(your_folder, file)
    
    print(f"\nCopying: {file}")
    print(f"From: {source}")
    print(f"To: {destination}")
    
    if os.path.exists(source):
        try:
            shutil.copy2(source, destination)
            print("✓ SUCCESS!")
        except Exception as e:
            print(f"✗ ERROR: {e}")
    else:
        print(f"✗ SOURCE FILE NOT FOUND!")

print("\n" + "=" * 50)
print("CHECKING WHAT EXISTS:")
print("=" * 50)

# Check Person C's folder
print("\nIn Person C's folder:")
for file in files:
    path = os.path.join(person_c_folder, file)
    if os.path.exists(path):
        print(f"✓ {file} exists")
    else:
        print(f"✗ {file} missing")

# Check your folder
print("\nIn your folder:")
for file in files:
    path = os.path.join(your_folder, file)
    if os.path.exists(path):
        print(f"✓ {file} exists")
    else:
        print(f"✗ {file} missing")

print("\n" + "=" * 50)