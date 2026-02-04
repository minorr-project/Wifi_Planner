import numpy as np
import os

def load_grid(path="grid.npy"):
    if not os.path.exists(path):
        raise FileNotFoundError("grid.npy not found. Run preprocessing first.")
    return np.load(path).astype("uint8")
