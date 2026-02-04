#!/usr/bin/env python3
"""
WiFi Router Placement Optimization - Full Pipeline Integration
Integrates all team members' code:
  - Person C: DXF → Grid (root scripts)
  - Member A: Genetic Algorithm
  - Member B: Signal Simulation
  - Member D: Visualization
"""

import os
import sys
import numpy as np
import json

# Ensure repo root is in path for imports
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# ============================================================
# STEP 1: Load the grid (produced by Person C's pipeline)
# ============================================================
def load_grid():
    """Load grid.npy and convert to string format for GA/signal code."""
    grid_path = os.path.join(REPO_ROOT, "grid.npy")
    meta_path = os.path.join(REPO_ROOT, "grid_meta.json")
    
    if not os.path.exists(grid_path):
        raise FileNotFoundError(
            "grid.npy not found. Run the DXF pipeline first:\n"
            "  python3 dxf_to_grid.py\n"
            "  python3 rasterize_to_grid.py"
        )
    
    # Load binary grid (0=free, 1=wall)
    grid_binary = np.load(grid_path)
    
    # Load metadata if available
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)
    
    # Convert to string format expected by GA/signal code
    # GA expects: "FREE", "WALL", "DOOR", "WINDOW"
    height, width = grid_binary.shape
    grid_str = []
    for y in range(height):
        row = []
        for x in range(width):
            row.append("WALL" if grid_binary[y, x] == 1 else "FREE")
        grid_str.append(row)
    
    return grid_binary, grid_str, meta


# ============================================================
# STEP 2: Run Genetic Algorithm (Member A)
# ============================================================
def run_optimization(grid_str, num_routers=3, generations=50, population_size=30, seed=42):
    """Run GA to find optimal router placement."""
    from member_A_genetic_Algorithm_core.ga_core import run_ga
    
    print(f"\n Running Genetic Algorithm...")
    print(f"   • Routers to place: {num_routers}")
    print(f"   • Generations: {generations}")
    print(f"   • Population size: {population_size}")
    
    result = run_ga(
        grid_str,
        num_routers=num_routers,
        generations=generations,
        population_size=population_size,
        seed=seed,
    )
    
    print(f"   • Best fitness (coverage): {result['best_fitness']:.2f}%")
    print(f"   • Best router positions: {result['best_routers']}")
    
    return result


# ============================================================
# STEP 3: Calculate final coverage metrics (Member B)
# ============================================================
def calculate_coverage(routers, grid_str):
    """Calculate coverage using Member B's signal physics."""
    from member_B_signal_simulation_engine.signal_physics import coverage_metrics
    
    coverage_pct, avg_signal = coverage_metrics(routers, grid_str)
    
    print(f"\n Signal Analysis (Member B):")
    print(f"   • Coverage: {coverage_pct:.2f}%")
    print(f"   • Average signal: {avg_signal:.2f} dBm")
    
    return coverage_pct, avg_signal


# ============================================================
# STEP 4: Visualization (Member D)
# ============================================================
def visualize_results(grid_binary, routers, output_dir="outputs"):
    """Generate visualization using Member D's code."""
    import matplotlib.pyplot as plt
    
    # Import Member D's visualization utilities
    sys.path.insert(0, os.path.join(REPO_ROOT, "member_D_visualization", "src"))
    from visualizationD2 import (
        calculate_coverage_for_routers,
        calculate_coverage_stats,
        plot_single_analysis,
    )
    
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    
    print(f"\n Generating visualizations...")
    
    # Calculate coverage map for visualization
    coverage_map = calculate_coverage_for_routers(grid_binary, routers)
    
    # Generate heatmap
    save_path = os.path.join(output_dir, "images", "optimized_placement.png")
    coverage_pct, fig = plot_single_analysis(
        grid_binary, coverage_map, routers, save_path=save_path
    )
    
    plt.close(fig)
    
    print(f"   • Saved: {save_path}")
    
    return coverage_map


# ============================================================
# STEP 5: Save results
# ============================================================
def save_results(routers, coverage_pct, avg_signal, meta, output_dir="outputs"):
    """Save optimization results to JSON."""
    os.makedirs(output_dir, exist_ok=True)
    
    results = {
        "routers": [{"x": int(r[0]), "y": int(r[1])} for r in routers],
        "coverage_percent": float(coverage_pct),
        "average_signal_dBm": float(avg_signal),
        "grid_meta": meta,
    }
    
    results_path = os.path.join(output_dir, "optimization_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"   • Saved: {results_path}")
    
    return results_path


# ============================================================
# MAIN PIPELINE
# ============================================================
def main():
    print("=" * 60)
    print(" WiFi Router Placement Optimization - Full Pipeline")
    print("=" * 60)
    
    # Configuration
    NUM_ROUTERS = 3
    GENERATIONS = 50
    POPULATION_SIZE = 30
    OUTPUT_DIR = "outputs"
    
    # Step 1: Load grid
    print("\n[1/5] Loading grid data (Person C)...")
    grid_binary, grid_str, meta = load_grid()
    print(f"   • Grid shape: {grid_binary.shape}")
    print(f"   • Wall cells: {int(grid_binary.sum())}")
    print(f"   • Free cells: {int(grid_binary.size - grid_binary.sum())}")
    
    # Step 2: Run GA optimization
    print("\n[2/5] Running optimization (Member A)...")
    ga_result = run_optimization(
        grid_str,
        num_routers=NUM_ROUTERS,
        generations=GENERATIONS,
        population_size=POPULATION_SIZE,
    )
    best_routers = ga_result["best_routers"]
    
    # Step 3: Calculate final coverage
    print("\n[3/5] Calculating signal coverage (Member B)...")
    coverage_pct, avg_signal = calculate_coverage(best_routers, grid_str)
    
    # Step 4: Generate visualizations
    print("\n[4/5] Generating visualizations (Member D)...")
    visualize_results(grid_binary, best_routers, OUTPUT_DIR)
    
    # Step 5: Save results
    print("\n[5/5] Saving results...")
    save_results(best_routers, coverage_pct, avg_signal, meta, OUTPUT_DIR)
    
    # Summary
    print("\n" + "=" * 60)
    print(" OPTIMIZATION COMPLETE")
    print("=" * 60)
    print(f" Router Positions: {best_routers}")
    print(f" Coverage: {coverage_pct:.2f}%")
    print(f" Average Signal: {avg_signal:.2f} dBm")
    print(f"\n Output files in: {OUTPUT_DIR}/")
    print("   • optimization_results.json")
    print("   • images/optimized_placement.png")
    print("=" * 60)


if __name__ == "__main__":
    main()
