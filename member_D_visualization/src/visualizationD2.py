"""
WiFi Router Placement Optimization - Visualization Module
Person D: Heatmap and Comparison Plots
UPDATED FOR DAY 2: Uses Person C's Real Grid Data
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import json

# ========== UTILITY FUNCTIONS ==========
def load_person_c_data():
    """Load Person C's real grid data"""
    try:
        grid = np.load("grid.npy")
        with open("grid_meta.json", "r") as f:
            meta = json.load(f)
        
        print(f"   Loaded Person C's grid: {grid.shape}")
        print(f"   Wall cells: {np.sum(grid)}")
        print(f"   Free cells: {grid.size - np.sum(grid)}")
        
        return grid, meta
    except FileNotFoundError:
        print("   ERROR: Person C's grid data not found!")
        print("   Please copy grid.npy and grid_meta.json from Person C's folder")
        return None, None

def calculate_coverage_stats(grid, coverage_map, threshold=30):
    """Calculate coverage statistics for real grid"""
    free_cells = np.sum(grid == 0)
    covered_cells = np.sum(coverage_map > threshold)
    coverage_percent = (covered_cells / free_cells) * 100 if free_cells > 0 else 0
    return free_cells, covered_cells, coverage_percent

def calculate_coverage_for_routers(grid, routers):
    """Calculate WiFi coverage for given router positions - FIXED FOR BETTER COLOR VARIATION"""
    coverage = np.zeros_like(grid, dtype=float)
    height, width = grid.shape
    
    # Calculate maximum possible distance in grid
    max_possible_dist = np.sqrt(height**2 + width**2)
    
    for ry, rx in routers:
        for y in range(height):
            for x in range(width):
                if grid[y, x] == 0:  # Only free space
                    # Calculate normalized distance (0 to 1)
                    distance = np.sqrt((x - rx)**2 + (y - ry)**2)
                    normalized_dist = distance / max_possible_dist
                    
                    # Signal decays quadratically for more realistic variation
                    # Close to router: ~100%, Far away: ~0-20%
                    signal = 100 * max(0, 1 - normalized_dist**0.7)
                    
                    # Reduce signal through walls if any in line of sight (simple check)
                    if distance > 0:
                        # Check if there's a wall nearby
                        check_dist = min(3, int(distance))
                        if check_dist > 0:
                            has_wall = False
                            for t in np.linspace(0, 1, check_dist + 1):
                                check_x = int(rx + t * (x - rx))
                                check_y = int(ry + t * (y - ry))
                                if (0 <= check_x < width and 0 <= check_y < height and 
                                    grid[check_y, check_x] == 1):
                                    has_wall = True
                                    break
                            
                            if has_wall:
                                signal *= 0.5  # 50% reduction through walls
                    
                    coverage[y, x] = max(coverage[y, x], signal)
    
    return coverage

# ========== MAIN VISUALIZATION FUNCTIONS ==========
def plot_single_analysis(grid, coverage_map, routers, save_path=None):
    """
    Create a clean single heatmap analysis
    Returns: coverage percentage
    """
    free_cells, covered_cells, coverage_percent = calculate_coverage_stats(grid, coverage_map)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    
    # ----- LEFT: FLOOR PLAN -----
    # Create enhanced floor visualization
    floor_display = grid.copy().astype(float)
    for rx, ry in routers:
        floor_display[rx, ry] = 0.5  # Mark router positions
    
    cmap_floor = plt.cm.colors.ListedColormap(['#FFFFFF', '#666666', '#FF4444'])
    ax1.imshow(floor_display, cmap=cmap_floor, origin='lower', 
               extent=[0, grid.shape[1], 0, grid.shape[0]])
    
    # Add router coverage circles (scaled to grid size)
    circle_radius = min(grid.shape[0], grid.shape[1]) / 4
    for rx, ry in routers:
        circle = plt.Circle((ry + 0.5, rx + 0.5), circle_radius, 
                          color='blue', alpha=0.1, linestyle='--', linewidth=1)
        ax1.add_patch(circle)
    
    # Label routers
    for i, (rx, ry) in enumerate(routers):
        ax1.plot(ry + 0.5, rx + 0.5, 'o', markersize=12, 
                color='red', markeredgecolor='black', markeredgewidth=2)
        ax1.text(ry + 0.5, rx + 0.8, f'Router {i+1}', 
                ha='center', va='bottom', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.9))
    
    ax1.set_title("FLOOR PLAN WITH ROUTER PLACEMENT", fontsize=12, fontweight='bold', pad=10)
    ax1.set_xlabel("X Coordinate (grid cells)", fontsize=10)
    ax1.set_ylabel("Y Coordinate (grid cells)", fontsize=10)
    ax1.grid(True, alpha=0.2, linestyle='-', color='gray')
    
    # ----- RIGHT: COVERAGE HEATMAP -----
    # Hide walls in heatmap
    coverage_display = np.where(grid == 1, np.nan, coverage_map)
    
    # Get actual signal range for proper color scaling
    signal_values = coverage_display[~np.isnan(coverage_display)]
    if len(signal_values) > 0:
        vmin, vmax = 0, 100  # Always use 0-100% scale for consistency
        print(f"  Signal range: {np.min(signal_values):.1f}% to {np.max(signal_values):.1f}%")
    else:
        vmin, vmax = 0, 100
    
    # Create heatmap with RdYlGn colormap
    im = ax2.imshow(coverage_display, cmap='RdYlGn', origin='lower', 
                   vmin=vmin, vmax=vmax, extent=[0, grid.shape[1], 0, grid.shape[0]])
    
    # Add router positions
    for i, (rx, ry) in enumerate(routers):
        ax2.plot(ry + 0.5, rx + 0.5, 'o', markersize=10, 
                color='black', markeredgecolor='white', markeredgewidth=2)
    
    ax2.set_title(f"WIFI SIGNAL COVERAGE\n{coverage_percent:.1f}% Coverage", 
                 fontsize=12, fontweight='bold', pad=10)
    ax2.set_xlabel("X Coordinate (grid cells)", fontsize=10)
    ax2.set_ylabel("Y Coordinate (grid cells)", fontsize=10)
    ax2.grid(True, alpha=0.2, linestyle='-', color='gray')
    
    # Colorbar with clear labels
    cbar = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label('Signal Strength (%)', fontsize=10)
    
    # Add colorbar ticks
    cbar.set_ticks([0, 25, 50, 75, 100])
    cbar.set_ticklabels(['0%', '25%', '50%', '75%', '100%'])
    
    # Statistics in clean box
    stats_text = (f'STATISTICS:\n'
                 f'Grid Size: {grid.shape[0]}×{grid.shape[1]}\n'
                 f'Free Cells: {free_cells}\n'
                 f'Covered Cells: {covered_cells}\n'
                 f'Coverage: {coverage_percent:.1f}%\n'
                 f'Routers: {len(routers)}')
    
    ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"   Saved: {save_path}")
    
    # Try to show window
    try:
        plt.show(block=False)
        plt.pause(0.5)
    except:
        print("  Plot window not available - image saved instead")
    
    return coverage_percent, fig

def plot_methods_comparison(comparison_data, save_path=None):
    """
    Compare different placement methods
    comparison_data: dict with method names as keys and 
                     (grid, coverage, routers, percent) as values
    """
    methods = list(comparison_data.keys())
    n_methods = len(methods)
    
    # Create subplots with MORE VERTICAL SPACE
    fig, axes = plt.subplots(2, n_methods, 
                            figsize=(4 * n_methods, 8),  # Increased height
                            gridspec_kw={'height_ratios': [1, 1.3]})  # More space for bottom
    
    if n_methods == 1:
        axes = axes.reshape(2, 1)
    
    # Plot each method
    for idx, (method_name, (grid, coverage, routers, percent)) in enumerate(comparison_data.items()):
        # Top: Floor plan
        ax_top = axes[0, idx]
        floor_display = grid.copy().astype(float)
        for rx, ry in routers:
            floor_display[rx, ry] = 0.5
        
        ax_top.imshow(floor_display, cmap='gray_r', origin='lower')
        
        # Add routers
        for rx, ry in routers:
            ax_top.plot(ry, rx, 'ro', markersize=6, markeredgecolor='black')
        
        # Method name
        ax_top.set_title(method_name.upper(), fontsize=11, fontweight='bold', pad=8)
        ax_top.set_xticks([])
        ax_top.set_yticks([])
        
        # Bottom: Coverage heatmap - FIXED COLOR SCALING
        ax_bottom = axes[1, idx]
        coverage_display = np.where(grid == 1, np.nan, coverage)
        
        # Use fixed 0-100% scale for all plots
        im = ax_bottom.imshow(coverage_display, cmap='RdYlGn', 
                             origin='lower', vmin=0, vmax=100)
        
        # Add routers
        for rx, ry in routers:
            ax_bottom.plot(ry, rx, 'ko', markersize=5, markeredgecolor='white')
        
        # Coverage percentage with colored background based on performance
        if percent > 66:
            color = 'green'
        elif percent > 33:
            color = 'orange'
        else:
            color = 'red'
            
        ax_bottom.text(0.5, -0.18, f"{percent:.1f}% Coverage",
                      transform=ax_bottom.transAxes, 
                      ha='center', fontsize=11, fontweight='bold',
                      bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.7))
        
        # Remove ticks
        ax_bottom.set_xticks([])
        ax_bottom.set_yticks([])
        ax_bottom.set_xticklabels([])
        ax_bottom.set_yticklabels([])
        
        # Add border
        for spine in ax_bottom.spines.values():
            spine.set_visible(True)
            spine.set_color('gray')
            spine.set_linewidth(0.5)
    
    # Main title
    plt.suptitle("WiFi Coverage Comparison - Person C's Real Grid", 
                 fontsize=14, fontweight='bold', y=1.02)
    
    # Add colorbar with signal strength labels
    if n_methods > 0:
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        cbar = plt.colorbar(im, cax=cbar_ax)
        cbar.set_label('Signal Strength (%)', fontsize=10, labelpad=15)
        
        # Add meaningful ticks
        cbar.set_ticks([0, 25, 50, 75, 100])
        cbar.set_ticklabels(['0%', '25%', '50%', '75%', '100%'])
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.05, 0.9, 0.98])
    plt.subplots_adjust(hspace=0.25, wspace=0.15)
    
    # Save if requested
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"   Saved: {save_path}")
    
    # Try to show window
    try:
        plt.show(block=False)
        plt.pause(0.5)
    except:
        print("   Comparison window not available - image saved instead")
    
    return fig

# ========== MAIN EXECUTION ==========
def main():
    """Main function to run the visualization"""
    print("═" * 50)
    print(" " * 10 + "WIFI ROUTER PLACEMENT OPTIMIZATION")
    print(" " * 15 + "VISUALIZATION MODULE")
    print(" " * 10 + "USING PERSON C'S REAL GRID DATA")
    print("═" * 50)
    
    # Create output directory
    os.makedirs("outputs/images", exist_ok=True)
    
    # Load Person C's real data
    print("\n Loading Person C's real grid data...")
    grid_data = load_person_c_data()
    
    if grid_data[0] is None:
        print(" Cannot proceed without Person C's grid data.")
        return
    
    grid, meta = grid_data
    height, width = grid.shape
    
    print(f"   Grid: {height}×{width} cells")
    print(f"   Free cells: {grid.size - np.sum(grid)}")
    
    # Generate realistic router positions for Person C's grid
    print("\n Generating router positions...")
    
    # Strategy 1: Smart placement (spread out)
    smart_routers = []
    
    # Try to place routers in different quadrants
    quadrant_positions = [
        (height//4, width//4),        # Top-left
        (height//4, 3*width//4),      # Top-right
        (3*height//4, width//4),      # Bottom-left
        (3*height//4, 3*width//4),    # Bottom-right
        (height//2, width//2),        # Center
    ]
    
    for y, x in quadrant_positions:
        # Check if position is valid and free
        if (0 <= y < height and 0 <= x < width and 
            grid[y, x] == 0 and (y, x) not in smart_routers):
            smart_routers.append((y, x))
            if len(smart_routers) >= 3:
                break
    
    # If not enough routers, find any free spots
    if len(smart_routers) < 3:
        free_spots = np.where(grid == 0)
        if len(free_spots[0]) > 0:
            indices = np.random.choice(len(free_spots[0]), min(3-len(smart_routers), len(free_spots[0])), replace=False)
            for idx in indices:
                pos = (free_spots[0][idx], free_spots[1][idx])
                if pos not in smart_routers:
                    smart_routers.append(pos)
    
    # Strategy 2: Random placement
    np.random.seed(42)
    random_routers = []
    free_spots = np.where(grid == 0)
    if len(free_spots[0]) >= 3:
        indices = np.random.choice(len(free_spots[0]), 3, replace=False)
        for idx in indices:
            random_routers.append((free_spots[0][idx], free_spots[1][idx]))
    
    # Strategy 3: Edge placement (along walls)
    edge_routers = []
    edge_positions = [
        (10, 10),
        (10, width - 10),
        (height - 10, 10),
        (height - 10, width - 10),
    ]
    
    for y, x in edge_positions:
        if (0 <= y < height and 0 <= x < width and 
            grid[y, x] == 0 and (y, x) not in edge_routers):
            edge_routers.append((y, x))
            if len(edge_routers) >= 3:
                break
    
    print(f"   • Smart routers: {smart_routers}")
    print(f"   • Random routers: {random_routers[:3] if random_routers else 'None'}")
    print(f"   • Edge routers: {edge_routers}")
    
    # Calculate coverage
    print("\n Calculating WiFi coverage with realistic signal decay...")
    
    # Smart placement
    smart_coverage = calculate_coverage_for_routers(grid, smart_routers)
    free_cells, covered_cells, smart_percent = calculate_coverage_stats(grid, smart_coverage)
    
    # Random placement
    if random_routers:
        cov_random = calculate_coverage_for_routers(grid, random_routers[:3])
        _, _, perc_random = calculate_coverage_stats(grid, cov_random)
    else:
        cov_random, perc_random = np.zeros_like(grid), 0
    
    # Edge placement
    if edge_routers:
        cov_edge = calculate_coverage_for_routers(grid, edge_routers)
        _, _, perc_edge = calculate_coverage_stats(grid, cov_edge)
    else:
        cov_edge, perc_edge = np.zeros_like(grid), 0
    
    print(f"   • Smart coverage: {smart_percent:.1f}%")
    print(f"   • Random coverage: {perc_random:.1f}%")
    print(f"   • Edge coverage: {perc_edge:.1f}%")
    
    # ========== SINGLE ANALYSIS ==========
    print("\n Creating detailed analysis plot...")
    single_coverage, fig1 = plot_single_analysis(
        grid, smart_coverage, smart_routers,
        save_path="outputs/images/heatmap_analysis_real.png"
    )
    
    # ========== METHODS COMPARISON ==========
    print("\n Generating methods comparison...")
    
    # Create comparison data
    comparison_data = {}
    
    if random_routers:
        comparison_data["Random"] = (grid, cov_random, random_routers[:3], perc_random)
    
    if edge_routers:
        comparison_data["Edge"] = (grid, cov_edge, edge_routers, perc_edge)
    
    comparison_data["Optimized"] = (grid, smart_coverage, smart_routers, smart_percent)
    
    # Create comparison plot
    fig2 = plot_methods_comparison(
        comparison_data,
        save_path="outputs/images/methods_comparison_real.png"
    )
    
    # ========== FINAL OUTPUT ==========
    print("\n" + "═" * 50)
    print(" VISUALIZATION COMPLETE")
    print("=" * 50)
    print(f"   • Grid: {grid.shape[0]}×{grid.shape[1]} cells")
    print(f"   • Free space: {grid.size - np.sum(grid)} cells")
    print(f" COVERAGE RESULTS:")
    
    if random_routers:
        print(f"   • Random Placement:    {perc_random:.1f}%")
    
    if edge_routers:
        print(f"   • Edge Placement:      {perc_edge:.1f}%")
    
    print(f"   • Optimized Placement: {smart_percent:.1f}%")
    
    if random_routers and perc_random > 0:
        improvement = smart_percent - perc_random
        print(f"   • Improvement: +{improvement:.1f}% over Random")
    
    print()
    print(f" IMAGES SAVED:")
    print(f"   • outputs/images/heatmap_analysis_real.png")
    print(f"   • outputs/images/methods_comparison_real.png")
    print()
    print("═" * 50)
    
    # Keep windows open
    try:
        plt.show(block=True)
    except:
        pass

# ========== ENTRY POINT ==========
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n Visualization interrupted by user")
    except Exception as e:
        print(f"\n Error: {e}")
    finally:
        print("\n Visualization module finished.")