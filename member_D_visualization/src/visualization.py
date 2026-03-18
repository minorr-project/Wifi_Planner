"""
WiFi Router Placement Optimization - Visualization Module
Person D: Heatmap and Comparison Plots
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# ========== UTILITY FUNCTIONS ==========
def create_dummy_data():
    """Create realistic test data for WiFi coverage"""
    grid_size = 10
    grid = np.zeros((grid_size, grid_size))
    
    # Add walls (1 = wall, 0 = free space)
    grid[3, :] = 1   # Horizontal wall at row 3
    grid[:, 7] = 1   # Vertical wall at column 7
    
    # Router positions (x, y coordinates)
    routers = [(2, 2), (8, 5)]
    
    # Calculate realistic signal coverage
    coverage = np.zeros((grid_size, grid_size))
    for rx, ry in routers:
        for x in range(grid_size):
            for y in range(grid_size):
                if grid[x, y] == 0:  # Only free space
                    distance = np.sqrt((x - rx)**2 + (y - ry)**2)
                    if distance <= 3.5:  # Router range
                        # Signal strength decays with distance
                        signal = max(0, 1.2 - (distance / 3))
                        coverage[x, y] = max(coverage[x, y], signal)
    
    return grid, coverage, routers

def calculate_coverage_stats(grid, coverage_map, threshold=0.3):
    """Calculate coverage statistics"""
    free_cells = np.sum(grid == 0)
    covered_cells = np.sum(coverage_map > threshold)
    coverage_percent = (covered_cells / free_cells) * 100 if free_cells > 0 else 0
    return free_cells, covered_cells, coverage_percent

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
    
    # Add router coverage circles
    for rx, ry in routers:
        circle = plt.Circle((ry + 0.5, rx + 0.5), 3.5, 
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
    ax1.set_xlabel("X Coordinate (meters)", fontsize=10)
    ax1.set_ylabel("Y Coordinate (meters)", fontsize=10)
    ax1.grid(True, alpha=0.2, linestyle='-', color='gray')
    
    # ----- RIGHT: COVERAGE HEATMAP -----
    # Hide walls in heatmap
    coverage_display = np.where(grid == 1, np.nan, coverage_map)
    
    im = ax2.imshow(coverage_display, cmap='RdYlGn', origin='lower', 
                   vmin=0, vmax=1, extent=[0, grid.shape[1], 0, grid.shape[0]])
    
    # Add router positions
    for i, (rx, ry) in enumerate(routers):
        ax2.plot(ry + 0.5, rx + 0.5, 'o', markersize=10, 
                color='black', markeredgecolor='white', markeredgewidth=2)
    
    ax2.set_title(f"WIFI SIGNAL COVERAGE\n{coverage_percent:.1f}% Coverage", 
                 fontsize=12, fontweight='bold', pad=10)
    ax2.set_xlabel("X Coordinate (meters)", fontsize=10)
    ax2.set_ylabel("Y Coordinate (meters)", fontsize=10)
    ax2.grid(True, alpha=0.2, linestyle='-', color='gray')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label('Signal Strength', fontsize=10)
    
    # Statistics in clean box
    stats_text = (f'STATISTICS:\n'
                 f'Grid Size: {grid.shape[0]}×{grid.shape[1]} m\n'
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
        print(f"  ✓ Saved: {save_path}")
    
    # Try to show window
    try:
        plt.show(block=False)
        plt.pause(0.5)
    except:
        print("  ℹ Plot window not available - image saved instead")
    
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
        
        # Method name - SIMPLE, NO EXTRA TEXT
        ax_top.set_title(method_name.upper(), fontsize=11, fontweight='bold', pad=8)
        ax_top.set_xticks([])
        ax_top.set_yticks([])
        
        # Bottom: Coverage heatmap
        ax_bottom = axes[1, idx]
        coverage_display = np.where(grid == 1, np.nan, coverage)
        
        im = ax_bottom.imshow(coverage_display, cmap='RdYlGn', 
                             origin='lower', vmin=0, vmax=1)
        
        # Add routers
        for rx, ry in routers:
            ax_bottom.plot(ry, rx, 'ko', markersize=5, markeredgecolor='white')
        
        # Coverage percentage - POSITIONED PROPERLY WITH MORE SPACE
        # Position at bottom with EXTRA padding
        ax_bottom.text(0.5, -0.18, f"{percent:.1f}% Coverage",  # Moved further down
                      transform=ax_bottom.transAxes, 
                      ha='center', fontsize=11, fontweight='bold')
        
        # Remove ALL ticks and labels to prevent overlap
        ax_bottom.set_xticks([])
        ax_bottom.set_yticks([])
        ax_bottom.set_xticklabels([])
        ax_bottom.set_yticklabels([])
        
        # Add thin border for clarity
        for spine in ax_bottom.spines.values():
            spine.set_visible(True)
            spine.set_color('gray')
            spine.set_linewidth(0.5)
    
    # Remove the main title to reduce clutter
    plt.suptitle("", fontsize=0)
    
    # Add colorbar with PROPER positioning
    if n_methods > 0:
        # Position colorbar carefully
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        cbar = plt.colorbar(im, cax=cbar_ax)
        cbar.set_label('Signal', fontsize=10, labelpad=10)
    
    # CRITICAL: Adjust layout to prevent ANY overlap
    plt.tight_layout(rect=[0, 0.05, 0.9, 0.98])  # Increased bottom margin
    
    # Additional spacing
    plt.subplots_adjust(hspace=0.25, wspace=0.15)
    
    # Save if requested
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"  ✓ Saved: {save_path}")
    
    # Try to show window
    try:
        plt.show(block=False)
        plt.pause(0.5)
    except:
        print("  ℹ Comparison window not available - image saved instead")
    
    return fig

# ========== MAIN EXECUTION ==========
def main():
    """Main function to run the visualization"""
    print("═" * 50)
    print(" " * 10 + "WIFI ROUTER PLACEMENT OPTIMIZATION")
    print(" " * 15 + "VISUALIZATION MODULE")
    print("═" * 50)
    
    # Create output directory
    os.makedirs("outputs/images", exist_ok=True)
    
    # Generate test data
    print("\n Generating test environment...")
    grid, coverage_map, routers = create_dummy_data()
    free_cells, covered_cells, coverage_percent = calculate_coverage_stats(grid, coverage_map)
    
    print(f"   • Grid: {grid.shape[0]}×{grid.shape[1]} meters")
    print(f"   • Routers: {len(routers)} at positions {routers}")
    print(f"   • Expected coverage: {coverage_percent:.1f}%")
    
    # ========== SINGLE ANALYSIS ==========
    print("\n Creating detailed analysis plot...")
    single_coverage, fig1 = plot_single_analysis(
        grid, coverage_map, routers,
        save_path="outputs/images/heatmap_analysis.png"
    )
    
    # ========== METHODS COMPARISON ==========
    print("\n Generating methods comparison...")
    
    # Function to calculate coverage for specific router positions
    def calculate_for_routers(router_positions):
        coverage = np.zeros_like(grid, dtype=float)
        for rx, ry in router_positions:
            for x in range(grid.shape[0]):
                for y in range(grid.shape[1]):
                    if grid[x, y] == 0:
                        dist = np.sqrt((x - rx)**2 + (y - ry)**2)
                        if dist <= 3.5:
                            signal = max(0, 1.2 - (dist / 3))
                            coverage[x, y] = max(coverage[x, y], signal)
        _, _, percent = calculate_coverage_stats(grid, coverage)
        return coverage, percent
    
    # Calculate REAL coverage for different methods
    cov_random, perc_random = calculate_for_routers([(1, 1), (6, 6)])
    cov_uniform, perc_uniform = calculate_for_routers([(3, 3), (7, 7)])
    
    # Create comparison data
    comparison_data = {
        "Random": (grid, cov_random, [(1, 1), (6, 6)], perc_random),
        "Uniform": (grid, cov_uniform, [(3, 3), (7, 7)], perc_uniform),
        "Optimized": (grid, coverage_map, routers, coverage_percent)
    }
    
    # Create comparison plot
    fig2 = plot_methods_comparison(
        comparison_data,
        save_path="outputs/images/methods_comparison.png"
    )
    
    # ========== FINAL OUTPUT ==========
    print("\n" + "═" * 50)
    print(" VISUALIZATION COMPLETE")
    print("═" * 50)
    print(f" COVERAGE RESULTS:")
    print(f"   • Random Placement:    {perc_random:.1f}%")
    print(f"   • Uniform Placement:   {perc_uniform:.1f}%")
    print(f"   • Optimized Placement: {coverage_percent:.1f}%")
    print(f"   • Improvement: +{coverage_percent - perc_random:.1f}% over Random")
    print()
    print(f" IMAGES SAVED:")
    print(f"   • outputs/images/heatmap_analysis.png")
    print(f"   • outputs/images/methods_comparison.png")
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