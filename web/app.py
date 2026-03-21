"""
WiFi Planner Web UI - Flask Application
"""

import os
import sys
import json
import uuid
import random
import base64
from io import BytesIO

import numpy as np
from flask import Flask, render_template, request, jsonify, send_file

# Add project root to path for imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import project modules
from member_A_genetic_Algorithm_core.ga_core import run_ga, get_free_cells
from member_B_signal_simulation_engine.signal_math import (
    calibrate_cell_size,
    coverage_metrics,
    best_signal,
    S_threshold,
)

# Flask app setup
app = Flask(__name__)
app.config['SECRET_KEY'] = 'wifi-planner-secret-key'

# Store results in memory (for demo purposes)
results_store = {}


# ============================================================
# Utility Functions
# ============================================================
def load_grid(step=8):
    """Load and preprocess the grid."""
    grid_path = os.path.join(PROJECT_ROOT, "grid.npy")
    meta_path = os.path.join(PROJECT_ROOT, "grid_meta.json")
    
    if not os.path.exists(grid_path):
        raise FileNotFoundError("grid.npy not found. Run the DXF pipeline first.")
    
    grid = np.load(grid_path)
    original_shape = grid.shape
    
    # Downsample for faster processing
    grid = grid[::step, ::step].astype(np.uint8)
    
    meta = {"original_shape": original_shape, "downsample_step": step}
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta.update(json.load(f))
    
    return grid, meta


def run_random_baseline(grid, num_routers=2, seed=42):
    """Random router placement."""
    rng = random.Random(seed)
    candidates = get_free_cells(grid)
    if len(candidates) < num_routers:
        raise ValueError("Not enough free cells for router placement.")
    return rng.sample(candidates, num_routers)


def run_uniform_baseline(grid, num_routers=2):
    """Uniform grid-based router placement."""
    H, W = grid.shape
    candidates = [
        (W // 4, H // 4),
        (3 * W // 4, 3 * H // 4),
        (W // 4, 3 * H // 4),
        (3 * W // 4, H // 4),
        (W // 2, H // 2),
    ]
    
    routers = []
    for (x, y) in candidates:
        if 0 <= x < W and 0 <= y < H and grid[y, x] == 0:
            routers.append((x, y))
        if len(routers) == num_routers:
            return routers
    
    # Fallback to free cells
    free_cells = get_free_cells(grid)
    for c in free_cells:
        if c not in routers:
            routers.append(c)
        if len(routers) == num_routers:
            return routers
    
    return routers[:num_routers]


def make_heatmap(grid, routers):
    """Generate signal heatmap data."""
    H, W = grid.shape
    heat = np.full((H, W), np.nan, dtype=float)
    
    for y in range(H):
        for x in range(W):
            if grid[y, x] == 1:
                continue
            heat[y, x] = best_signal((x, y), routers, grid)
    
    return heat


def generate_heatmap_image(grid, routers, title="Signal Heatmap"):
    """Generate a heatmap image and return as base64."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    heat = make_heatmap(grid, routers)
    heat_clipped = np.clip(heat, -95, -30)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot heatmap
    im = ax.imshow(heat_clipped, origin='lower', cmap='RdYlGn')
    
    # Plot router positions
    for i, (x, y) in enumerate(routers):
        ax.plot(x, y, 'ko', markersize=12, markeredgecolor='white', markeredgewidth=2)
        ax.text(x, y + 2, f'R{i+1}', ha='center', va='bottom', fontsize=10,
                fontweight='bold', color='black',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    ax.set_title(f'{title}\nThreshold: {S_threshold} dBm', fontsize=14)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Signal Strength (dBm)')
    
    plt.tight_layout()
    
    # Save to bytes
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def generate_comparison_image(grid, methods):
    """Generate comparison image for multiple methods."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    names = list(methods.keys())
    n = len(names)
    
    fig, axes = plt.subplots(2, n, figsize=(5 * n, 8))
    if n == 1:
        axes = np.array([[axes[0]], [axes[1]]])
    
    last_im = None
    
    for i, name in enumerate(names):
        routers = methods[name]
        
        # Top: floorplan + routers
        ax_top = axes[0, i]
        ax_top.imshow(grid, cmap='gray_r', origin='lower')
        for j, (x, y) in enumerate(routers):
            ax_top.plot(x, y, 'ro', markersize=8, markeredgecolor='black')
            ax_top.text(x, y + 1, f'R{j+1}', ha='center', va='bottom',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
        ax_top.set_title(f'{name} (Placement)')
        ax_top.set_xticks([])
        ax_top.set_yticks([])
        
        # Bottom: signal heatmap
        ax_bot = axes[1, i]
        heat = make_heatmap(grid, routers)
        heat = np.clip(heat, -95, -30)
        last_im = ax_bot.imshow(heat, origin='lower', cmap='RdYlGn')
        for (x, y) in routers:
            ax_bot.plot(x, y, 'ko', markersize=6, markeredgecolor='white')
        ax_bot.set_title(f'{name} (Signal dBm)')
        ax_bot.set_xticks([])
        ax_bot.set_yticks([])
    
    if last_im is not None:
        cbar = fig.colorbar(last_im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
        cbar.set_label('Signal (dBm)')
    
    plt.tight_layout()
    
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    
    return base64.b64encode(buf.getvalue()).decode('utf-8')


# ============================================================
# Routes
# ============================================================
@app.route('/')
def index():
    """Main page."""
    return render_template('index.html')


@app.route('/api/grid-info')
def grid_info():
    """Get grid metadata."""
    try:
        grid, meta = load_grid()
        calibrate_cell_size(grid)
        
        return jsonify({
            'success': True,
            'shape': grid.shape,
            'original_shape': meta.get('original_shape'),
            'downsample_step': meta.get('downsample_step'),
            'wall_cells': int(grid.sum()),
            'free_cells': int(grid.size - grid.sum()),
        })
    except FileNotFoundError as e:
        return jsonify({'success': False, 'error': str(e)}), 404


@app.route('/api/grid-preview')
def grid_preview():
    """Get grid preview image."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        grid, _ = load_grid()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(grid, cmap='gray_r', origin='lower')
        ax.set_title('Floor Plan Grid')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        plt.tight_layout()
        
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=150)
        buf.seek(0)
        plt.close(fig)
        
        return jsonify({
            'success': True,
            'image': base64.b64encode(buf.getvalue()).decode('utf-8')
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/optimize', methods=['POST'])
def optimize():
    """Run optimization."""
    try:
        data = request.get_json() or {}
        
        num_routers = int(data.get('num_routers', 2))
        generations = int(data.get('generations', 20))
        population_size = int(data.get('population_size', 30))
        method = data.get('method', 'ga')
        seed = int(data.get('seed', 42))
        
        # Load grid
        grid, meta = load_grid()
        calibrate_cell_size(grid)
        
        results = {}
        
        if method == 'compare':
            # Run all methods
            methods_data = {}
            
            # Random
            random_routers = run_random_baseline(grid, num_routers, seed)
            cov_r, avg_r = coverage_metrics(random_routers, grid)
            methods_data['Random'] = {
                'routers': random_routers,
                'coverage': round(cov_r, 2),
                'avg_signal': round(avg_r, 2)
            }
            
            # Uniform
            uniform_routers = run_uniform_baseline(grid, num_routers)
            cov_u, avg_u = coverage_metrics(uniform_routers, grid)
            methods_data['Uniform'] = {
                'routers': uniform_routers,
                'coverage': round(cov_u, 2),
                'avg_signal': round(avg_u, 2)
            }
            
            # GA
            ga_result = run_ga(grid, num_routers=num_routers, 
                             generations=generations,
                             population_size=population_size, seed=seed)
            ga_routers = ga_result['best_routers']
            cov_g, avg_g = coverage_metrics(ga_routers, grid)
            methods_data['GA Optimized'] = {
                'routers': ga_routers,
                'coverage': round(cov_g, 2),
                'avg_signal': round(avg_g, 2),
                'fitness': round(ga_result['best_fitness'], 2)
            }
            
            # Generate comparison image
            router_dict = {k: v['routers'] for k, v in methods_data.items()}
            image = generate_comparison_image(grid, router_dict)
            
            results = {
                'method': 'compare',
                'methods': methods_data,
                'image': image
            }
            
        else:
            # Single method
            if method == 'random':
                routers = run_random_baseline(grid, num_routers, seed)
                title = 'Random Placement'
            elif method == 'uniform':
                routers = run_uniform_baseline(grid, num_routers)
                title = 'Uniform Placement'
            else:  # ga
                ga_result = run_ga(grid, num_routers=num_routers,
                                  generations=generations,
                                  population_size=population_size, seed=seed)
                routers = ga_result['best_routers']
                results['fitness'] = round(ga_result['best_fitness'], 2)
                results['history'] = ga_result['history']
                title = 'GA Optimized Placement'
            
            coverage, avg_signal = coverage_metrics(routers, grid)
            image = generate_heatmap_image(grid, routers, title)
            
            results.update({
                'method': method,
                'routers': [{'x': int(x), 'y': int(y)} for x, y in routers],
                'coverage': round(coverage, 2),
                'avg_signal': round(avg_signal, 2),
                'image': image,
                'threshold': S_threshold
            })
        
        # Store result
        job_id = str(uuid.uuid4())[:8]
        results_store[job_id] = results
        results['job_id'] = job_id
        
        return jsonify({'success': True, 'results': results})
        
    except Exception as e:
        import traceback
        return jsonify({
            'success': False, 
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@app.route('/api/results/<job_id>')
def get_results(job_id):
    """Get stored results."""
    if job_id in results_store:
        return jsonify({'success': True, 'results': results_store[job_id]})
    return jsonify({'success': False, 'error': 'Results not found'}), 404


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    print("Starting WiFi Planner Web UI...")
    print(f"Project root: {PROJECT_ROOT}")
    print("Open http://localhost:5000 in your browser")
    app.run(debug=True, host='0.0.0.0', port=5000)
