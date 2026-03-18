# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

import os
import sys

# Add project root to Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from member_A_genetic_Algorithm_core.ga_core import run_ga
from member_D_visualization.visualization_integrated import visualize_all
from utils import load_grid, preprocess, run_random_baseline, run_uniform_baseline, print_metrics

def main():
    grid = load_grid()
    grid = preprocess(grid)

    random_routers = run_random_baseline(grid)
    uniform_routers = run_uniform_baseline(grid)

    ga_result = run_ga(
        grid,
        num_routers=2,
        generations=20,
        seed=42
    )

    print_metrics(random_routers, uniform_routers, ga_result)

    visualize_all(
        grid,
        {
            "Random": random_routers,
            "Uniform": uniform_routers,
            "GA Optimized": ga_result["best_routers"]
        }
    )


if __name__ == "__main__":
    main()
