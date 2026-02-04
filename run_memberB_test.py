from common.io import load_grid
from member_B_signal_simulation_engine.signal_math import (
    fitness_function,
    calibrate_cell_size,
    coverage_metrics,
)
from member_A_genetic_Algorithm_core.ga_core import run_ga


def main():
    # 1) Load grid (Member C)
    grid = load_grid()

    # 2) Calibrate cell size (Member B)
    cell_m = calibrate_cell_size(grid)
    print("Calibrated CELL_SIZE_METERS =", cell_m)

    # 3) QUICK sanity check: a router at the grid center (always in-bounds)
    H, W = grid.shape
    center_router = (W // 2, H // 2)

    cov, avg = coverage_metrics([center_router], grid)
    print("Center router:", center_router)
    print("coverage%:", cov, "avg:", avg)

    score = fitness_function([center_router], grid)
    print("Fitness score (center):", score)

    # 4) RUN GA (Member A) on the SAME grid
    result = run_ga(
        grid,
        num_routers=2,        # try 1, 2, 3
        population_size=30,
        generations=50,
        seed=42,
    )

    print("\n=== GA RESULT ===")
    print("Best routers:", result["best_routers"])
    print("Best fitness:", result["best_fitness"])


if __name__ == "__main__":
    main()
