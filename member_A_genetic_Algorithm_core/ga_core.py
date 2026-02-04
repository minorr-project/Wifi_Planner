"""
Genetic Algorithm core for optimizing Wi-Fi router placement.

Assumptions / Integration:
- Grid format (Member C):
    grid[y][x] == 0 -> FREE
    grid[y][x] == 1 -> WALL
- Fitness function (Member B):
    fitness_function(routers, grid, beta=...)
  where routers is List[(x, y)]
"""

from __future__ import annotations

import random
from typing import List, Tuple, Optional, Dict, Any

# ------------------------------------------------------------
# Import fitness_function (edit ONLY this line if your file name differs)
# ------------------------------------------------------------
try:
    # Change 'signal_math' to the actual filename that contains fitness_function
    from member_B_signal_simulation_engine.signal_math import fitness_function
except ModuleNotFoundError:
    # Allow running this file directly without setting PYTHONPATH
    import os
    import sys

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from member_B_signal_simulation_engine.signal_math import fitness_function


Cell = Tuple[int, int]        # (x, y)
Individual = List[Cell]       # routers list


# ------------------------------------------------------------
# Helpers: support numpy arrays OR list-of-lists grids
# ------------------------------------------------------------
def _grid_shape(grid) -> Tuple[int, int]:
    """Return (H, W) for numpy array or list-of-lists."""
    try:
        # numpy array
        h, w = grid.shape
        return int(h), int(w)
    except Exception:
        # list-of-lists
        h = len(grid)
        w = len(grid[0]) if h else 0
        return h, w


def _grid_get(grid, y: int, x: int) -> int:
    """Get cell value (0/1) for numpy array or list-of-lists."""
    try:
        return int(grid[y, x])  # numpy
    except Exception:
        return int(grid[y][x])  # list-of-lists


# ------------------------------------------------------------
# Candidate extraction (FREE cells only)
# ------------------------------------------------------------
def get_free_cells(grid) -> List[Cell]:
    """Collect all FREE cells (value==0) as candidate router locations."""
    free_cells: List[Cell] = []
    H, W = _grid_shape(grid)

    for y in range(H):
        for x in range(W):
            if _grid_get(grid, y, x) == 0:
                free_cells.append((x, y))

    return free_cells


# ------------------------------------------------------------
# GA operators
# ------------------------------------------------------------
def random_individual(
    num_routers: int,
    candidates: List[Cell],
    rng: random.Random,
) -> Individual:
    """Create a random individual with unique router positions."""
    if num_routers > len(candidates):
        raise ValueError("num_routers exceeds number of candidate cells")
    return rng.sample(candidates, num_routers)


def tournament_selection(
    population: List[Individual],
    fitnesses: List[float],
    k: int,
    rng: random.Random,
) -> Individual:
    """Pick the best among k randomly chosen individuals."""
    chosen = rng.sample(range(len(population)), k)
    best_idx = max(chosen, key=lambda i: fitnesses[i])
    return population[best_idx]


def crossover(
    parent_a: Individual,
    parent_b: Individual,
    rng: random.Random,
) -> Tuple[Individual, Individual]:
    """One-point crossover; ensures same length output."""
    if len(parent_a) != len(parent_b):
        raise ValueError("Parents must have same length")
    if len(parent_a) < 2:
        return parent_a[:], parent_b[:]

    cut = rng.randint(1, len(parent_a) - 1)
    child1 = parent_a[:cut] + parent_b[cut:]
    child2 = parent_b[:cut] + parent_a[cut:]
    return child1, child2


def repair_individual(
    indiv: Individual,
    candidates: List[Cell],
    rng: random.Random,
) -> Individual:
    """
    Ensure all router positions are unique and valid candidates.
    If duplicates exist, replace them with unused candidates.
    """
    seen = set()
    unique: Individual = []

    for cell in indiv:
        if cell in candidates and cell not in seen:
            unique.append(cell)
            seen.add(cell)

    # Fill missing routers if any were removed
    if len(unique) < len(indiv):
        remaining = [c for c in candidates if c not in seen]
        needed = len(indiv) - len(unique)
        if needed > 0:
            if needed > len(remaining):
                # fallback: allow reusing remaining if extremely constrained
                unique.extend(rng.choices(candidates, k=needed))
            else:
                unique.extend(rng.sample(remaining, needed))

    return unique


def mutate(
    indiv: Individual,
    candidates: List[Cell],
    mutation_rate: float,
    rng: random.Random,
) -> Individual:
    """Randomly replace router positions with new candidates."""
    new_indiv = indiv[:]
    for i in range(len(new_indiv)):
        if rng.random() < mutation_rate:
            new_indiv[i] = rng.choice(candidates)
    return new_indiv


# ------------------------------------------------------------
# Fitness evaluation
# ------------------------------------------------------------
def evaluate_population(
    population: List[Individual],
    grid,
    beta: float,
) -> List[float]:
    """Compute fitness for each individual."""
    return [fitness_function(indiv, grid, beta=beta) for indiv in population]


# ------------------------------------------------------------
# Main GA loop
# ------------------------------------------------------------
def run_ga(
    grid,
    num_routers: int,
    population_size: int = 30,
    generations: int = 50,
    crossover_rate: float = 0.8,
    mutation_rate: float = 0.2,
    elite_size: int = 2,
    tournament_k: int = 3,
    beta: float = 0.0,
    candidates: Optional[List[Cell]] = None,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Run GA and return best solution with history.

    Returns dict:
      best_routers: List[(x, y)]
      best_fitness: float
      history: List[float] (best fitness per generation)
    """
    rng = random.Random(seed)

    if candidates is None:
        candidates = get_free_cells(grid)
    if not candidates:
        raise ValueError("No FREE candidate cells available (grid might be wrong).")

    population = [
        random_individual(num_routers, candidates, rng)
        for _ in range(population_size)
    ]

    history: List[float] = []
    best_routers: Individual = []
    best_fitness = float("-inf")

    for _gen in range(generations):
        fitnesses = evaluate_population(population, grid, beta)

        # Track best in this generation
        gen_best_idx = max(range(len(population)), key=lambda i: fitnesses[i])
        gen_best_fit = fitnesses[gen_best_idx]
        if gen_best_fit > best_fitness:
            best_fitness = gen_best_fit
            best_routers = population[gen_best_idx][:]

        history.append(gen_best_fit)

        # Elitism: keep top elite_size
        elite_indices = sorted(
            range(len(population)),
            key=lambda i: fitnesses[i],
            reverse=True
        )[:elite_size]

        new_population: List[Individual] = [population[i][:] for i in elite_indices]

        # Create the rest
        while len(new_population) < population_size:
            parent_a = tournament_selection(population, fitnesses, tournament_k, rng)
            parent_b = tournament_selection(population, fitnesses, tournament_k, rng)

            if rng.random() < crossover_rate:
                child1, child2 = crossover(parent_a, parent_b, rng)
            else:
                child1, child2 = parent_a[:], parent_b[:]

            child1 = mutate(child1, candidates, mutation_rate, rng)
            child2 = mutate(child2, candidates, mutation_rate, rng)

            child1 = repair_individual(child1, candidates, rng)
            child2 = repair_individual(child2, candidates, rng)

            new_population.append(child1)
            if len(new_population) < population_size:
                new_population.append(child2)

        population = new_population

    return {
        "best_routers": best_routers,
        "best_fitness": best_fitness,
        "history": history,
    }


# ------------------------------------------------------------
# Smoke test
# ------------------------------------------------------------
if __name__ == "__main__":
    try:
        import numpy as np

        # 0=free, 1=wall
        grid = np.zeros((20, 20), dtype=int)
        grid[:, 10] = 1  # vertical wall

        result = run_ga(grid, num_routers=2, generations=20, seed=42)
        print("Best routers:", result["best_routers"])
        print("Best fitness:", result["best_fitness"])
        print("History (first 5):", result["history"][:5])
    except Exception as e:
        print("GA smoke test failed:", e)
