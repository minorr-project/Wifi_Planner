# Member A — Genetic Algorithm Core

This module runs the Genetic Algorithm (GA) to optimize router placement.
It depends on the signal simulation fitness function in:

`member_B_signal_simulation_engine/signal_physics.py`

## Main file

- `ga_core.py` — GA loop + selection + crossover + mutation

## Expected grid format

The GA expects a `grid` where each cell is a string:

- `"FREE"` for free space
- `"WALL"`, `"DOOR"`, `"WINDOW"` for obstacles

This matches the grid format used by the signal simulation code.

## Quick usage (example)

```python
from member_A_genetic_Algorithm_core.ga_core import run_ga

# Example: 20x20 free grid
grid = [["FREE" for _ in range(20)] for _ in range(20)]

result = run_ga(
    grid,
    num_routers=3,
    generations=30,
    population_size=20,
    seed=42,
)

print(result["best_routers"])
print(result["best_fitness"])
```

## Running the smoke test

You can run the module directly without setting `PYTHONPATH`:

```bash
python member_A_genetic_Algorithm_core/ga_core.py
```

## Output

`run_ga()` returns a dict:

- `best_routers`: list of (x, y) router positions
- `best_fitness`: best coverage-based fitness
- `history`: best fitness per generation
