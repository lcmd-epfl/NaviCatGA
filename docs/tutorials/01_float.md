# 1. Continuous optimization

`FloatGenAlgSolver` is the simplest solver (genes are plain floats, no chemistry involved) and the best place to learn the pattern every other solver follows.

## The two hooks every solver needs

`GenAlgSolver` (the base class) runs the entire GA loop but has no idea what a chromosome *means*. You supply that meaning through two callables:

- **`assembler`**: turns a raw chromosome (an array of genes) into the object that gets scored. `FloatGenAlgSolver`'s default assembler, `make_array()`, is a no-op: the chromosome *is* the array.
- **`fitness_function`**: scores the assembled object. This is the one function you always have to write yourself.

## A minimal run

```python
from navicatGA.float_solver import FloatGenAlgSolver

def sphere(x):
    """Maximize -sum((x_i - 0.5)^2): a peak at x = [0.5, 0.5, 0.5]."""
    return -sum((xi - 0.5) ** 2 for xi in x)

solver = FloatGenAlgSolver(
    n_genes=3,
    fitness_function=sphere,
    variables_limits=(0, 1),      # shared (min, max) for every gene
    pop_size=50,
    max_gen=100,
    selection_strategy="tournament",
    random_state=42,               # reproducible run
    to_file=False,                 # skip writing output.log for this quick example
    verbose=False,
)
result = solver.solve()
print(result.best_individual, result.best_fitness)
# [0.50143372 0.50003259 0.50000241] -2.0566351926863767e-06
```

`solve()` returns a `GAResult` namedtuple (`best_individual`, `best_fitness`, `best_pfitness`, `population`, `fitness`, `generations`, `runtime`). The same values are also set on the solver as `solver.best_individual_`, `solver.best_fitness_`, etc, for code that predates `GAResult`.

## GA-mechanics parameters (shared by every solver)

These aren't specific to `FloatGenAlgSolver`: every solver forwards them to `GenAlgSolver` via `**base_kwargs`, so the same names work everywhere:

| Param | Meaning |
| :--- | :--- |
| `pop_size` | Number of chromosomes per generation. |
| `mutation_rate` | Fraction of genes mutated per generation. |
| `selection_rate` | Fraction of the population kept as parents. |
| `selection_strategy` | `roulette_wheel` \| `tournament` \| `two_by_two` \| `random` \| `boltzmann`. |
| `n_crossover_points` | Number of crossover cut points. |
| `max_gen` / `max_conv` | Hard generation cap / generations with an unchanged best fitness before stopping early. |
| `excluded_genes` | Gene indices held fixed (never mutated/crossed). |
| `lru_cache` | Cache fitness by chromosome; needs `assembler`'s output to be hashable. |
| `prune_duplicates` | Drop and refill duplicate chromosomes each generation. |
| `scalarizer` | Combine multiple fitness outputs into one; see [Tutorial 5](05_multi_objective.md). |

See `templates/float.yaml` for the full list with defaults filled in.

## `variables_limits`

- A single `(min, max)` pair applies to every gene (as above).
- A list of `n_genes` `(min, max)` pairs sets per-gene bounds, e.g. `[(0, 1), (-10, 10), (0, 100)]`.
- Integer bounds (`(0, 10)` with both ints) sample integers instead of floats.

Next: [Tutorial 2: SMILES fragment optimization](02_smiles.md), where the assembler actually does something.
