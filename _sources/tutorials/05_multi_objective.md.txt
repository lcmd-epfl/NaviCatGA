# 5. Multi-objective optimization

Any `fitness_function` can return a tuple of scores instead of one. The GA still needs a single number to rank chromosomes against each other, so you also pass a **`scalarizer`**: an object with a `.scalarize(fitness_array)` method that combines the tuple into one value per chromosome.

navicatGA doesn't implement a scalarizer itself; [`matter-chimera`](https://github.com/aspuru-guzik-group/chimera) is the recommended one, but anything with a compatible `.scalarize()` works.

## Two competing objectives

```python
from chimera import Chimera
from navicatGA.float_solver import FloatGenAlgSolver

def two_objectives(x):
    obj1 = -sum((xi - 0.2) ** 2 for xi in x)  # maximize: peak near 0.2
    obj2 = -sum((xi - 0.8) ** 2 for xi in x)  # maximize: peak near 0.8
    return obj1, obj2                          # order matches `goals` below

scalarizer = Chimera(tolerances=[0.05, 0.05], goals=["max", "max"])

solver = FloatGenAlgSolver(
    n_genes=3,
    fitness_function=two_objectives,
    variables_limits=(0, 1),
    scalarizer=scalarizer,
    pop_size=30,
    max_gen=30,
    random_state=1,
    to_file=False,
    verbose=False,
)
result = solver.solve()
print(result.best_individual)  # [0.294 0.280 0.299] - Chimera's tolerance-weighted compromise
print(result.best_fitness)     # scalarized score, not either raw objective
```

`tolerances` sets how much each objective is allowed to trade off against the higher-priority ones (Chimera objectives are prioritized left-to-right; see the [Chimera paper](https://pubs.rsc.org/en/content/articlelanding/2018/sc/c8sc02239a) for the full algorithm); `goals` (`"max"`/`"min"`) must have one entry per fitness output, in the same order the tuple is returned.

## Reading the un-scalarized scores back

`GenAlgSolver.calculate_fitness` keeps both the scalarized fitness (used for selection) and the raw per-objective values: `solver.fitness_` is scalarized, `solver.printable_fitness` is the raw per-objective array (one row per chromosome, one column per objective):

```python
print(solver.fitness_[0])           # 1.0 - scalarized, used for selection
print(solver.printable_fitness[0])  # [-0.00878484 -1.11968321] - the two raw objectives
```

Useful for logging every objective per candidate to a file even though only the scalarized value drives selection.

## In a YAML config

```yaml
solver:
  type: smiles
  fitness_function: your_project.fitness:multi_objective_score
  scalarizer:
    class: chimera:Chimera
    kwargs:
      tolerances: [0.25, 0.1, 0.25]
      goals: [max, max, min]
  params:
    n_genes: 8
    # ...
```

See [Tutorial 7](07_custom_solver.md) for how to write `multi_objective_score` (and any other custom assembler/fitness function) from scratch.

Next: [Tutorial 6: 3D structures with AaronTools](06_xyz.md).
