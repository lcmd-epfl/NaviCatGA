# 4. Config-driven runs

Tutorials 1 to 3 built a solver directly in Python. Every parameter came from a hardcoded constructor call, which is what most launcher scripts end up doing.

`navicatGA.config.build_solver_from_yaml` builds the same solver from a YAML file instead, so the launcher script stays generic.

## The float example, as YAML

This reproduces [Tutorial 1](01_float.md)'s idea using `navicatGA`'s own bundled Hartmann6 example fitness function instead of a locally-defined one: a fitness function has to live in an importable module to be referenceable from YAML in the first place.

```yaml
# config.yaml
solver:
  type: float
  fitness_function: navicatGA.fitness_functions_float:fitness_function_float(10)
  params:
    n_genes: 6
    variables_limits: [0, 1]
    pop_size: 50
    max_gen: 50
    selection_strategy: tournament
    random_state: 420
    to_file: false
    verbose: false
```

```python
from navicatGA.config import build_solver_from_yaml

solver, config = build_solver_from_yaml("config.yaml")
result = solver.solve()
print(result.best_individual, result.best_fitness)
```

## Reference syntax

`fitness_function`, `chromosome_to_*`, and `scalarizer.class` are resolved by dotted `module.path:attr` reference:

- `module.path:attr`: use the attribute directly (a plain function).
- `module.path:attr()`: call it with no arguments and use the return value (for factories like `concatenate_list()`).
- `module.path:attr(10)` or `module.path:attr(target=350.0, function_number=6)`: call it with literal positional/keyword arguments (numbers, strings, lists, dicts, anything [`ast.literal_eval`](https://docs.python.org/3/library/ast.html#ast.literal_eval) accepts, not arbitrary expressions).

## Values that can't be static YAML

An alphabet read from a database, a starting population generated from a directory of structures, anything computed at runtime, isn't expressible in YAML. Build it in your launcher script and pass it as an extra keyword argument, which overrides/extends the YAML's `params` block:

```python
alphabet_list = my_alphabet_builder(...)
solver, config = build_solver_from_yaml("config.yaml", alphabet_list=alphabet_list)
```

## Starting your own project

Copy the three files in [`templates/`](https://github.com/lcmd-epfl/NaviCatGA/tree/master/templates). See [Configuration Templates](../config_templates.md) for the full walkthrough:

```bash
cp templates/smiles.yaml   my_project/config.yaml
cp templates/assembler_and_fitness.py my_project/
cp templates/launcher.py   my_project/
```

Next: [Tutorial 5: multi-objective optimization](05_multi_objective.md).
