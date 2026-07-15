# Configuration Templates

Ready-to-use templates live in the repository under [`templates/`](https://github.com/lcmd-epfl/NaviCatGA/tree/master/templates):

| File | Purpose |
| :--- | :--- |
| `templates/float.yaml` | Config template for `FloatGenAlgSolver` |
| `templates/smiles.yaml` | Config template for `SmilesGenAlgSolver` |
| `templates/selfies.yaml` | Config template for `SelfiesGenAlgSolver` |
| `templates/xyz.yaml` | Config template for `XYZGenAlgSolver` |
| `templates/launcher.py` | Generic runner: build a solver from a YAML config and call `solve()` |
| `templates/assembler_and_fitness.py` | Stub `assemble()`/`score()` functions to fill in for your problem |

Each YAML template lists every parameter that solver's `_BASE_DEFAULTS` accepts plus its solver-specific params, with the library defaults filled in and required fields marked.

## Usage

1. Copy the template matching your representation: `cp templates/smiles.yaml my_project/config.yaml`.
2. Copy `templates/assembler_and_fitness.py` into your project and fill in `assemble()` (chromosome → object to score) and `score()` (object → fitness). Point your YAML's `fitness_function`/`chromosome_to_*` at them by dotted reference, e.g. `fitness_function: my_project.assembler_and_fitness:score`.
3. Fill in `params` for your problem (`n_genes` is always required).
4. Copy `templates/launcher.py` into your project and run it:

```bash
python launcher.py --config my_project/config.yaml
```

Or build and run inline:

```python
from navicatGA.config import build_solver_from_yaml

solver, config = build_solver_from_yaml("my_project/config.yaml")
result = solver.solve()
```

## Runtime-computed values

Anything computed at runtime rather than static (an alphabet read from a database, a starting population generated from a directory of structures) isn't expressible in YAML. Build it in your launcher and pass it as an extra keyword argument, which overrides/extends the YAML's `params` block:

```python
alphabet_list = my_alphabet_builder(...)
solver, config = build_solver_from_yaml("my_project/config.yaml", alphabet_list=alphabet_list)
```

## Reference resolution

`fitness_function`, `chromosome_to_*`, and `scalarizer.class` are resolved by dotted `module.path:attr` reference:

- `module.path:attr`: the attribute used directly.
- `module.path:attr()`: call it with no arguments first (several of navicatGA's own bundled example assemblers are factories, e.g. `navicatGA.wrappers_smiles:chromosome_to_smiles()`).
- `module.path:attr(10)` / `module.path:attr(target=350.0, function_number=6)`: call it with literal positional/keyword arguments, e.g. `navicatGA.fitness_functions_selfies:fitness_function_selfies(1)`.

## Full walkthrough

See [Tutorial 7](tutorials/07_custom_solver.md) for a complete worked example: writing `assemble()`/`score()` from scratch, wiring them into a YAML config, and the two launcher patterns (runtime-computed values, multi-cycle runs with per-cycle logging).
