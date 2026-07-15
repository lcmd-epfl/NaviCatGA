# Templates

- One YAML config template per solver (`float.yaml`, `smiles.yaml`, `selfies.yaml`, `xyz.yaml`), each listing every parameter that solver's `_BASE_DEFAULTS` accepts plus its solver-specific params, with the library defaults filled in and required fields marked.
- `launcher.py` — a generic runner: build a solver from a YAML config and call `solve()`.
- `assembler_and_fitness.py` — stub `assemble()`/`score()` functions to fill in for your problem; this is what your YAML's `chromosome_to_*`/`fitness_function` references point at.

## Usage

1. Copy the template that matches your representation into your project, e.g. `cp templates/smiles.yaml my_project/config.yaml`.
2. Copy `templates/assembler_and_fitness.py` into your project and fill in `assemble()` (chromosome → object to score) and `score()` (object → fitness). Point `fitness_function` (and `chromosome_to_*` if you're not using the library default) in your YAML at them via a `module.path:attr` reference, e.g. `fitness_function: my_project.assembler_and_fitness:score`. Use `module.path:attr()` if that attribute is a zero-arg factory instead of a plain callable (as several of navicatGA's own bundled examples are, e.g. `concatenate_list()`).
3. Fill in `params` for your problem (`n_genes` is always required; everything else has a working default).
4. Copy `templates/launcher.py` into your project and run it (`python launcher.py --config my_project/config.yaml`), or build and run inline:

```python
from navicatGA.config import build_solver_from_yaml

solver, config = build_solver_from_yaml("my_project/config.yaml")
result = solver.solve()
```

If your alphabet/starting population is computed at runtime (read from a database, generated from a directory of structures, etc. — the common case for `smiles`/`xyz`), don't try to express it in YAML: compute it in your launcher and pass it as an extra kwarg, which overrides/extends `params`:

```python
alphabet_list = my_alphabet_builder(...)
solver, config = build_solver_from_yaml("my_project/config.yaml", alphabet_list=alphabet_list)
```

See `ga_flp/config/config.yaml` + `ga_flp/launcher.py` for a full worked example, including a `scalarizer` block and a project-specific alphabet builder — it's a filled-in, extended version of these same three templates.
