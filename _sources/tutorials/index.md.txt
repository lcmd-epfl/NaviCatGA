# Tutorials

Staged walkthroughs, from a minimal continuous-optimization run to a full config-driven, multi-objective setup, ending with how to build your own assembler/fitness function/launcher from scratch. Every code snippet has been run against a real `navicatGA` install and its output verified, except where a page says otherwise (Tutorial 6 needs `AaronTools`, not installed while writing these). Copy-paste them as a starting point.

```{toctree}
:maxdepth: 1

01_float
02_smiles
03_selfies
04_config_driven
05_multi_objective
06_xyz
07_custom_solver
```

| Tutorial | You'll learn |
| :--- | :--- |
| [1. Continuous optimization](01_float.md) | The core `assembler`/`fitness_function` pattern with `FloatGenAlgSolver`; `GenAlgSolver`'s shared GA-mechanics params. |
| [2. SMILES fragment optimization](02_smiles.md) | Alphabets, `excluded_genes`, seeding a starting population, `SmilesGenAlgSolver`. |
| [3. Editing a real molecule (SELFIES)](03_selfies.md) | Why SELFIES over SMILES; freezing a scaffold and mutating only new positions. |
| [4. Config-driven runs](04_config_driven.md) | Moving a Python constructor call into a YAML file with `navicatGA.config`, using `templates/`. |
| [5. Multi-objective optimization](05_multi_objective.md) | Returning several fitness values and combining them with a `scalarizer` (chimera). |
| [6. 3D structures (XYZ/AaronTools)](06_xyz.md) | Where `XYZGenAlgSolver` fits, and its extra dependency. |
| [7. Writing your own assembler/fitness/launcher](07_custom_solver.md) | Building `assemble()`/`score()` from scratch for a custom problem, and the launcher patterns for runtime-only values and multi-cycle runs. |
