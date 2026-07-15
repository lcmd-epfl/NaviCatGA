# NaviCatGA

**A flexible pure-Python Genetic Algorithm optimizer, part of the [NaviCat project](https://github.com/lcmd-epfl/NaviCat).**

[![DOI](https://zenodo.org/badge/322634265.svg)](https://zenodo.org/badge/latestdoi/322634265)

NaviCatGA is developed by [LCMD](https://www.epfl.ch/labs/lcmd/) at EPFL, part of the NaviCat project funded within the [NCCR Catalysis](https://www.nccr-catalysis.ch/) of the [Swiss National Science Foundation](https://www.snf.ch/en).

The base solver's dependencies are minimal (`numpy`, `matplotlib`); each representation-specific child solver pulls in chemistry dependencies only when used (`rdkit`, `selfies`, `AaronTools`, `pyscf`, `matter-chimera`). See [Installation](installation.md).

---

## Key ideas

- **One base class, several representations.** `GenAlgSolver` runs the generic GA loop (selection, crossover, mutation, convergence). It knows nothing about what a chromosome *means*: that's supplied by an `assembler` (chromosome → object to score) and a `fitness_function` (object → score), passed in at construction.
- **Four ready-made solvers**: `FloatGenAlgSolver` (continuous vectors), `SmilesGenAlgSolver`/`SelfiesGenAlgSolver` (molecules via SMILES/SELFIES fragments), `XYZGenAlgSolver` (3D structures via AaronTools substituent/ligand swaps). They're example/reference implementations meant to be adapted or subclassed, not a fixed closed API.
- **Config-driven construction.** `navicatGA.config.build_solver_from_yaml` builds any solver from a YAML file instead of a hardcoded constructor call, resolving `fitness_function`/`chromosome_to_*`/`scalarizer` by dotted reference. See [Configuration Templates](config_templates.md).

## Quick start

```python
from navicatGA.float_solver import FloatGenAlgSolver

def my_fitness(x):
    return -sum((xi - 0.5) ** 2 for xi in x)

solver = FloatGenAlgSolver(
    n_genes=3,
    fitness_function=my_fitness,
    variables_limits=(0, 1),
    pop_size=50,
    max_gen=100,
)
result = solver.solve()
print(result.best_individual, result.best_fitness)
```

Start with [Tutorial 1](tutorials/01_float.md) for the full walkthrough of what `assembler`/`fitness_function` mean and how the rest of the GA-mechanics params work, then move on to the SMILES/SELFIES/config-driven/multi-objective tutorials as your problem needs them. `navicatGA/test/` also doubles as a set of worked examples (there's no separate `examples/` directory).

---

## Citation

If you use NaviCatGA in your research, please cite:

[![DOI](https://img.shields.io/badge/DOI-10.1002/cmtd.202100107-red)](https://chemistry-europe.onlinelibrary.wiley.com/doi/abs/10.1002/cmtd.202100107)

[Genetic Optimization of Homogeneous Catalysts](https://chemistry-europe.onlinelibrary.wiley.com/doi/abs/10.1002/cmtd.202100107)

```bibtex
@article{laplaza_genetic_2022,
	title = {Genetic {Optimization} of {Homogeneous} {Catalysts}},
	url = {https://chemistry-europe.onlinelibrary.wiley.com/doi/abs/10.1002/cmtd.202100107},
	doi = {10.1002/cmtd.202100107},
	language = {en},
	journal = {Chemistry–Methods},
	volume = {2},
	pages = {e202100107},
	author = {Laplaza, Raúl and Gallarati, Simone and Corminboeuf, Clémence},
	month = mar,
	year = {2022},
}
```

---

```{toctree}
:maxdepth: 1
:caption: Getting Started

installation
```

```{toctree}
:maxdepth: 1
:caption: Tutorials

tutorials/index
```

```{toctree}
:maxdepth: 1
:caption: Configuration

config_templates
```

```{toctree}
:maxdepth: 2
:caption: API Reference

api
```
