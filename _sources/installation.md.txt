# Installation

## Base install

The base solver (`GenAlgSolver`, `FloatGenAlgSolver`) only needs `numpy` and `matplotlib`:

```bash
python setup.py install --record files.txt
# uninstall cleanly: rm $(cat files.txt)
```

## Representation-specific dependencies

Each child solver's extra dependencies are only needed if you use that solver:

| Solver | Extra dependencies |
| :--- | :--- |
| `SmilesGenAlgSolver` | [`rdkit`](https://www.rdkit.org/) (or swap in openbabel/other chemoinformatics bindings) |
| `SelfiesGenAlgSolver` | [`selfies`](https://github.com/aspuru-guzik-group/selfies), `rdkit` |
| `XYZGenAlgSolver` | [`AaronTools`](https://github.com/QChASM/AaronTools.py) |

Optional, feature-scoped extras:

- `pyscf`: only needed by the `quantum_wrappers_{selfies,xyz}.py` electronic-structure fitness functions (exemplary, not core functionality); these imports are lazy, so plain SMILES/SELFIES runs (e.g. logP) don't require it.
- [`matter-chimera`](https://github.com/aspuru-guzik-group/chimera): recommended for multi-objective scalarization; any object with a `.scalarize()` method works instead.
- [`tqdm`](https://github.com/tqdm/tqdm): progress bar over fitness evaluations via `progress_bars=True`; the import is lazy, so nothing is lost without it.
- `pyyaml`: needed by `navicatGA.config` for YAML-driven solver construction.

## Full pinned environment

`environment.yml` at the repo root pins a complete conda environment including `aarontools`, `pyscf`, and `matter-chimera` for running the full test suite.

## Building these docs locally

```bash
pip install -r docs/requirements.txt
sphinx-build -b html docs/ docs/_build/html
```

Then open `docs/_build/html/index.html`.
