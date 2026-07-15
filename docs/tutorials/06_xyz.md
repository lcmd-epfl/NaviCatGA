# 6. 3D structures (XYZ/AaronTools)

`XYZGenAlgSolver` swaps 3D fragments (substituents, ligands) in and out of a scaffold using [AaronTools.py](https://github.com/QChASM/AaronTools.py), rather than operating on a string representation. It shares the same `AlphabetGenAlgSolver` base as `SmilesGenAlgSolver`/`SelfiesGenAlgSolver` ([Tutorial 2](02_smiles.md)), with the same `alphabet_list`/`starting_population`/`excluded_genes`/`equivalences` shape, but with an assembler and fitness function that operate on AaronTools `Geometry` objects instead of strings.

```{note}
This page couldn't be run against a live install while writing it: `AaronTools` isn't installed in this environment. The snippet below is adapted directly from `navicatGA/test/test_scaffolds.py` (a real, working example in the test suite), not fabricated, but treat it as a starting point to verify yourself rather than a guaranteed-to-run copy-paste.
```

## Install

```bash
pip install AaronTools  # or: git clone https://github.com/QChASM/AaronTools.py
```

## A scaffold-substitution example

```python
import os
from navicatGA.xyz_solver import XYZGenAlgSolver
from navicatGA.chemistry_xyz import get_alphabet_from_path, get_default_alphabet
from navicatGA.wrappers_xyz import chromosome_to_xyz, geom2dihedral, geom2sub_sterimol
from navicatGA.quantum_wrappers_xyz import geom2ehomo  # needs pyscf

# alphabet_list[0]: scaffold geometries loaded from a directory of .xyz files
# alphabet_list[1:]: substituents drawn from navicatGA's built-in default alphabet
alphabet_list = [
    get_alphabet_from_path("path/to/scaffolds/"),
    get_default_alphabet(),
    get_default_alphabet(),
]

def fitness(geom):
    ehomo = geom2ehomo(geom, lot=0)
    dihedral = geom2dihedral(geom, "2", "1", "18", "21")
    b1 = (geom2sub_sterimol(geom, "19", "B1") + geom2sub_sterimol(geom, "20", "B1")) / 2
    return -(0.4 * dihedral + 0.31 * ehomo + 11.77 / b1 + 26.59)

solver = XYZGenAlgSolver(
    n_genes=3,
    pop_size=10,
    max_gen=5,
    mutation_rate=0.15,
    fitness_function=fitness,
    chromosome_to_xyz=chromosome_to_xyz(),
    alphabet_list=alphabet_list,
    starting_random=True,
    random_state=24,
    to_file=False,
)
result = solver.solve()
```

## Differences from the string-based solvers

- **No reversal on crossover.** SMILES/SELFIES crossover reverses gene order in some cases (see `create_offspring` in each solver); `XYZGenAlgSolver` never does, since reversing a 3D fragment order doesn't have a meaningful analogue.
- **`_setup_alphabet(..., sort_alphabet=False)`.** AaronTools geometries aren't orderable, unlike SMILES/SELFIES tokens, so the shared alphabet-setup logic skips sorting for this solver.
- **`chemistry_xyz.get_alphabet_from_path(directory)`** builds an alphabet from every structure file in a directory: the common way to seed `alphabet_list` here, analogous to reading a runtime-computed alphabet for [`SmilesGenAlgSolver`](02_smiles.md) (see [Tutorial 7](07_custom_solver.md) for that pattern in general).
- **`quantum_wrappers_xyz.py`**: optional `pyscf`-based fitness helpers (electronic structure), same lazy-import treatment as the SELFIES/SMILES quantum wrappers ([Tutorial 3](03_selfies.md)).
- **Heavier per-evaluation cost.** `XYZGenAlgSolver`'s own defaults reflect this: `pop_size=5`, `max_gen=15` (vs. 100/500 for the other solvers), since geometry optimization/fitness evaluation is typically far more expensive than a string-based one.

See `navicatGA/queue_wrappers_xyz.py` for a queue-based variant if your fitness evaluations should be dispatched to an external job scheduler instead of run in-process.
