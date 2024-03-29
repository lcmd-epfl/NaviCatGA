NaviCatGA: A flexible Genetic Algorithm Optimizer for the NaviCat project
=========================================================================

![NaviCatGA logo](./images/navicatga_logo.png)
[![DOI](https://zenodo.org/badge/322634265.svg)](https://zenodo.org/badge/latestdoi/322634265)


## Contents
* [About](#about-)
* [Install](#install-)
* [Documentation](#documentation-)
* [Examples](#examples-)

## About [↑](#about)

NaviCatGA is part of the [NaviCat project](https://github.com/lcmd-epfl/NaviCat) from [LCMD](https://www.epfl.ch/labs/lcmd/) at EPFL.

The NaviCat project is funded within the [NCCR Catalysis](https://www.nccr-catalysis.ch/)  of the [Swiss National Science Foundation](https://www.snf.ch/en).

The NaviCatGA code runs on pure python and is thus it is easy to adapt for particular applications.
Dependencies are minimal for the base class: 
- `numpy`
- `matplotlib`

The library is projected as base class containing the core methods, which are inherited by child classes to define the problem.

![Inheritance diagram](./images/inheritance.png)

Several child solver classes are provided as fairly complete examples, but might need to be adapted or monkeypatched for particular applications.

The child classes and some functionalities have additional dependencies:

- The selfies_solver class implementation requires `selfies` (https://github.com/aspuru-guzik-group/selfies) and `rdkit` (https://www.rdkit.org/). `rdkit` may be replaced manually by openbabel python bindings or other chemoinformatic modules.
- The smiles_solver class implementation requires `rdkit` (https://www.rdkit.org/). `rdkit` may be replaced manually by openbabel python bindings or other chemoinformatic modules.
- The xyz_solver class implementation requires `AaronTools` (https://github.com/QChASM/AaronTools.py). 
- Wrappers and chemistry modules contain functions that depend on `pyscf` to solve the electronic structure problem. However, these are provided for exemplary purposes and not a core functionality.
- `matter-chimera` (https://github.com/aspuru-guzik-group/chimera) is recommended for scalarization. Alternatively, a scalarizer object with a scalarize method can be passed to the solver.

Additional features require `alive-progress` (for progress bars, very useful for CLI usage). However, these are implemented by monkeypatching the base class, and thus no functionality is lost without them.


## Install [↑](#install)

Installation is as simple as:
```python
python setup.py install --record files.txt
```

This ensures easy uninstall. Just remove all files listed in files.txt using:
```bash
rm $(cat files.txt)
```

## Documentation [↑](#documentation)

The documentation is available [here](https://navicatga.readthedocs.io/).

## Examples [↑](#examples)

The tests subdirectory contains a copious amount of tests which double as examples.

---


