# 7. Writing your own assembler, fitness function, and launcher

Every earlier tutorial used a bundled example assembler or fitness function. This one builds both from scratch for a custom problem, then wires them into a reusable launcher script: the three pieces `templates/assembler_and_fitness.py` and `templates/launcher.py` are stubs for.

## Assembler requirements

An assembler is `fn(chromosome) -> object_to_score`, where `chromosome` is a 1-D array of genes (strings, for the alphabet-based solvers). Two rules:

1. **Raise on an invalid chromosome.** navicatGA retries mutations/crossovers via `navicatGA.helpers.check_error`, which calls your assembler and catches any exception to decide whether a candidate chromosome is usable. If your assembler silently returns garbage instead of raising, invalid candidates slip through instead of being retried.
2. **Return something your fitness function can score**, and something hashable too, but only if you're using `lru_cache=True`.

## Fitness function requirements

`fn(assembled_object) -> float`, or `-> tuple[float, ...]` if you're combining several objectives with a `scalarizer` (see [Tutorial 5](05_multi_objective.md)). It's called on every chromosome in the population every generation, so keep it as cheap as your problem allows: an expensive fitness function (an external simulation, a subprocess call) dominates total runtime far more than the GA mechanics do.

## A worked example: a custom ester assembler

This ignores navicatGA's bundled `chromosome_to_smiles()` and `sc2logp` entirely and writes both from scratch: a 2-gene chromosome assembled into `R1-C(=O)-O-R2`, scored by logP.

```python
# assembler_and_fitness.py
from rdkit import Chem
from rdkit.Chem import Descriptors


def assemble(chromosome):
    """chromosome[0], chromosome[1] are SMILES fragments for R1, R2 in
    R1-C(=O)-O-R2. Raises ValueError if the result isn't a valid molecule -
    this is what lets navicatGA retry a mutation/crossover instead of
    accepting a broken candidate.
    """
    r1, r2 = chromosome[0], chromosome[1]
    smiles = f"{r1}C(=O)O{r2}"
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    return smiles


def score(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return Descriptors.MolLogP(mol)
```

Used directly:

```python
from navicatGA.smiles_solver import SmilesGenAlgSolver
from assembler_and_fitness import assemble, score

solver = SmilesGenAlgSolver(
    n_genes=2,
    pop_size=10,
    max_gen=5,
    fitness_function=score,
    chromosome_to_smiles=assemble,
    alphabet_list=["C", "CC", "CCC", "c1ccccc1"],
    starting_random=True,
    random_state=3,
    to_file=False,
    verbose=False,
)
result = solver.solve()
print(assemble(result.best_individual), result.best_fitness)
# c1ccccc1C(=O)Oc1ccccc1 2.905800000000001
```

## The same thing, config-driven

Once `assembler_and_fitness.py` is an importable module, reference it from YAML instead ([Tutorial 4](04_config_driven.md)):

```yaml
# config.yaml
solver:
  type: smiles
  fitness_function: assembler_and_fitness:score
  chromosome_to_smiles: assembler_and_fitness:assemble
  params:
    n_genes: 2
    alphabet_list: ["C", "CC", "CCC", "c1ccccc1"]
    pop_size: 10
    max_gen: 5
    starting_random: true
    random_state: 3
    to_file: false
    verbose: false
```

```python
from navicatGA.config import build_solver_from_yaml

solver, config = build_solver_from_yaml("config.yaml")
result = solver.solve()
```

## Building the launcher

`templates/launcher.py` is the generic version of the script above. Copy it, then extend it per the two patterns below.

### Pattern: a value that can't be static YAML

Some inputs are computed at runtime rather than fixed; the most common case is an alphabet loaded from a file. This doesn't belong in YAML (it isn't static config, it's data); load it in the launcher and pass it as an extra keyword to `build_solver_from_yaml`, which overrides/extends the YAML's `params` block:

```python
# launcher.py
import argparse
from navicatGA.config import build_solver_from_yaml


def load_alphabet(path):
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--alphabet_file", default="alphabet.txt")
    args = parser.parse_args()

    alphabet_list = load_alphabet(args.alphabet_file)
    solver, config = build_solver_from_yaml(args.config, alphabet_list=alphabet_list)

    result = solver.solve()
    print(f"Best individual: {result.best_individual}")
    print(f"Best fitness: {result.best_fitness}")
    solver.close_solver_logger()


if __name__ == "__main__":
    main()
```

### Pattern: multiple cycles with per-cycle logging

`solve()` continues from where it left off if called again (it reuses `solver.population_`/`solver.fitness_` if already set). Call `solver.solve(1)` in a loop instead of one `solver.solve()` call when you want to inspect, log, or checkpoint between generations:

```python
for cycle in range(args.num_cycles):
    solver.solve(1)
    print(f"[cycle {cycle}] best fitness so far: {solver.best_fitness_}")
    # e.g. write solver.population_/solver.fitness_/solver.printable_fitness
    # to a per-cycle file here, or call solver.write_population() to depict
    # the current population (SmilesGenAlgSolver/SelfiesGenAlgSolver/XYZGenAlgSolver)
```

`templates/launcher.py` already implements this `--num_cycles` flag; add the `load_alphabet`-style runtime-value pattern on top of it for your own problem.

## Recap: the three files

| File | Role |
| :--- | :--- |
| `assembler_and_fitness.py` | Your `assemble()`/`score()`, the only problem-specific logic. |
| `config.yaml` | Solver type, references to the above, GA-mechanics params. |
| `launcher.py` | Loads any runtime-only values, builds the solver from YAML, runs it, handles output. |

See [Configuration Templates](../config_templates.md) for the full template files to copy.
