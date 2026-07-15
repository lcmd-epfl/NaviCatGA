# 2. SMILES fragment optimization

`SmilesGenAlgSolver` builds a molecule by picking one SMILES fragment per gene from a per-gene **alphabet**, concatenating them (or however your `chromosome_to_smiles` assembler combines them), and scoring the result.

## A minimal run

This uses `navicatGA`'s own bundled example assembler (`chromosome_to_smiles()`, which builds a `core(P(...)(...)(...))(P(...)(...)(...))` phosphine scaffold; see `navicatGA/wrappers_smiles.py` if you want to see exactly what it does) and `smiles2logp` as the fitness function:

```python
from navicatGA.smiles_solver import SmilesGenAlgSolver
from navicatGA.wrappers_smiles import chromosome_to_smiles, smiles2logp

alphabet_list = ["C", "N", "O", "F", "[H]"]

solver = SmilesGenAlgSolver(
    n_genes=7,                              # this assembler expects exactly 7 genes
    pop_size=20,
    max_gen=10,
    mutation_rate=0.15,
    fitness_function=smiles2logp,
    chromosome_to_smiles=chromosome_to_smiles(),   # note the (): it's a factory
    alphabet_list=alphabet_list,
    starting_random=True,
    excluded_genes=[0],                     # freeze the first gene (the core)
    starting_population=[["[Fe]"]],         # seed gene 0 with an iron core
    random_state=7,
    to_file=False,
    verbose=False,
)
result = solver.solve()
print(result.best_individual)  # ['[Fe]' 'F' 'F' 'F' 'F' 'F' 'F']
print(result.best_fitness)     # 5.611700000000002
```

## Ingredients specific to alphabet-based solvers

`SmilesGenAlgSolver`, `SelfiesGenAlgSolver`, and `XYZGenAlgSolver` all share a base (`AlphabetGenAlgSolver`) and its extra constructor params:

- **`alphabet_list`**: either one alphabet shared by every gene (a flat list, as above), or a list of `n_genes` alphabets, one per gene, for `multi_alphabet=True`. When you pass per-gene alphabets, navicatGA groups genes with an identical alphabet into **equivalence classes** automatically (or pass `equivalences` yourself); crossover reasons about a whole equivalence group together.
- **`starting_population`**: a list of starting chromosomes (list of lists). Padded/trimmed randomly to `pop_size` if it doesn't already match.
- **`starting_random`**: if `True`, every chromosome position not in `excluded_genes` is randomized from `alphabet_list` before the first generation.
- **`max_counter`**: how many times a mutation/crossover is retried before giving up and falling back to a parent, when the assembler rejects the result (e.g. an invalid SMILES). navicatGA relies on your assembler *raising* on an invalid chromosome to know to retry.

## Factory-style assemblers

Some of navicatGA's bundled example assemblers are factories: calling them returns the actual assembler function (`concatenate_list()`, `make_array()`, `chromosome_to_smiles()`). This matters again in [Tutorial 4](04_config_driven.md), where you reference them from YAML.

Next: [Tutorial 3: editing a real molecule with SELFIES](03_selfies.md).
