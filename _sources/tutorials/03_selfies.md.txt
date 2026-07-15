# 3. Editing a real molecule (SELFIES)

Raw SMILES mutations can easily produce invalid molecules (unbalanced rings/branches), which is why `SmilesGenAlgSolver`'s assembler has to validate-and-retry on every mutation. [SELFIES](https://github.com/aspuru-guzik-group/selfies) tokens are constructed so that **every** mutation decodes to a valid molecule, making `SelfiesGenAlgSolver` the better default for open-ended molecule editing.

## Starting from a real molecule and freezing it

This example starts from melatonin, appends genes for the GA to fill in, and optimizes logP while never touching the original scaffold:

```python
from selfies import encoder
from navicatGA.selfies_solver import SelfiesGenAlgSolver
from navicatGA.fitness_functions_selfies import fitness_function_selfies
from navicatGA.chemistry_selfies import count_selfie_chars
from navicatGA.wrappers_selfies import sc2smiles

starting_smiles = "CC(=O)NCCc1c[nH]c2c1cc(cc2)OC"  # melatonin
starting_selfies = [encoder(starting_smiles)]
n_starting_genes = count_selfie_chars(starting_selfies[0])  # 25

solver = SelfiesGenAlgSolver(
    starting_selfies=starting_selfies,
    n_genes=int(n_starting_genes * 2),           # room to grow past the scaffold
    excluded_genes=list(range(n_starting_genes)), # freeze the scaffold's own genes
    fitness_function=fitness_function_selfies(1), # 1 = logP; see below for the full list
    max_gen=10,
    pop_size=10,
    random_state=42,
    to_file=False,
    verbose=False,
)
result = solver.solve()
print(result.best_fitness)                 # 8.506...
print(sc2smiles(result.best_individual))    # the resulting SMILES
```

## Bundled fitness functions

`navicatGA.fitness_functions_selfies.fitness_function_selfies(function_number)` is a factory: numbers 1 to 9 select logP, inverse logP, molecular weight, negative MW, MW×inverse-logP, molecular volume, HOMO energy, LUMO energy, HOMO-LUMO gap respectively (see the module docstring/source for the exact mapping). Numbers 7 to 9 need `pyscf` (imported lazily, so 1 to 6 work without it). `fitness_function_target_property(target, function_number, score_modifier_number)` scores how close a property is to a *target* value instead of maximizing it outright; referencing it with arguments from a YAML config works the same way as `fitness_function_selfies(1)` above, see [Configuration Templates](../config_templates.md#reference-resolution).

For anything else, write your own `fn(chromosome) -> float` and pass it as `fitness_function` directly. The numbered functions are examples, not a closed API.

## Other `SelfiesGenAlgSolver`-specific params

- **`starting_stoned`**: generate a chemical subspace around a single starting molecule via the [STONED](https://github.com/aspuru-guzik-group/stoned-selfies) method, instead of a fixed/random starting population. Incompatible with `starting_random`; needs exactly one `starting_selfies` entry.
- **`branching`**: add branch tokens to the alphabet so mutations can grow new branches, not just substitute atoms.
- **`variables_limits`**: if `True`, sets SELFIES semantic constraints (valence rules) on the alphabet.

The rest (`alphabet_list`, `excluded_genes`, `multi_alphabet`/`equivalences`, `max_counter`, GA-mechanics params) work exactly as in [Tutorial 2](02_smiles.md), since both solvers share the same `AlphabetGenAlgSolver` base.

Next: [Tutorial 4: moving this into a YAML config](04_config_driven.md).
