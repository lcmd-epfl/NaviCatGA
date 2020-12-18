#!/usr/bin/env python3

from selfies_solver import SelfiesGenAlgSolver
from fitness_functions_selfies import fitness_function_selfies
from selfies import decoder
from chemistry.evo import sanitize_smiles
from rdkit import Chem
from rdkit.Chem import AllChem, AddHs, MolToXYZFile
from rdkit.Chem.AllChem import EmbedMolecule, ETKDGv2, MMFFOptimizeMolecule

solver = SelfiesGenAlgSolver(
    n_genes=15,
    pop_size=100,
    max_gen=250,
    mutation_rate=0.25,
    selection_rate=0.75,
    variables_limits=(0, 1),
    fitness_function=fitness_function_selfies(1),
    selection_strategy="roulette_wheel",
)
solver.solve()
smiles = decoder("".join(x for x in list(solver.best_individual_)))
print(solver.best_fitness_)
mol = sanitize_smiles(smiles)[0]
mol_h = AddHs(mol)
EmbedMolecule(mol_h, useRandomCoords=True)
MMFFOptimizeMolecule(mol_h)
MolToXYZFile(mol_h, "test.xyz")
