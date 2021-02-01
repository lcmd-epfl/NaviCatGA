#!/usr/bin/env python3

from selfies import encoder
from simpleGA.selfies_solver import SelfiesGenAlgSolver
from simpleGA.chemistry_selfies import count_selfie_chars
from simpleGA.wrappers_selfies import (
    sc2smiles,
    sc2mol_structure,
    mol_structure2depictions,
)
from simpleGA.quantum_wrappers_selfies import sc2ehomo


def fitness_function_wrapper():

    return lambda chromosome: sc2ehomo(chromosome, lot=1)


def test_acetic_acid_01():
    starting_smiles = "CC(=O)O"  # Acetic acid
    starting_selfies = [encoder(starting_smiles)]
    n_starting_genes = count_selfie_chars(starting_selfies[0])
    print(
        "This test attempts to modify acetic acid to improve a given property. \n The starting SMILES is : {0} \n The starting SELFIES is : {1} \n The number of genes required is : {2}".format(
            starting_smiles, starting_selfies, n_starting_genes
        )
    )

    # We will now maximize the homo energy
    solver = SelfiesGenAlgSolver(
        starting_selfies=starting_selfies,  #
        n_genes=int(n_starting_genes * 1.2),  # We need at least n_starting_genes
        excluded_genes=list(range(n_starting_genes)),  # We do not modify the backbone
        fitness_function=fitness_function_wrapper(),
        pop_size=10,
        max_gen=5,  # This is a simple test, no need for many
        random_state=420,
        to_file=False,
        to_stdout=True,
        verbose=True,
        progress_bars=True,
        lru_cache=True,
    )
    solver.solve()
    print(
        "After optimization, the corresponding SMILES is : {0}".format(
            sc2smiles(solver.best_individual_)
        )
    )
    mol = sc2mol_structure(solver.best_individual_)
    mol_structure2depictions(mol, "acetic_acid_ehomo")


if __name__ == "__main__":
    test_acetic_acid_01()
