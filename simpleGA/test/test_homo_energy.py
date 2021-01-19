#!/usr/bin/env python3

from selfies import encoder
from simpleGA.selfies_solver import SelfiesGenAlgSolver
from simpleGA.fitness_functions_selfies import fitness_function_selfies
from simpleGA.chemistry import count_selfie_chars
from simpleGA.wrappers import sc2smiles, sc2mol_structure, mol_structure2depictions


def test_homo_energy():
    starting_smiles = "CC(=O)O"  # Acetic acid
    starting_selfies = [encoder(starting_smiles)]
    n_starting_genes = count_selfie_chars(starting_selfies[0])
    print(
        "This test attempts to modify melatonin to improve a given property. \n The starting SMILES is : {0} \n The starting SELFIES is : {1} \n The number of genes required is : {2}".format(
            starting_smiles, starting_selfies, n_starting_genes
        )
    )

    # We will now maximize the homo energy
    solver = SelfiesGenAlgSolver(
        starting_selfies=starting_selfies,  #
        n_genes=int(n_starting_genes * 1.5),  # We need at least n_starting_genes
        excluded_genes=list(range(n_starting_genes)),  # We do not modify the backbone
        fitness_function=fitness_function_selfies(7),  # See fitness_functions_selfies
        pop_size=10,
        max_gen=10,  # This is a simple test, no need for many
        random_state=420,
        logger_file="acetic_acid_homo_energy.log",
        to_stdout=True,
        verbose=True,
        progress_bars=True,
        lru_cache=True,
    )
    solver.solve()
    print(
        "The best candidate for gap maximization has a gap of : {0}".format(
            solver.best_fitness_
        )
    )
    print(
        "The corresponding SMILES is : {0}".format(sc2smiles(solver.best_individual_))
    )
    mol = sc2mol_structure(solver.best_individual_)
    mol_structure2depictions(mol, root_name="acetic_acid_homo_energy")


if __name__ == "__main__":
    test_homo_energy()
