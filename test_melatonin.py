#!/usr/bin/env python3

from selfies_solver import SelfiesGenAlgSolver
from fitness_functions_selfies import fitness_function_selfies
from selfies import decoder, encoder
from chemistry.evo import (
    count_selfie_chars,
)
from chemistry.wrappers import (
    sc2smiles,
    sc2mol_structure,
    mol_structure2depictions,
)


def test_melatonin():

    starting_smiles = "CC(=O)NCCc1c[nH]c2c1cc(cc2)OC"
    starting_selfies = encoder(starting_smiles)
    n_starting_genes = count_selfie_chars(starting_selfies)
    print(
        "This test attempts to modify melatonin to improve a given property. \n The starting SMILES is : {0} \n The starting SELFIES is : {1} \n The number of genes required is : {2}".format(
            starting_smiles, starting_selfies, n_starting_genes
        )
    )

    # We will now maximize logp
    solver = SelfiesGenAlgSolver(
        starting_selfies=starting_selfies,  # We start the run from the melatonin molecule
        n_genes=int(n_starting_genes * 2),  # We need at least n_starting_genes
        excluded_genes=list(
            range(n_starting_genes)
        ),  # We do not modify the melatonin molecule
        fitness_function=fitness_function_selfies(1),  # See fitness_functions_selfies
        max_gen=100,  # This is a simple test
        logger_file="logp.log",
        verbose=True,
    )
    solver.solve()
    print(
        "The best candidate for logp maximization has a logp of : {0}".format(
            solver.best_fitness_
        )
    )
    print(
        "The corresponding SMILES is : {0}".format(sc2smiles(solver.best_individual_))
    )
    mol = sc2mol_structure(solver.best_individual_)
    mol_structure2depictions(mol, root_name="logp")

    # We will not maximize 1/logp
    solver = SelfiesGenAlgSolver(
        starting_selfies=starting_selfies,  # We start the run from the melatonin molecule
        n_genes=int(n_starting_genes * 2),  # We need at least n_starting_genes
        excluded_genes=list(
            range(n_starting_genes)
        ),  # We do not modify the melatonin molecule
        fitness_function=fitness_function_selfies(
            2
        ),  # See fitness_functions_selfies, this is inverse logp
        max_gen=100,  # This is a simple test
        logger_file="ilogp.log",
        verbose=True,
    )
    solver.solve()
    print(
        "The best candidate for ilogp maximization has an ilogp of : {0}".format(
            solver.best_fitness_
        )
    )
    print(
        "The corresponding SMILES is : {0}".format(sc2smiles(solver.best_individual_))
    )
    mol = sc2mol_structure(solver.best_individual_)
    mol_structure2depictions(mol, root_name="ilogp")

    # We will not maximize molecular weight
    solver = SelfiesGenAlgSolver(
        starting_selfies=starting_selfies,  # We start the run from the melatonin molecule
        n_genes=int(n_starting_genes * 2),  # We need at least n_starting_genes
        excluded_genes=list(
            range(n_starting_genes)
        ),  # We do not modify the melatonin molecule
        fitness_function=fitness_function_selfies(
            3
        ),  # See fitness_functions_selfies, this is mw
        max_gen=100,  # This is a simple test
        logger_file="mw.log",
        verbose=True,
    )
    solver.solve()
    print(
        "The best candidate for mw maximization has a mw of : {0}".format(
            solver.best_fitness_
        )
    )
    print(
        "The corresponding SMILES is : {0}".format(sc2smiles(solver.best_individual_))
    )
    mol = sc2mol_structure(solver.best_individual_)
    mol_structure2depictions(mol, root_name="mw")

    # We will not maximize molecular volume
    solver = SelfiesGenAlgSolver(
        starting_selfies=starting_selfies,  # We start the run from the melatonin molecule
        n_genes=int(n_starting_genes * 2),  # We need at least n_starting_genes
        excluded_genes=list(
            range(n_starting_genes)
        ),  # We do not modify the melatonin molecule
        fitness_function=fitness_function_selfies(
            6
        ),  # See fitness_functions_selfies, this is mv
        max_gen=5,  # This is a simple test and this run is more expensive
        pop_size=10,  # So we reduce the size of everything
        logger_file="mv.log",
        verbose=True,
    )
    solver.solve()
    print(
        "The best candidate for mv maximization has a mv of : {0}".format(
            solver.best_fitness_
        )
    )
    print(
        "The corresponding SMILES is : {0}".format(sc2smiles(solver.best_individual_))
    )
    mol = sc2mol_structure(solver.best_individual_)
    mol_structure2depictions(mol, root_name="mv")


if __name__ == "__main__":
    test_melatonin()
