#!/usr/bin/env python3

from selfies import encoder
from simpleGA.selfies_solver import SelfiesGenAlgSolver
from simpleGA.fitness_functions_selfies import fitness_function_target_selfies
from simpleGA.wrappers_selfies import (
    sc2smiles,
    sc2mol_structure,
    mol_structure2depictions,
)


def test_tanimoto_methane_19():

    target_smiles = "CC(=O)NCCc1c[nH]c2c1cc(cc2)OC"
    target_selfies = encoder(target_smiles)
    print(
        "This test attempts to generate melatonin from methane. \n The target SELFIES is : {0}".format(
            target_selfies
        )
    )

    solver = SelfiesGenAlgSolver(
        starting_selfies=["[C]"],
        n_genes=30,
        fitness_function=fitness_function_target_selfies(
            target_selfies, function_number=1
        ),  # See fitness_function_target_selfies
        max_gen=100,
        pop_size=50,
        n_crossover_points=2,
        random_state=420,
        logger_file="tanimoto_methane.log",
        verbose=True,
    )
    solver.solve()
    print(
        "The best candidate has a Tanimoto distance of : {0}".format(
            solver.best_fitness_
        )
    )
    print(
        "The corresponding SELFIES is : {0}".format(
            encoder(sc2smiles(solver.best_individual_))
        )
    )
    mol = sc2mol_structure(solver.best_individual_)
    mol_structure2depictions(mol, root_name="tanimoto_methane")


def test_tanimoto_random_20():

    target_smiles = "CC(=O)NCCc1c[nH]c2c1cc(cc2)OC"
    target_selfies = encoder(target_smiles)
    print(
        "This test attempts to generate melatonin from random molecules. \n The target SELFIES is : {0}".format(
            target_selfies
        )
    )

    solver = SelfiesGenAlgSolver(
        starting_random=True,
        n_genes=30,
        fitness_function=fitness_function_target_selfies(
            target_selfies, function_number=1
        ),  # See fitness_function_target_selfies
        max_gen=100,
        pop_size=50,
        n_crossover_points=2,
        random_state=420,
        logger_file="tanimoto_random.log",
        verbose=True,
    )
    solver.solve()
    print(
        "The best candidate has a Tanimoto distance of : {0}".format(
            solver.best_fitness_
        )
    )
    print(
        "The corresponding SELFIES is : {0}".format(
            encoder(sc2smiles(solver.best_individual_))
        )
    )
    mol = sc2mol_structure(solver.best_individual_)
    mol_structure2depictions(mol, root_name="tanimoto_random")


def test_levenshtein_methane_21():

    target_smiles = "CC(=O)NCCc1c[nH]c2c1cc(cc2)OC"
    target_selfies = encoder(target_smiles)
    print(
        "This test attempts to generate melatonin from methane. \n The target SELFIES is : {0}".format(
            target_selfies
        )
    )

    solver = SelfiesGenAlgSolver(
        starting_selfies=["[C]"],
        n_genes=30,
        fitness_function=fitness_function_target_selfies(
            target_selfies, function_number=2
        ),  # See fitness_function_target_selfies
        max_gen=100,
        pop_size=50,
        n_crossover_points=1,
        random_state=420,
        logger_file="levenshtein_methane.log",
        verbose=True,
    )
    solver.solve()
    print(
        "The best candidate has a Levenshtein distance of : {0}".format(
            solver.best_fitness_
        )
    )
    mol = sc2mol_structure(solver.best_individual_)
    mol_structure2depictions(mol, root_name="levenshtein_methane")


def test_levenshtein_random_22():

    target_smiles = "CC(=O)NCCc1c[nH]c2c1cc(cc2)OC"
    target_selfies = encoder(target_smiles)
    print(
        "This test attempts to generate melatonin from random molecules. \n The target SELFIES is : {0}".format(
            target_selfies
        )
    )

    solver = SelfiesGenAlgSolver(
        starting_random=True,
        n_genes=30,
        fitness_function=fitness_function_target_selfies(
            target_selfies, function_number=2
        ),  # See fitness_function_target_selfies
        max_gen=100,
        pop_size=50,
        n_crossover_points=1,
        random_state=420,
        logger_file="levenshtein_random.log",
        verbose=True,
    )
    solver.solve()
    print(
        "The best candidate has a Levenshtein distance of : {0}".format(
            solver.best_fitness_
        )
    )
    mol = sc2mol_structure(solver.best_individual_)
    mol_structure2depictions(mol, root_name="levenshtein_random")


if __name__ == "__main__":
    test_tanimoto_methane_19()
    test_tanimoto_random_20()
    test_levenshtein_methane_21()
    test_levenshtein_random_22()
