#!/usr/bin/env python3

from selfies import encoder
from simpleGA.selfies_solver import SelfiesGenAlgSolver
from simpleGA.fitness_functions_selfies import fitness_function_target_property
from simpleGA.chemistry import count_selfie_chars
from simpleGA.wrappers import (
    sc2smiles,
    sc2mv,
    sc2mol_structure,
    mol_structure2depictions,
)


def test_biphenyl():

    starting_smiles = "C1=CC=C(C=C1)C2=CC=CC=C2"
    starting_selfies = [encoder(starting_smiles)]
    n_starting_genes = count_selfie_chars(starting_selfies[0])
    print(
        "This test attempts to modify biphenyl to target a given property. \n The starting SMILES is : {0} \n The starting SELFIES is : {1} \n The number of genes required is : {2}".format(
            starting_smiles, starting_selfies, n_starting_genes
        )
    )

    # We will now tailor molecular volume to 350
    solver = SelfiesGenAlgSolver(
        starting_selfies=starting_selfies,  # We start the run from the biphenyl molecule
        n_genes=int(n_starting_genes * 2),  # We need at least n_starting_genes
        excluded_genes=list(
            range(n_starting_genes)
        ),  # We do not modify the biphenyl molecule
        fitness_function=fitness_function_target_property(
            target=350.0, function_number=6, score_modifier_number=2
        ),  # See fitness_functions_selfies, this is mv
        max_gen=50,  # This is a simple test and this run is more expensive
        pop_size=10,  # So we reduce the size of everything
        random_state=420,
        logger_level="INFO",
        logger_file="biphenyl_mv.log",
        verbose=True,
        progress_bars=True,
    )
    solver.solve()
    print(
        "The best candidate for mv tailoring has a fitness of : {0}".format(
            solver.best_fitness_
        )
    )
    print(
        "The corresponding SMILES is : {0}".format(sc2smiles(solver.best_individual_))
    )
    mol = sc2mol_structure(solver.best_individual_)
    mol_structure2depictions(mol, root_name="biphenyl_mv")
    print("The corresponding mv is : {0}".format(sc2mv(solver.best_individual_)))


if __name__ == "__main__":
    test_biphenyl()
