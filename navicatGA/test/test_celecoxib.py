#!/usr/bin/env python3

from selfies import encoder
from navicatGA.selfies_solver import SelfiesGenAlgSolver
from navicatGA.fitness_functions_selfies import fitness_function_target_property
from navicatGA.chemistry_selfies import count_selfie_chars
from navicatGA.wrappers_selfies import (
    sc2smiles,
    sc2logp,
    sc2mol_structure,
    mol_structure2depictions,
)


def test_celecoxib_07():

    starting_smiles = "CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(=O)(=O)N)C(F)(F)F"
    starting_selfies = [encoder(starting_smiles)]
    n_starting_genes = count_selfie_chars(starting_selfies[0])
    print(
        "This test attempts to generate celecoxib derivatives to target a given property. \n The starting SMILES is : {0} \n The starting SELFIES is : {1} \n The number of genes required is : {2}".format(
            starting_smiles, starting_selfies, n_starting_genes
        )
    )

    # We will now tailor logp to 0.5
    solver = SelfiesGenAlgSolver(
        starting_selfies=starting_selfies,  # We start the run from the celecoxib molecule
        starting_stoned=True,  # We used the STONED algorithm to generate a starting subspace of celecoxib
        n_genes=int(n_starting_genes * 2),  # We need at least n_starting_genes
        excluded_genes=list(
            range(n_starting_genes)
        ),  # We do not modify some of the genes of the molecule
        fitness_function=fitness_function_target_property(
            target=0.5, function_number=1, score_modifier_number=3
        ),  # See fitness_functions_selfies, this is logp
        max_gen=25,  # This is a simple test and this run is more expensive
        pop_size=10,  # So we reduce the size of everything
        random_state=666,
        logger_level="INFO",
        logger_file="celecoxib_logp.log",
        verbose=True,
        progress_bars=True,
    )
    solver.solve()
    print(
        "The best candidate for logp tailoring has a fitness of : {0}".format(
            solver.best_fitness_
        )
    )
    print(
        "The corresponding SMILES is : {0}".format(sc2smiles(solver.best_individual_))
    )
    mol = sc2mol_structure(solver.best_individual_)
    mol_structure2depictions(mol, root_name="celecoxib_logp")
    print("The corresponding logp is : {0}".format(sc2logp(solver.best_individual_)))
    solver.close_solver_logger()


if __name__ == "__main__":
    test_celecoxib_07()
