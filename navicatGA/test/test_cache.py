#!/usr/bin/env python3

from selfies import encoder
from navicatGA.selfies_solver import SelfiesGenAlgSolver
from navicatGA.fitness_functions_selfies import fitness_function_selfies
from navicatGA.chemistry_selfies import count_selfie_chars
from navicatGA.wrappers_selfies import sc2smiles


def test_ibuprofen_mv_06(lru_cache=False):
    starting_smiles = "CC(C)Cc1ccc(cc1)[C@](C)C(=O)O"  # Ibuprofen
    starting_selfies = [encoder(starting_smiles)]
    n_starting_genes = count_selfie_chars(starting_selfies[0])
    print(
        "This test attempts to modify ibuprofen to improve a given property. \n The starting SMILES is : {0} \n The starting SELFIES is : {1} \n The number of genes required is : {2}".format(
            starting_smiles, starting_selfies, n_starting_genes
        )
    )

    # We will now maximize molecular volume
    solver = SelfiesGenAlgSolver(
        starting_selfies=starting_selfies,  # We start the run from the melatonin molecule
        n_genes=int(n_starting_genes * 2),  # We need at least n_starting_genes
        excluded_genes=list(
            range(n_starting_genes)
        ),  # We do not modify the core structure
        fitness_function=fitness_function_selfies(
            6
        ),  # See fitness_functions_selfies, this is mv
        max_gen=10,  # This is a simple test and this run is more expensive
        pop_size=100,
        random_state=666,
        lru_cache=lru_cache,
        logger_level="INFO",
        logger_file="ibuprofen_mv.log",
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
    print("Runtime was : {0} seconds.".format(solver.runtime_))
    return solver.runtime_


if __name__ == "__main__":
    t1 = test_ibuprofen_mv_06(lru_cache=False)
    t2 = test_ibuprofen_mv_06(lru_cache=True)
    assert t1 > t2
    print("{0} seconds saved by cacheing.".format(t1 - t2))
