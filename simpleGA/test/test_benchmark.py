#!/usr/bin/env python3

import numpy as np
from selfies import encoder
from simpleGA.selfies_solver import SelfiesGenAlgSolver
from simpleGA.exceptions import InvalidInput
from simpleGA.fitness_functions_selfies import fitness_function_target_selfies
from simpleGA.wrappers_selfies import sc2selfies, sc2tanimoto_to_target, sc2depictions


def test_rediscover_paracetamol(
    n_crossover_points=2, mutation_rate=0.10, max_gen=250, pop_size=100
):

    target_smiles = "CC(=O)Nc1ccc(cc1)O"  # Paracetamol
    target_selfies = encoder(target_smiles)
    print(
        "This test attempts to generate paracetamol from random using {0} crossover points and a mutation rate of {1}.\nThe target SELFIES is : {2}".format(
            n_crossover_points, mutation_rate, target_selfies
        )
    )

    solver = SelfiesGenAlgSolver(
        starting_random=True,
        n_genes=17,
        fitness_function=fitness_function_target_selfies(
            target_selfies, function_number=2
        ),  # See fitness_function_target_selfies, levenshtein distance here
        max_gen=max_gen,
        pop_size=pop_size,
        mutation_rate=mutation_rate,
        n_crossover_points=n_crossover_points,
        random_state=133742,
        logger_file="benchmark_{0}_{1}_{2}.log".format(
            n_crossover_points, mutation_rate, max_gen
        ),
        verbose=False,
        progress_bars=True,
    )
    solver.solve()
    print(
        "The best candidate has a Tanimoto similarity to target of : {0}".format(
            sc2tanimoto_to_target(solver.best_individual_, target_selfies)
        )
    )
    print("       Its SELFIES is : {0}".format(sc2selfies(solver.best_individual_)))
    sc2depictions(
        solver.best_individual_,
        "benchmark_{0}_{1}_{2}".format(n_crossover_points, mutation_rate, max_gen),
    )
    return sc2tanimoto_to_target(solver.best_individual_, target_selfies)


def test_crossover_points_02():
    print("\n\nTesting different number of crossover points:\n")
    tanimoto_list = []
    for i in range(4):
        try:
            tanimoto = test_rediscover_paracetamol(
                n_crossover_points=i, mutation_rate=0.15, max_gen=50, pop_size=25
            )
            tanimoto_list.append(tanimoto)
        except InvalidInput:
            print("n_crossover_points < 1 is not a valid input!")
    print(
        "The maximum Tanimoto similarity was {0} in iteration {1}\n\n".format(
            max(tanimoto_list), tanimoto_list.index(max(tanimoto_list))
        )
    )


def test_mutation_rate_03():
    print("\n\nTesting different mutation rates:\n")
    tanimoto_list = []
    for j in np.linspace(0.05, 0.25, 5, endpoint=False):
        mutation_rate = np.round(j, 2)
        tanimoto = test_rediscover_paracetamol(
            n_crossover_points=1, mutation_rate=mutation_rate, max_gen=50, pop_size=25
        )
        tanimoto_list.append(tanimoto)
    print(
        "The maximum Tanimoto similarity was {0} in iteration {1}\n\n".format(
            max(tanimoto_list), tanimoto_list.index(max(tanimoto_list))
        )
    )


if __name__ == "__main__":
    test_crossover_points_02()
    test_mutation_rate_03()
