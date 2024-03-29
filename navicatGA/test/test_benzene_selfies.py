#!/usr/bin/env python3

from navicatGA.selfies_solver import SelfiesGenAlgSolver
from navicatGA.fitness_functions_selfies import fitness_function_selfies


def test_benzene_selfies_04():
    starting_selfies = ["[C][C=][C][C=][C][C=][Ring1][Branch1_2]"]
    solver = SelfiesGenAlgSolver(
        n_genes=16,
        pop_size=10,
        max_gen=25,
        fitness_function=fitness_function_selfies(2),
        starting_selfies=starting_selfies,
        excluded_genes=[0, 1, 2, 3, 4, 5, 6, 7, 8],
        random_state=420,
        logger_level="INFO",
        n_crossover_points=1,
        verbose=False,
        progress_bars=True,
        to_file=False,
        to_stdout=True,
    )
    solver.solve()
    solver.close_solver_logger()


if __name__ == "__main__":
    test_benzene_selfies_04()
