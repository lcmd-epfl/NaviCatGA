#!/usr/bin/env python3

from navicatGA.selfies_solver import SelfiesGenAlgSolver
from navicatGA.fitness_functions_selfies import fitness_function_target_property


def test_target_property_18():
    starting_selfies = ["[C][O]"]
    solver = SelfiesGenAlgSolver(
        n_genes=4,
        pop_size=10,
        max_gen=25,
        fitness_function=fitness_function_target_property(
            function_number=9, target=0.1
        ),  # homo-lumo gap
        starting_selfies=starting_selfies,
        excluded_genes=[0],
        random_state=666,
        logger_level="INFO",
        n_crossover_points=1,
        verbose=True,
        progress_bars=True,
        to_file=False,
        to_stdout=True,
    )
    solver.solve()
    solver.close_solver_logger()


if __name__ == "__main__":
    test_target_property_18()
