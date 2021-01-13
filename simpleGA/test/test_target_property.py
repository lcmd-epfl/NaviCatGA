#!/usr/bin/env python3

from simpleGA.selfies_solver import SelfiesGenAlgSolver
from simpleGA.fitness_functions_selfies import fitness_function_target_property


def test_target_property():
    starting_selfies = ["[C][O]"]
    solver = SelfiesGenAlgSolver(
        n_genes=10,
        pop_size=50,
        max_gen=100,
        fitness_function=fitness_function_target_property(
            function_number=1, target=0.5
        ),  # logp target
        starting_selfies=starting_selfies,
        excluded_genes=[0, 1],
        random_state=66,
        logger_level="INFO",
        n_crossover_points=1,
        verbose=True,
        progress_bars=True,
        to_file=False,
        to_stdout=True,
    )
    solver.solve()


if __name__ == "__main__":
    test_target_property()
