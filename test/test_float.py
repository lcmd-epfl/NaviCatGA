#!/usr/bin/env python3

from continuous_solver import ContinuousGenAlgSolver
from fitness_functions_float import fitness_function_float


def test_float():
    solver = ContinuousGenAlgSolver(
        n_genes=6,
        pop_size=100,
        max_gen=250,
        mutation_rate=0.15,
        selection_rate=0.25,
        variables_limits=(0, 1),
        fitness_function=fitness_function_float(10),
        selection_strategy="tournament",
        to_file=False,
        progress_bars=True,
        verbose=True,
    )
    solver.solve()
    print(
        "The maximum of the Hartmann6 function is found at (0.2, 0.15, 0.47, 0.27, 0.31, 0.65)"
    )
    print(
        "The GA run found a maxima at ({0}, {1}, {2}, {3}, {4}, {5})".format(
            solver.best_individual_[0],
            solver.best_individual_[1],
            solver.best_individual_[2],
            solver.best_individual_[3],
            solver.best_individual_[4],
            solver.best_individual_[5],
        )
    )


if __name__ == "__main__":
    test_float()
