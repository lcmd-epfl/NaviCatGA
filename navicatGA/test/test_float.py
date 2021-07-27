#!/usr/bin/env python3

from simpleGA.continuous_solver import ContinuousGenAlgSolver
from simpleGA.fitness_functions_float import fitness_function_float


def test_float_08():
    solver = ContinuousGenAlgSolver(
        n_genes=6,
        pop_size=50,
        max_gen=50,
        mutation_rate=0.05,
        selection_rate=0.25,
        variables_limits=(0, 1),
        fitness_function=fitness_function_float(10),
        selection_strategy="roulette_wheel",
        n_crossover_points=1,
        random_state=420,
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


def test_float_09():
    solver = ContinuousGenAlgSolver(
        n_genes=6,
        pop_size=50,
        max_gen=50,
        mutation_rate=0.05,
        selection_rate=0.25,
        variables_limits=(0, 1),
        fitness_function=fitness_function_float(10),
        selection_strategy="tournament",
        n_crossover_points=2,
        random_state=420,
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


def test_float_10():
    solver = ContinuousGenAlgSolver(
        n_genes=6,
        pop_size=50,
        max_gen=50,
        mutation_rate=0.05,
        selection_rate=0.25,
        variables_limits=(0, 1),
        fitness_function=fitness_function_float(10),
        selection_strategy="random",
        n_crossover_points=3,
        random_state=420,
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
    test_float_08()
    test_float_09()
    test_float_10()
