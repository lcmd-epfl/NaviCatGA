#!/usr/bin/env python3

from continuous_solver import ContinuousGenAlgSolver
from fitness_functions_float import fitness_function_float

solver = ContinuousGenAlgSolver(
    n_genes=6,
    pop_size=50,
    max_gen=250,
    mutation_rate=0.05,
    selection_rate=0.5,
    variables_limits=(0, 1),
    fitness_function=fitness_function_float(10),
    selection_strategy="roulette_wheel",
)
solver.solve()
print(solver.best_individual_)
print(solver.best_fitness_)
