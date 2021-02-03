#!/usr/bin/env python3

from simpleGA.xyz_solver import XYZGenAlgSolver
from simpleGA.fitness_functions_xyz import fitness_function_xyz
from simpleGA.chemistry_xyz import (
    get_starting_xyz_from_smi,
)
from simpleGA.wrappers_xyz import gl2geom


def test_benzene_xyz_23():
    starting_smiles = ["C1=CC=CC=C1"]
    starting_xyz = get_starting_xyz_from_smi(starting_smiles)
    solver = XYZGenAlgSolver(
        n_genes=4,
        pop_size=10,
        max_gen=10,
        mutation_rate=0.05,
        fitness_function=fitness_function_xyz(1),
        starting_xyz=starting_xyz,
        random_state=420,
        starting_random=True,
        logger_level="INFO",
        n_crossover_points=1,
        verbose=False,
        progress_bars=True,
        to_file=False,
        to_stdout=True,
    )
    solver.solve()
    geom = gl2geom(solver.best_individual_)[1]
    geom.write("substituted_benzene")


if __name__ == "__main__":
    test_benzene_xyz_23()
