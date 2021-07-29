#!/usr/bin/env python3
import os
from navicatGA.xyz_solver import XYZGenAlgSolver
from navicatGA.quantum_wrappers_xyz import geom2ehl
from navicatGA.chemistry_xyz import get_alphabet_from_path, get_default_alphabet
from chimera import Chimera

# For multiobjective optimization we use chimera to scalarize
chimera = Chimera(tolerances=[0.1, 0.4], absolutes=[False, False], goals=["max", "min"])

database = os.path.join(os.path.dirname(os.path.realpath(__file__)), "scaffolds/")


def my_fitness_function(lot=0):
    return lambda geom: (geom2ehl(geom, lot=lot))


alphabet_list = [
    get_alphabet_from_path(database),
    get_default_alphabet(),
    get_default_alphabet(),
]


def test_scalarizer_26():
    print(alphabet_list[0])
    solver = XYZGenAlgSolver(
        n_genes=3,
        pop_size=5,
        max_gen=5,
        mutation_rate=0.25,
        selection_rate=0.15,
        fitness_function=my_fitness_function(lot=0),
        alphabet_list=alphabet_list,
        scalarizer=chimera,
        random_state=1337,
        starting_random=True,
        logger_level="TRACE",
        selection_strategy="boltzmann",
        prune_duplicates=True,
        n_crossover_points=1,
        verbose=True,
        to_stdout=True,
        to_file=False,
        show_stats=True,
    )
    solver.solve()
    print(solver.printable_fitness)


if __name__ == "__main__":
    test_scalarizer_26()
