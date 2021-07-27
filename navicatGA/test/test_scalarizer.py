#!/usr/bin/env python3
from simpleGA.xyz_solver import XYZGenAlgSolver
from simpleGA.wrappers_xyz import gl2geom
from simpleGA.quantum_wrappers_xyz import geom2ehl
from chimera import Chimera
import os

# For multiobjective optimization we use chimera to scalarize
chimera = Chimera(tolerances=[0.1, 0.4], absolutes=[False, False], goals=["min", "max"])


def my_fitness_function(lot=0):
    return lambda chromosome: (hff(gl2geom(chromosome)[1]))


def hff(geom, lot=0):
    ehl = geom2ehl(geom, lot)
    return ehl


def test_scalarizer_26():
    solver = XYZGenAlgSolver(
        n_genes=3,
        pop_size=4,
        max_gen=5,
        mutation_rate=0.05,
        selection_rate=0.15,
        fitness_function=my_fitness_function(lot=0),
        scalarizer=chimera,
        path_scaffolds="scaffolds",
        random_state=1337,
        starting_random=True,
        logger_level="DEBUG",
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
