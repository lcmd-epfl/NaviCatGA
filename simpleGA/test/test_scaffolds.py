#!/usr/bin/env python3
from simpleGA.xyz_solver import XYZGenAlgSolver
from simpleGA.wrappers_xyz import gl2geom
from simpleGA.wrappers_xyz import geom2dihedral, geom2sub_sterimol
from simpleGA.quantum_wrappers_xyz import geom2ehomo


def my_fitness_function():
    return lambda chromosome: (mlr(gl2geom(chromosome)[1]))


def mlr(geom, lot=0):
    ehomo = geom2ehomo(geom, lot)
    dihe = geom2dihedral(geom, "2", "1", "18", "21")
    b1 = (geom2sub_sterimol(geom, "19", "B1") + geom2sub_sterimol(geom, "20", "B1")) / 2
    val = 0.4 * dihe + 0.31 * ehomo + 11.77 / b1 + 26.59
    print(
        "Fitness evaluated as 0.4*{0} + 0.31*{1} + 11.77/{2} + 26.59 = {3}".format(
            dihe, ehomo, b1, val
        )
    )
    return -val


def test_scaffolds_23():
    solver = XYZGenAlgSolver(
        n_genes=3,
        pop_size=5,
        max_gen=2,
        mutation_rate=0.05,
        fitness_function=my_fitness_function(),
        path_scaffolds="scaffolds",
        random_state=24,
        starting_random=True,
        logger_level="DEBUG",
        n_crossover_points=1,
        verbose=True,
        progress_bars=True,
        to_file=False,
        to_stdout=True,
    )
    solver.solve()


if __name__ == "__main__":
    test_scaffolds_23()
