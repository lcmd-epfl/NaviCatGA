#!/usr/bin/env python3
import os
from navicatGA.xyz_solver import XYZGenAlgSolver
from navicatGA.chemistry_xyz import get_alphabet_from_path, get_default_alphabet
from navicatGA.wrappers_xyz import geom2dihedral, geom2sub_sterimol, chromosome_to_xyz
from navicatGA.quantum_wrappers_xyz import geom2ehomo


def my_fitness_function():
    return lambda geom: (mlr(geom))


database = os.path.join(os.path.dirname(os.path.realpath(__file__)), "scaffolds/")

alphabet_list = [
    get_alphabet_from_path(database),
    get_default_alphabet(),
    get_default_alphabet(),
]


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
        pop_size=10,
        max_gen=5,
        mutation_rate=0.15,
        fitness_function=my_fitness_function(),
        chromosome_to_xyz=chromosome_to_xyz(),
        alphabet_list=alphabet_list,
        random_state=24,
        starting_random=True,
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
    test_scaffolds_23()
