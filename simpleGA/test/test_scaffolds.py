#!/usr/bin/env python3
from simpleGA.xyz_solver import XYZGenAlgSolver
from simpleGA.wrappers_xyz import gl2geom
from simpleGA.wrappers_xyz import geom2dihedral, geom2sub_sterimol
from simpleGA.quantum_wrappers_xyz import geom2ehomo


def my_fitness_function():
    return lambda chromosome: (mlr(chromosome))


def mlr(chromosome):
    geom = gl2geom(chromosome)[1]
    dihe = geom2dihedral(geom, "C", "O", "B", "C")
    ehomo = geom2ehomo(geom)
    b1 = geom2sub_sterimol(geom, "36", "B1")
    print(
        "Fitness evaluated as 0.4*{0} + 0.31*{1} + 11.77/{2} + 26.59".format(
            dihe, ehomo, b1
        )
    )
    return 0.4 * dihe + 0.31 * ehomo + 11.77 / b1 + 26.59


def test_scaffolds_24():
    solver = XYZGenAlgSolver(
        n_genes=5,
        pop_size=10,
        max_gen=5,
        mutation_rate=0.05,
        fitness_function=my_fitness_function(),
        path_scaffolds="scaffolds",
        random_state=420,
        starting_random=True,
        logger_level="INFO",
        n_crossover_points=1,
        verbose=True,
        progress_bars=True,
        to_file=False,
        to_stdout=True,
    )
    solver.solve()
    geom = gl2geom(solver.best_individual_)[1]
    geom.write("substituted_scaffold")


if __name__ == "__main__":
    test_scaffolds_24()
