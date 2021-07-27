#!/usr/bin/env python3
from navicatGA.smiles_solver import SmilesGenAlgSolver
from navicatGA.wrappers_smiles import sc2smiles, sc2mol_structure, sc2mw, sc2depictions
import pandas as pd
import os


database = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),"database.xls"
)


def read_database(filename="database.xls") -> pd.DataFrame:
    # global fdb
    fdb = pd.read_excel(filename)
    fdb.dropna()
    return fdb


# Get lists from xls
fdb = read_database(database)
alphabet_list = list(set(fdb.Substituent_1.dropna()))

starting_smiles = ["[Ni@]"]


def my_fitness_function():
    return lambda chromosome: (krr(chromosome))


def krr(chromosome):
    val = sc2mw(chromosome)
    return val


def test_ni_catalyst_24():
    solver = SmilesGenAlgSolver(
        n_genes=7,
        pop_size=50,
        max_gen=10,
        mutation_rate=0.05,
        fitness_function=my_fitness_function(),
        starting_smiles=starting_smiles,
        alphabet_list=alphabet_list,
        random_state=24,
        starting_random=True,
        logger_level="INFO",
        n_crossover_points=2,
        verbose=True,
        progress_bars=True,
        to_file=False,
        to_stdout=True,
    )
    solver.solve()
    sc2depictions(solver.best_individual_, "best_ni_catalyst")


if __name__ == "__main__":
    test_ni_catalyst_24()
