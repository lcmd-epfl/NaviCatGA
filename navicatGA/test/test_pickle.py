#!/usr/bin/env python3
from navicatGA.smiles_solver import SmilesGenAlgSolver
from navicatGA.wrappers_smiles import sc2smiles, sc2mol_structure, sc2logp, sc2depictions
import pandas as pd
import pickle


def read_database(filename="database.xls") -> pd.DataFrame:
    # global fdb
    fdb = pd.read_excel(filename)
    fdb.dropna()
    return fdb


# Get lists from xls
fdb = read_database("database.xls")
alphabet_list = list(set(fdb.Substituent_1.dropna()))

starting_smiles = ["[Fe@]"]


def my_fitness_function():
    return lambda chromosome: (krr(chromosome))


def krr(chromosome):
    val = sc2logp(chromosome)
    return val


def test_pickle_25():
    solver = SmilesGenAlgSolver(
        n_genes=7,
        pop_size=50,
        mutation_rate=0.15,
        prune_duplicates=True,
        fitness_function=my_fitness_function(),
        starting_smiles=starting_smiles,
        alphabet_list=alphabet_list,
        random_state=24,
        starting_random=True,
        logger_level="INFO",
        n_crossover_points=1,
        verbose=True,
        to_file=False,
        to_stdout=True,
    )
    solver.solve(10)
    file = open("solver.pkl", "wb")
    solver.fitness_function = None
    pickle.dump(solver, file, pickle.HIGHEST_PROTOCOL)
    del solver
    file.close()
    file = open("solver.pkl", "rb")
    solver = pickle.load(file)
    file.close()
    solver.fitness_function = my_fitness_function()
    solver.solve(10)
    sc2depictions(solver.best_individual_, "best_fe_soluble")


if __name__ == "__main__":
    test_pickle_25()
