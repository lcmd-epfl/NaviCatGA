#!/usr/bin/env python3
from navicatGA.smiles_solver import SmilesGenAlgSolver
from navicatGA.wrappers_smiles import chromosome_to_smiles, smiles2logp, sc2depictions
import pandas as pd
import pickle
import os

database = os.path.join(os.path.dirname(os.path.realpath(__file__)), "database.xls")


def read_database(filename="database.xls") -> pd.DataFrame:
    # global fdb
    fdb = pd.read_excel(filename)
    fdb.dropna()
    return fdb


# Get lists from xls
fdb = read_database(database)
alphabet_list = list(set(fdb.Substituent_1.dropna()))

starting_smiles = [["[Fe]"]]


def my_fitness_function():
    return lambda smiles: (smiles2logp(smiles))


def test_pickle_25():
    solver = SmilesGenAlgSolver(
        n_genes=7,
        pop_size=10,
        mutation_rate=0.15,
        prune_duplicates=True,
        fitness_function=my_fitness_function(),
        chromosome_to_smiles=chromosome_to_smiles(),
        starting_population=starting_smiles,
        alphabet_list=alphabet_list,
        random_state=24,
        excluded_genes=[0],
        logger_level="DEBUG",
        n_crossover_points=1,
        verbose=True,
        to_file=False,
        to_stdout=True,
    )
    solver.solve(10)
    file = open("solver.pkl", "wb")
    solver.fitness_function = None
    solver.assembler = None
    solver.chromosome_to_smiles = None
    pickle.dump(solver, file, pickle.HIGHEST_PROTOCOL)
    del solver
    file.close()
    file = open("solver.pkl", "rb")
    solver = pickle.load(file)
    file.close()
    solver.fitness_function = my_fitness_function()
    solver.assembler = chromosome_to_smiles()
    print(
        f"Solver has still n_genes {solver.n_genes} and pop_size {solver.pop_size} and so on."
    )
    solver.solve(10)
    sc2depictions(solver.best_individual_, "best_fe_soluble")
    solver.close_solver_logger()


if __name__ == "__main__":
    test_pickle_25()
