#!/usr/bin/env python3
from simpleGA.smiles_solver import SmilesGenAlgSolver
from simpleGA.wrappers_smiles import sc2smiles, sc2mol_structure, sc2mw
import pandas as pd


def read_database(filename="database.xls") -> pd.DataFrame:
    # global fdb
    fdb = pd.read_excel(filename)
    fdb.dropna()
    return fdb


# Get lists from xls
fdb = read_database("database.xls")
substituent_list = list(set(fdb.Substituent_1.dropna()))
# id_1 = fdb.Id_1
print(substituent_list)

starting_smiles = ["[Ni@]"]


def my_fitness_function():
    return lambda chromosome: (krr(chromosome))


def krr(chromosome):
    val = sc2mw(chromosome)
    return val


def test_ni_catalyst_24():
    solver = SmilesGenAlgSolver(
        n_genes=7,
        pop_size=5,
        max_gen=5,
        mutation_rate=0.05,
        fitness_function=my_fitness_function(),
        starting_smiles=starting_smiles,
        substituent_list=substituent_list,
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
    test_ni_catalyst_24()
