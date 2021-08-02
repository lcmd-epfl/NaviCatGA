#!/usr/bin/env python3

from navicatGA.selfies_solver import SelfiesGenAlgSolver
from navicatGA.score_modifiers import score_modifier
from navicatGA.wrappers_selfies import (
    sc2smiles,
    sc2mol_structure,
    mol_structure2depictions,
)
from navicatGA.quantum_wrappers_selfies import sc2gap
from navicatGA.wrappers_selfies import sc2logp, sc2mw


# In this test, we dont use a chimera scalarizer and we simply define a combined fitness function
def fitness_function_wrapper(target_1, target_2, target_3):

    return (
        lambda chromosome: (
            0.4 * score_modifier(sc2gap(chromosome, lot=0), target_1, 3)
            + 0.4 * score_modifier(sc2logp(chromosome), target_2, 1)
            + 0.2 * score_modifier(sc2mw(chromosome), target_3, 3)
        )
        / 3
    )


def test_real_application_16():
    starting_selfies = ["[C][O][=C][C][=N][Ring_1]"]
    solver = SelfiesGenAlgSolver(
        n_genes=15,
        pop_size=10,
        max_gen=10,
        fitness_function=fitness_function_wrapper(
            target_1=0.05, target_2=0.1, target_3=65
        ),  # homo-lumo gap, logp, mw
        starting_selfies=starting_selfies,
        starting_stoned=True,
        prune_duplicates=True,
        mutation_rate=0.05,
        selection_rate=0.4,
        random_state=666,
        n_crossover_points=1,
        verbose=False,
        progress_bars=True,
        to_file=True,
        selection_strategy="boltzmann",
        to_stdout=False,
        logger_level="INFO",
        logger_file="real_application.log",
        show_stats=True,
    )
    solver.solve()
    print(
        "After optimization, the corresponding SMILES is : {0}".format(
            sc2smiles(solver.best_individual_)
        )
    )
    print(
        "It has properties: \n HOMO-LUMO gap : {0} \n LogP : {1} \n Molecular weight : {2}".format(
            sc2gap(solver.best_individual_),
            sc2logp(solver.best_individual_),
            sc2mw(solver.best_individual_),
        )
    )
    mol = sc2mol_structure(solver.best_individual_)
    mol_structure2depictions(mol, "real_application")
    solver.close_solver_logger()


if __name__ == "__main__":
    test_real_application_16()
