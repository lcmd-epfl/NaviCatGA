from typing import Sequence
import logging
import random
import numpy as np

from selfies import (
    get_alphabet_from_selfies,
    get_semantic_robust_alphabet,
    set_semantic_constraints,
)
from simpleGA.genetic_algorithm_base import GenAlgSolver
from simpleGA.evo import (
    sanitize_multiple_smiles,
    get_selfie_chars,
    check_selfie_chars,
)
from simpleGA.fitness_functions_selfies import fitness_function_selfies
from rdkit import rdBase


logger = logging.getLogger(__name__)


class SelfiesGenAlgSolver(GenAlgSolver):
    def __init__(
        self,
        n_genes: int,
        starting_selfies: str = "[nop]",
        starting_random: bool = False,
        alphabet_list: list = list(get_semantic_robust_alphabet()),
        fitness_function=None,
        max_gen: int = 500,
        pop_size: int = 100,
        mutation_rate: float = 0.25,
        selection_rate: float = 0.25,
        selection_strategy: str = "tournament",
        verbose: bool = True,
        show_stats: bool = False,
        plot_results: bool = False,
        excluded_genes: Sequence = None,
        variables_limits: dict = None,
        problem_type=str,
        n_crossover_points: int = 1,
        branching: bool = False,
        max_counter: int = 10,
        random_state: int = None,
        logger_file: str = "output.log",
        logger_level: str = "INFO",
        to_stdout: bool = True,
        to_file: bool = True,
        progress_bars: bool = False,
    ):

        GenAlgSolver.__init__(
            self,
            fitness_function=fitness_function,
            n_genes=n_genes,
            max_gen=max_gen,
            pop_size=pop_size,
            mutation_rate=mutation_rate,
            selection_rate=selection_rate,
            selection_strategy=selection_strategy,
            verbose=verbose,
            show_stats=show_stats,
            plot_results=plot_results,
            excluded_genes=excluded_genes,
            n_crossover_points=n_crossover_points,
            random_state=random_state,
            logger_file=logger_file,
            logger_level=logger_level,
            to_stdout=to_stdout,
            to_file=to_file,
            progress_bars=progress_bars,
        )

        if variables_limits is not None:
            set_semantic_constraints(variables_limits)

        self.branching = branching
        if self.branching:
            tuples = [
                (i + 1, j + 1) for i in range(self.n_genes) for j in range(self.n_genes)
            ]
            for i in tuples:
                if i[0] == i[1]:
                    pass
                else:
                    # alphabet.add("[Branch{0}_{1}]".format(i[0], i[1]))
                    pass
        self.alphabet = list(sorted(alphabet_list))
        self.problem_type = problem_type

        if starting_random:
            starting_selfies = ""
            for i in range(random.randint(1, n_genes)):
                starting_selfies += np.random.choice(self.alphabet, size=1)[0]
        self.starting_selfies = starting_selfies
        self.max_counter = int(max_counter)

    def initialize_population(self):
        """
        Initializes the population of the problem according to the
        population size and number of genes and according to the problem
        type (SELFIES elements here).
        :return: a numpy array with a sanitized initialized population
        """

        population = np.zeros(shape=(self.pop_size, self.n_genes), dtype="object")
        chromosome = get_selfie_chars(self.starting_selfies, self.n_genes)
        assert check_selfie_chars(chromosome)
        for i in range(self.pop_size):
            population[i][:] = chromosome[0 : self.n_genes]
        return population

    def get_crossover_points(self):
        """
        Retrieves random crossover points
        :return: a numpy array with the crossover points
        """

        return np.sort(
            np.random.choice(
                np.arange(self.n_genes), self.n_crossover_points, replace=False
            )
        )

    def create_offspring(
        self, first_parent, sec_parent, crossover_pt, offspring_number
    ):
        """
        Creates an offspring from 2 parents.
        """
        beta = np.random.rand(1)[0]
        gamma = np.random.rand(1)[0]
        backup_sec_parent = sec_parent
        backup_first_parent = first_parent
        if self.allowed_mutation_genes is not None:
            mask_allowed = np.zeros(first_parent.size, dtype=bool)
            mask_forbidden = np.ones(first_parent.size, dtype=bool)
            mask_allowed[self.allowed_mutation_genes] = True
            mask_forbidden[self.allowed_mutation_genes] = False
            full_offspring = np.empty_like(first_parent, dtype=object)
            full_offspring[mask_forbidden] = first_parent[mask_forbidden]
            first_parent = first_parent[mask_allowed]
            sec_parent = sec_parent[mask_allowed]

        offspring = np.empty_like(first_parent, dtype=object)
        valid_smiles = False

        if beta < 0.5:
            first_parent = first_parent[::-1]
        if gamma > 0.5:
            sec_parent = sec_parent[::-1]

        if offspring_number == "first":
            while not valid_smiles:
                offspring = np.empty_like(first_parent, dtype=object)
                counter = 0
                c = int(0)
                offspring[c : crossover_pt[0]] = first_parent[c : crossover_pt[0]]
                offspring[crossover_pt[0] :] = sec_parent[crossover_pt[0] :]
                for ci, cj in zip(crossover_pt[::2], crossover_pt[1::2]):
                    offspring[c:ci] = first_parent[c:ci]
                    offspring[ci:] = sec_parent[ci:]
                    c = cj
                if self.allowed_mutation_genes is not None:
                    full_offspring[mask_allowed] = offspring[:]
                    offspring = full_offspring
                logger.debug(
                    "Offspring chromosome attempt {0}: {1}".format(counter, offspring)
                )
                valid_smiles = check_selfie_chars(offspring)
                crossover_pt = self.get_crossover_points()
                counter += 1
                if counter > self.max_counter:
                    logger.debug(
                        "Counter in create offspring exceeded {0}, using default.".format(
                            self.max_counter
                        )
                    )
                    valid_smiles = True
                    offspring = backup_first_parent
            logger.debug("Final offspring chromosome: {0}".format(offspring))
            return offspring

        if offspring_number == "second":
            while not valid_smiles:
                offspring = np.empty_like(first_parent, dtype=object)
                counter = 0
                c = int(0)
                offspring[c : crossover_pt[0]] = sec_parent[c : crossover_pt[0]]
                offspring[crossover_pt[0] :] = first_parent[crossover_pt[0] :]
                for ci, cj in zip(crossover_pt[::2], crossover_pt[1::2]):
                    if not ci:
                        ci = crossover_pt[0]
                    offspring[c:ci] = sec_parent[c:ci]
                    offspring[ci:] = first_parent[ci:]
                    c = cj
                if self.allowed_mutation_genes is not None:
                    full_offspring[mask_allowed] = offspring[:]
                    offspring = full_offspring
                logger.debug(
                    "Offspring chromosome attempt {0}: {1}".format(counter, offspring)
                )
                valid_smiles = check_selfie_chars(offspring)
                crossover_pt = self.get_crossover_points()
                counter += 1
                if counter > self.max_counter:
                    logger.debug(
                        "Counter in create offspring exceeded {0}, using default.".format(
                            self.max_counter
                        )
                    )
                    valid_smiles = True
                    offspring = backup_sec_parent
            logger.debug("Final offspring chromosome: {0}".format(offspring))
            return offspring

    def mutate_population(self, population, n_mutations):
        """
        Mutates the population by randomizing specific positions of the
        population individuals.
        :param population: the population at a given iteration
        :param n_mutations: number of mutations to be performed.
        :return: the mutated population
        """

        valid_smiles = False
        mutation_rows, mutation_cols = super(
            SelfiesGenAlgSolver, self
        ).mutate_population(population, n_mutations)
        for i, j in zip(mutation_rows, mutation_cols):
            counter = 0
            while not valid_smiles:
                backup_gene = population[i, j]
                population[i, j] = np.random.choice(self.alphabet, size=1)[0]
                logger.debug(
                    "Mutated chromosome attempt {0}: {1}".format(
                        counter, population[i, :]
                    )
                )
                valid_smiles = check_selfie_chars(population[i, :])
                counter += 1
                if counter > self.max_counter:
                    logger.debug(
                        "Counter in mutate exceeded {0}, using default.".format(
                            self.max_counter
                        )
                    )
                    population[i, j] = backup_gene
                    valid_smiles = True
            valid_smiles = False

        return population


def test_benzene():
    starting_selfies = "[C][C=][C][C=][C][C=][Ring1][Branch1_2]"
    solver = SelfiesGenAlgSolver(
        n_genes=16,
        pop_size=25,
        max_gen=100,
        fitness_function=fitness_function_selfies(2),
        starting_selfies=starting_selfies,
        excluded_genes=[0, 1, 2, 3, 4, 5, 6, 7, 8],
        logger_level="INFO",
        n_crossover_points=2,
        verbose=False,
        progress_bars=True,
        to_file=False,
        to_stdout=True,
    )
    solver.solve()


if __name__ == "__main__":
    test_benzene()
