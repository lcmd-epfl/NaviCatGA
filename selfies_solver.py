from typing import Sequence
import logging
import numpy as np

from genetic_algorithm_base import GenAlgSolver
from utils.helpers import get_input_dimensions
from chemistry.evo import (
    sanitize_multiple_smiles,
    encode_smiles,
    get_selfie_chars,
    check_selfie_chars,
    sc2mol_structure,
    mol_structure2depictions,
)
from selfies import get_alphabet_from_selfies
from fitness_functions_selfies import fitness_function_selfies
from rdkit import rdBase


logger = logging.getLogger(__name__)


class SelfiesGenAlgSolver(GenAlgSolver):
    def __init__(
        self,
        n_genes: int,
        starting_selfies: list = "[C]",
        alphabet_list: list = [
            "[epsilon]",
            "[C]",
            "[=C]",
            "[#C]",
            "[O]",
            "[=O]",
            "[N]",
            "[=N]",
            "[#N]",
            "[S]",
            "[=S]",
            "[P]",
            "[=P]",
            "[F]",
            "[Cl]",
            "[Br]",
            "[I]",
            "[Ring1]",
            "[Ring2]",
            "[Branch1_1]",
        ],
        fitness_function=None,
        max_gen: int = 500,
        pop_size: int = 100,
        mutation_rate: float = 0.45,
        selection_rate: float = 0.25,
        selection_strategy: str = "tournament",
        verbose: bool = True,
        show_stats: bool = False,
        plot_results: bool = False,
        excluded_genes: Sequence = None,
        variables_limits=(-10, 10),
        problem_type=str,
        n_crossover_points: int = 1,
        branching: bool = False,
        max_counter: int = 10,
        random_state: int = None,
    ):
        """
        :param fitness_function: can either be a fitness function or
        a class implementing a fitness function + methods to override
        the default ones: create_offspring, mutate_population, initialize_population
        :param n_genes: number of genes (variables) to have in each chromosome
        :param max_gen: maximum number of generations to perform the optimization
        :param pop_size: population size
        :param mutation_rate: rate at which random mutations occur
        :param selection_rate: percentage of the population to be selected for crossover
        :param selection_strategy: strategy to use for selection
        :param verbose: whether to print iterations status
        :param show_stats: whether to print stats at the end
        :param plot_results: whether to plot results of the run at the end
        :param variables_limits: limits for each variable [(x1_min, x1_max), (x2_min, x2_max), ...].
        If only one tuple is provided, then it is assumed the same for every variable
        :param problem_type: whether problem is of float or integer type
        """

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
        )

        if not variables_limits:  # TBC
            min_max = np.iinfo(np.int64)
            variables_limits = [(min_max.min, min_max.max) for _ in range(n_genes)]

        if get_input_dimensions(variables_limits) == 1:
            variables_limits = [variables_limits for _ in range(n_genes)]

        alphabet = get_alphabet_from_selfies(alphabet_list)
        alphabet.add("[nop]")
        self.branching = branching
        if self.branching:
            tuples = [
                (i + 1, j + 1) for i in range(self.n_genes) for j in range(self.n_genes)
            ]
            for i in tuples:
                if i[0] == i[1]:
                    pass
                else:
                    alphabet.add("[Branch{0}_{1}]".format(i[0], i[1]))
        self.alphabet = list(sorted(alphabet))
        self.variables_limits = variables_limits
        self.problem_type = problem_type
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


if __name__ == "__main__":
    print("rdkit version is : {0}".format(rdBase.rdkitVersion))
    starting_selfies = "[C][C=][C][C=][C][C=][Ring1][Branch1_2]"

    solver = SelfiesGenAlgSolver(
        n_genes=16,
        fitness_function=fitness_function_selfies(2),
        starting_selfies=starting_selfies,
        excluded_genes=[0, 1, 2, 3, 4, 5, 6, 7, 8],
    )
    solver.solve()
    print(solver.best_fitness_)
    mol_structure = sc2mol_structure(solver.best_individual_)
    mol_structure2depictions(mol_structure, root_name="substituted_benzene")
