from typing import Sequence

import numpy as np

from genetic_algorithm_base import GenAlgSolver
from utils.helpers import get_input_dimensions
from chemistry.evo import sanitize_multiple_smiles, encode_smiles, get_selfie_chars
from selfies import get_alphabet_from_selfies


class SelfiesGenAlgSolver(GenAlgSolver):
    def __init__(
        self,
        n_genes: int,
        starting_smiles: list = ["c1ccccc1"],
        alphabet_list: list = [
            "[C][=C][C][=C][C][=C][Ring1][Branch1_2]",
            "[F][C][F]",
            "[O][=O]",
            "[C][N][C][Branch3][epsilon][C][C][C][=C][C][C][Branch3][=O][=C][Ring][#N][O][C][O][Ring][#N]",
            "[C][C][O][C][C]",
            "[N][#N]",
            "[C][O][C]",
        ],
        fitness_function=None,
        max_gen: int = 1000,
        pop_size: int = 100,
        mutation_rate: float = 0.5,
        selection_rate: float = 0.15,
        selection_strategy: str = "roulette_wheel",
        verbose: bool = False,
        show_stats: bool = False,
        plot_results: bool = False,
        excluded_genes: Sequence = None,
        variables_limits=(-10, 10),
        problem_type=str,
        n_crossover_points: int = 1,
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

        self.variables_limits = variables_limits
        self.problem_type = problem_type
        self.starting_smiles = starting_smiles
        alphabet = get_alphabet_from_selfies(alphabet_list)
        alphabet.add('[nop]')
        self.alphabet = list(sorted(alphabet))

    def initialize_population(self):
        """
        Initializes the population of the problem according to the
        population size and number of genes and according to the problem
        type (SELFIES elements here).
        :return: a numpy array with a sanitized initialized population
        """

        population = np.zeros(shape=(self.pop_size, self.n_genes), dtype="object")
        sanitized_smiles = sanitize_multiple_smiles(self.starting_smiles)
        sanitized_selfies = encode_smiles(sanitized_smiles)
        chromosomes = [get_selfie_chars(i, self.n_genes) for i in sanitized_selfies]
        if self.pop_size > len(chromosomes):
            for j in range(self.pop_size):
                population[j, :] = chromosomes[j % len(chromosomes)]
        else:  # The user gave too many initial stuff, we wont use all
            for j in range(self.pop_size):
                population[j, :] = chromosomes[j]

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
        Creates an offspring from 2 parents. It performs the crossover
        according the following rule:
        p_new = first_parent[crossover_pt] + beta * (first_parent[crossover_pt] - sec_parent[crossover_pt])
        offspring = [first_parent[:crossover_pt], p_new, sec_parent[crossover_pt + 1:]
        where beta is a random number between 0 and 1, and can be either positive or negative
        depending on if it's the first or second offspring
        http://index-of.es/z0ro-Repository-3/Genetic-Algorithm/R.L.Haupt,%20S.E.Haupt%20-%20Practical%20Genetic%20Algorithms.pdf
        :param first_parent: first parent's chromosome
        :param sec_parent: second parent's chromosome
        :param crossover_pt: point(s) at which to perform the crossover
        :param offspring_number: whether it's the first or second offspring from a pair of parents.
        Important if there's different logic to be applied to each case.
        :return: the resulting offspring.
        """

        crossover_pt = crossover_pt[0]

        beta = np.random.rand(1)[0]
        if beta > 0.5:
            if offspring_number == "first":
                sec_parent = sec_parent[::-1]
                p_new = first_parent[crossover_pt]
            if offspring_number == "second":
                first_parent = first_parent[::-1]
                p_new = sec_parent[crossover_pt]
        else:
            if offspring_number == "first":
                p_new = sec_parent[crossover_pt]
            if offspring_number == "second":
                p_new = first_parent[crossover_pt]

        return np.hstack(
            (first_parent[:crossover_pt], p_new, sec_parent[crossover_pt + 1 :])
        )

    def mutate_population(self, population, n_mutations):
        """
        Mutates the population by randomizing specific positions of the
        population individuals.
        :param population: the population at a given iteration
        :param n_mutations: number of mutations to be performed.
        :return: the mutated population
        """

        mutation_rows, mutation_cols = super(
            SelfiesGenAlgSolver, self
        ).mutate_population(population, n_mutations)

        population[mutation_rows, mutation_cols] = np.random.choice(
            self.alphabet, size=1
        )[0]

        return population
