from typing import Sequence

import numpy as np

from simpleGA.base_solver import GenAlgSolver
from simpleGA.helpers import get_input_dimensions
from simpleGA.fitness_functions_float import fitness_function_float


class ContinuousGenAlgSolver(GenAlgSolver):
    def __init__(
        self,
        n_genes: int,
        fitness_function=None,
        max_gen: int = 1000,
        pop_size: int = 100,
        mutation_rate: float = 0.15,
        selection_rate: float = 0.5,
        selection_strategy: str = "roulette_wheel",
        verbose: bool = False,
        show_stats: bool = False,
        plot_results: bool = False,
        excluded_genes: Sequence = None,
        variables_limits=(-10, 10),
        problem_type=float,
        n_crossover_points: int = 1,
        random_state: int = None,
        logger_file: str = "output.log",
        logger_level: str = "INFO",
        to_stdout: bool = True,
        to_file: bool = True,
        progress_bars: bool = False,
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
            logger_file=logger_file,
            logger_level=logger_level,
            to_stdout=to_stdout,
            to_file=to_file,
            progress_bars=progress_bars,
        )

        if not variables_limits:
            min_max = np.iinfo(np.int64)
            variables_limits = [(min_max.min, min_max.max) for _ in range(n_genes)]

        if get_input_dimensions(variables_limits) == 1:
            variables_limits = [variables_limits for _ in range(n_genes)]

        self.variables_limits = variables_limits
        self.problem_type = problem_type

    def initialize_population(self):
        """
        Initializes the population of the problem according to the
        population size and number of genes and according to the problem
        type (either integers or floats).
        :return: a numpy array with a randomized initialized population
        """

        population = np.empty(shape=(self.pop_size, self.n_genes))

        for i, variable_limits in enumerate(self.variables_limits):
            if self.problem_type == float:
                population[:, i] = np.random.uniform(
                    variable_limits[0], variable_limits[1], size=self.pop_size
                )
            else:
                population[:, i] = np.random.randint(
                    variable_limits[0], variable_limits[1] + 1, size=self.pop_size
                )

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

        beta = (
            np.random.rand(1)[0]
            if offspring_number == "first"
            else -np.random.rand(1)[0]
        )

        if self.problem_type == float:
            p_new = first_parent[crossover_pt] - beta * (
                first_parent[crossover_pt] - sec_parent[crossover_pt]
            )
        else:
            p_new = first_parent[crossover_pt] - np.round(
                beta * (first_parent[crossover_pt] - sec_parent[crossover_pt])
            )

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
            ContinuousGenAlgSolver, self
        ).mutate_population(population, n_mutations)

        population[mutation_rows, mutation_cols] = self.initialize_population()[
            mutation_rows, mutation_cols
        ]

        return population


def test_bohachevsky():
    solver = ContinuousGenAlgSolver(
        n_genes=2,
        pop_size=100,
        max_gen=250,
        mutation_rate=0.10,
        selection_rate=0.15,
        variables_limits=(-100, 100),
        fitness_function=fitness_function_float(7),
        selection_strategy="random",
        to_file=False,
        progress_bars=True,
        verbose=True,
    )
    solver.solve()


if __name__ == "__main__":
    test_bohachevsky()
