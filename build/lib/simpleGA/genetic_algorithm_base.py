import datetime
from abc import abstractmethod
from typing import Sequence

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from simpleGA.exceptions import NoFitnessFunction, InvalidInput
from simpleGA.exception_messages import exception_messages
from simpleGA.progress_bars import set_progress_bars
from simpleGA.cache import set_lru_cache
from simpleGA.helpers import get_elapsed_time
from simpleGA.logger import configure_logger, close_logger


allowed_selection_strategies = {"roulette_wheel", "two_by_two", "random", "tournament"}


class GenAlgSolver:
    def __init__(
        self,
        n_genes: int,
        fitness_function=None,
        max_gen: int = 1000,
        max_conv: int = 100,
        pop_size: int = 100,
        mutation_rate: float = 0.15,
        selection_rate: float = 0.5,
        selection_strategy: str = "roulette_wheel",
        verbose: bool = True,
        show_stats: bool = False,
        plot_results: bool = False,
        excluded_genes: Sequence = None,
        n_crossover_points: int = 1,
        random_state: int = None,
        lru_cache: bool = False,
        hashable_fitness_function=None,
        scalarizer=None,
        prune_duplicates=False,
        to_stdout: bool = True,
        to_file: bool = True,
        logger_file: str = "output.log",
        logger_level: str = "INFO",
        progress_bars: bool = False,
        problem_type: str = "base",
    ):
        """
        :param n_genes: number of genes (variables) to have in each chromosome
        :param fitness_function: a fitness function that takes a chromosome and returns one (or more) fitness scores
        :param max_gen: maximum number of generations to perform the optimization
        :param max_conv: maximum number of generations with same max fitness until convergence is assumed
        :param pop_size: number of chromosomes in population
        :param mutation_rate: rate at which random mutations occur
        :param selection_rate: top percentage of the population to be selected for crossover
        :param selection_strategy: strategy to use for selection, several available
        :param verbose: whether to print iterations status
        :param show_stats: whether to print stats at the end
        :param plot_results: whether to plot results of the run at the end
        :param excluded_genes: indices of chromosomes that should not be changed during run
        :param n_crossover_points: number of slices to make for the crossover
        :param random_state: optional. whether the random seed should be set
        :param lru_cache: whether to use lru_cacheing, which is monkeypatched into the class. Requires that the fitness function is hashable.
        :param hashable_fitness_function: specific fitness function that derives to an ultimately hashable argument.
        :param scalarizer: chimera scalarizer object initialized to work on the results of fitness function
        :param prune_duplicates: whether to prune duplicates in each generation
        :param to_stdout: whether to write output to stdout
        :param to_file: whether to write output to file
        :param logger_file: name of the file where output will be written if to_file is True
        :param progess_bars: whether to monkeypatch progress bars for monitoring run
        :param problem_type: passing a simple flag from child class for some in built hashable fitness functions.
        """

        if isinstance(random_state, int):
            np.random.seed(random_state)

        self.logger = configure_logger(
            logger_file=logger_file,
            logger_level=logger_level,
            to_stdout=to_stdout,
            to_file=to_file,
        )
        self.n_genes = n_genes
        self.allowed_mutation_genes = np.arange(self.n_genes)
        if fitness_function is None and hashable_fitness_function is not None:
            fitness_function = hashable_fitness_function
        self.check_input_base(
            fitness_function, selection_strategy, pop_size, excluded_genes
        )
        if lru_cache:
            self.check_input_base_cache(
                hashable_fitness_function, selection_strategy, pop_size, excluded_genes
            )
        self.scalarizer = scalarizer
        self.selection_strategy = selection_strategy
        self.max_gen = max_gen
        self.max_conv = max_conv
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.selection_rate = selection_rate
        self.n_crossover_points = n_crossover_points
        self.verbose = verbose
        self.show_stats = show_stats
        self.plot_results = plot_results
        self.pop_keep = int(np.floor(selection_rate * pop_size))
        if self.pop_keep < 2:
            self.pop_keep = 2
        self.prob_intervals = self.get_selection_probabilities()
        self.n_matings = int(np.floor((self.pop_size - self.pop_keep) / 2))
        self.n_mutations = self.get_number_mutations()
        self.generations_ = 0
        self.best_individual_ = None
        self.best_fitness_ = 0
        self.best_pfitness_ = 0
        self.population_ = None
        self.fitness_ = None
        self.printable_fitness = None
        self.mean_fitness_ = None
        self.max_fitness_ = None
        self.runtime_ = 0.0
        self.problem_type = problem_type
        self.prune_duplicates = prune_duplicates
        self.hashable_fitness_function = hashable_fitness_function

        if progress_bars:
            self.logger.info("Setting up progress bars through monkeypatching.")
            set_progress_bars(self)
        if lru_cache:
            self.logger.info("Setting up lru cache through monkeypatching.")
            set_lru_cache(self)

    def check_input_base(
        self, fitness_function, selection_strategy, pop_size, excluded_genes
    ):

        if not fitness_function:
            try:
                getattr(self, "fitness_function")
            except AttributeError:
                raise NoFitnessFunction(
                    "A fitness function must be defined or provided as an argument"
                )
        else:
            self.fitness_function = fitness_function

        if selection_strategy not in allowed_selection_strategies:
            raise InvalidInput(
                exception_messages["InvalidSelectionStrategy"](
                    selection_strategy, allowed_selection_strategies
                )
            )

        if pop_size < 2:
            raise (InvalidInput(exception_messages["InvalidPopulationSize"]))

        if isinstance(excluded_genes, (list, tuple, np.ndarray)):
            self.allowed_mutation_genes = [
                item
                for item in self.allowed_mutation_genes
                if item not in excluded_genes
            ]

        elif excluded_genes is not None:
            raise InvalidInput(
                exception_messages["InvalidExcludedGenes"](excluded_genes)
            )

    def check_input_base_cache(
        self, hashable_fitness_function, selection_strategy, pop_size, excluded_genes
    ):

        if not hashable_fitness_function:
            try:
                getattr(self, "hashable_fitness_function")
            except AttributeError:
                raise NoFitnessFunction(
                    "A hashable fitness function must be defined or provided as an argument"
                )
        else:
            self.hashable_fitness_function = hashable_fitness_function

        if selection_strategy not in allowed_selection_strategies:
            raise InvalidInput(
                exception_messages["InvalidSelectionStrategy"](
                    selection_strategy, allowed_selection_strategies
                )
            )

        if pop_size < 2:
            raise (InvalidInput(exception_messages["InvalidPopulationSize"]))

        if isinstance(excluded_genes, (list, tuple, np.ndarray)):
            self.allowed_mutation_genes = [
                item
                for item in self.allowed_mutation_genes
                if item not in excluded_genes
            ]

        elif excluded_genes is not None:
            raise InvalidInput(
                exception_messages["InvalidExcludedGenes"](excluded_genes)
            )

    def solve(self, niter=None):
        """
        Performs the genetic algorithm optimization according to the parameters
        provided at initialization. Will run for max_gen generations, or until it
        converges for max_conv iterations, or for niter iterations if used
        interactively. Will start using previous state if available.
        :return: None
        """

        start_time = datetime.datetime.now()
        if self.mean_fitness_ is None:
            mean_fitness = np.ndarray(shape=(1, 0))
        else:
            self.logger.info("Continuing run with previous mean fitness in memory.")
            mean_fitness = self.mean_fitness_
        if self.max_fitness_ is None:
            max_fitness = np.ndarray(shape=(1, 0))
        else:
            self.logger.info("Continuing run with previous max fitness in memory.")
            max_fitness = self.max_fitness_
        if self.population_ is None:
            population = self.initialize_population()
        else:
            self.logger.info("Continuing run with previous population in memory.")
            population = self.population_

        fitness, printable_fitness = self.calculate_fitness(population)
        fitness, population, printable_fitness = self.sort_by_fitness(
            fitness, population, printable_fitness
        )

        gen_interval = max(round(self.max_gen / 10), 1)
        gen_n = 1
        conv = 0
        if isinstance(niter, int):
            niter = min(self.max_gen, niter)
        else:
            niter = self.max_gen
        for _ in range(niter):
            gen_n += 1
            self.generations_ += 1

            mean_fitness = np.append(mean_fitness, fitness.mean())
            max_fitness = np.append(max_fitness, fitness[0])
            ma, pa = self.select_parents(fitness)
            ix = np.arange(0, self.pop_size - self.pop_keep - 1, 2)
            xp = np.array(
                list(map(lambda _: self.get_crossover_points(), range(self.n_matings)))
            )

            for i in range(xp.shape[0]):

                # create first offspring
                population[-1 - ix[i], :] = self.create_offspring(
                    population[ma[i], :], population[pa[i], :], xp[i], "first"
                )

                # create second offspring
                population[-1 - ix[i] - 1, :] = self.create_offspring(
                    population[pa[i], :], population[ma[i], :], xp[i], "second"
                )

            population = self.mutate_population(population, self.n_mutations)
            if self.prune_duplicates:
                try:
                    population = np.unique(population, axis=0)
                    nrefill = self.pop_size - population.shape[0]
                    population = np.hstack(
                        (population, self.refill_population(nrefill))
                    )
                except TypeError:
                    pruned_pop = np.zeros(shape=(1, self.n_genes), dtype=object)
                    pruned_pop[0, :] = population[0, :]
                    for i in range(1, self.pop_size):
                        if not all(population[i, :] == pruned_pop[-1, :]):
                            pruned_pop = np.vstack((pruned_pop, population[i]))
                    nrefill = self.pop_size - pruned_pop.shape[0]
                    if nrefill > 0:
                        self.logger.debug(
                            f"Replacing a total of {nrefill} chromosomes due to duplications."
                        )
                        population = np.vstack(
                            (pruned_pop, self.refill_population(nrefill))
                        )
            rest_fitness, rest_printable_fitness = self.calculate_fitness(
                population[1:, :]
            )
            fitness = np.hstack((fitness[0], rest_fitness))
            for i in range(1, len(rest_fitness)):
                printable_fitness = rest_printable_fitness[i]
            fitness, population, printable_fitness = self.sort_by_fitness(
                fitness, population, printable_fitness
            )
            self.best_individual_ = population[0, :]
            if np.isclose(self.best_fitness_, fitness[0]):
                conv += 1
            self.best_fitness_ = fitness[0]
            self.best_pfitness_ = printable_fitness[0]

            if self.verbose:
                self.logger.info("Generation: {0}".format(self.generations_))
                self.logger.info("Best fitness: {0}".format(self.best_pfitness_))
                self.logger.trace("Best individual: {0}".format(population[0, :]))
                self.logger.trace(
                    "Population at generation: {0}: {1}".format(
                        self.generations_, population
                    )
                )

            if gen_n >= niter or conv > self.max_conv:
                break

        self.population_ = population
        self.fitness_ = fitness
        self.mean_fitness_ = mean_fitness
        self.max_fitness_ = max_fitness

        if self.plot_results:
            self.plot_fitness_results(
                self.mean_fitness_, self.max_fitness_, self.generations_
            )

        end_time = datetime.datetime.now()
        self.runtime_, time_str = get_elapsed_time(start_time, end_time)

        if self.show_stats:
            self.print_stats(time_str)

    def calculate_fitness(self, population):
        """
        Calculates the fitness of the population
        :param population: population state at a given iteration
        :return: the fitness of the current population
        """
        if self.scalarizer is None:
            nvals = 1
            fitness = np.zeros(shape=(population.shape[0], nvals), dtype=float)
            for i in range(population.shape[0]):
                fitness[i, :] = self.fitness_function(population[i])
            self.printable_fitness = fitness
            return np.squeeze(fitness), np.squeeze(fitness)
        else:
            nvals = len(self.scalarizer.goals)
            fitness = np.zeros(shape=(population.shape[0], nvals), dtype=float)
            for i in range(population.shape[0]):
                fitness[i, :] = self.fitness_function(population[i])
            self.printable_fitness = fitness
            return self.scalarizer.scalarize(fitness), (fitness)

    def select_parents(self, fitness):
        """
        Selects the parents according to a given selection strategy.
        Options are:
        roulette_wheel: Selects individuals from mating pool giving
        higher probabilities to fitter individuals.
        two_by_two: Pairs fittest individuals two by two
        random: Selects individuals from mating pool randomly.
        tournament: Selects individuals by choosing groups of 3 candidate
        individuals and then selecting the fittest one from the 3.
        :param fitness: the fitness values of the population at a given iteration
        :return: a tuple containing the selected 2 parents for each mating
        """

        ma, pa = None, None

        if self.selection_strategy == "roulette_wheel":

            ma = np.apply_along_axis(
                self.roulette_wheel_selection, 1, np.random.rand(self.n_matings, 1)
            )
            pa = np.apply_along_axis(
                self.roulette_wheel_selection, 1, np.random.rand(self.n_matings, 1)
            )

        elif self.selection_strategy == "two_by_two":

            range_max = int(self.n_matings * 2)

            ma = np.arange(range_max)[::2]
            pa = np.arange(range_max)[1::2]

            if ma.shape[0] > pa.shape[0]:
                ma = ma[:-1]

        elif self.selection_strategy == "random":

            ma = np.apply_along_axis(
                self.random_selection, 1, np.random.rand(self.n_matings, 1)
            )
            pa = np.apply_along_axis(
                self.random_selection, 1, np.random.rand(self.n_matings, 1)
            )

        elif self.selection_strategy == "tournament":

            range_max = int(self.n_matings * 2)

            ma = self.tournament_selection(fitness, range_max)
            pa = self.tournament_selection(fitness, range_max)

        return ma, pa

    def roulette_wheel_selection(self, value):
        """
        Performs roulette wheel selection
        :param value: random value defining which individual is selected
        :return: the selected individual
        """
        return np.argmin(value > self.prob_intervals) - 1

    def random_selection(self, value):
        """
        Performs random selection
        :param value: random value defining which individual is selected
        :return: the selected individual
        """
        return np.argmin(value > self.prob_intervals) - 1

    def tournament_selection(self, fitness, range_max):
        """
        Performs tournament selection.
        :param fitness: the fitness values of the population at a given iteration
        :param range_max: range of individuals that can be selected for the tournament
        :return: the selected individuals
        """

        selected_individuals = np.random.choice(range_max, size=(self.n_matings, 3))

        return np.array(
            list(
                map(
                    lambda x: self.tournament_selection_helper(x, fitness),
                    selected_individuals,
                )
            )
        )

    @staticmethod
    def tournament_selection_helper(selected_individuals, fitness):
        """
        Helper for tournament selection method. Selects the fittest individual
        from a pool of candidate individuals
        :param selected_individuals: group of candidate individuals for
        tournament selection
        :param fitness: the fitness values of the population at a given iteration
        :return: the selected individual
        """

        individuals_fitness = fitness[selected_individuals]

        return selected_individuals[np.argmax(individuals_fitness)]

    def get_selection_probabilities(self):

        if self.selection_strategy == "roulette_wheel":
            mating_prob = (
                np.arange(1, self.pop_keep + 1) / np.arange(1, self.pop_keep + 1).sum()
            )[::-1]
            return np.array([0, *np.cumsum(mating_prob[: self.pop_keep + 1])])

        elif self.selection_strategy == "random":
            return np.linspace(0, 1, self.pop_keep + 1)

    def get_number_mutations(self):
        return int(np.ceil((self.pop_size - 1) * self.n_genes * self.mutation_rate))

    @staticmethod
    def sort_by_fitness(fitness, population, printable_fitness):
        """
        Sorts the population by its fitness.
        :param fitness: fitness of the population
        :param population: population state at a given iteration
        :return: the sorted fitness array and sorted population array
        """

        sorted_fitness = np.argsort(fitness)[::-1]
        population = population[sorted_fitness, :]
        fitness = fitness[sorted_fitness]
        try:
            printable_fitness = printable_fitness[sorted_fitness, :]
        except IndexError:
            printable_fitness = fitness
        return fitness, population, printable_fitness

    def get_crossover_points(self):
        """
        Retrieves random crossover points
        :return: a numpy array with the crossover points
        """
        crossover_points = np.random.choice(
            np.arange(len(self.allowed_mutation_genes)),
            self.n_crossover_points,
            replace=False,
        )
        return np.asarray(crossover_points).sort()

    @staticmethod
    def plot_fitness_results(mean_fitness, max_fitness, iterations):
        """
        Plots the evolution of the mean and max fitness of the population
        :param mean_fitness: mean fitness array for each generation
        :param max_fitness: max fitness array for each generation
        :param iterations: total number of generations
        :return: None
        """

        plt.figure(figsize=(7, 7))
        x = np.arange(1, iterations + 1)
        plt.plot(x, max_fitness, label="max fitness")
        plt.plot(x, mean_fitness, label="mean fitness")
        plt.legend()
        plt.tight_layout()
        plt.savefig("evolution.png")
        plt.close()

    def print_stats(self, time_str):
        """
        Prints the statistics of the optimization run
        :param time_str: time string given by the method get_elapsed_time
        :return: None
        """

        self.logger.info("\n#############################")
        self.logger.info("#           STATS           #")
        self.logger.info("#############################\n\n")
        self.logger.info(f"Total running time: {time_str}\n")
        self.logger.info(f"Population size: {self.pop_size}")
        self.logger.info(f"Number variables: {self.n_genes}")
        self.logger.info(f"Selection rate: {self.selection_rate}")
        self.logger.info(f"Mutation rate: {self.mutation_rate}")
        self.logger.info(f"Number Generations: {self.generations_}")
        self.logger.info(f"Best fitness: {self.best_fitness_}")
        self.logger.info(f"Best individual: {self.best_individual_}")

    @abstractmethod
    def initialize_population(self):
        """
        Initializes the population of the problem. To be implemented in each child class.
        :return: a numpy array with a randomized initialized population
        """
        pass

    @staticmethod
    def create_offspring(first_parent, sec_parent, crossover_pt, offspring_number):
        """
        Creates an offspring from 2 parents. It uses the crossover point(s)
        to determine how to perform the crossover. To be implemented on each child class.
        :param first_parent: first parent's chromosome
        :param sec_parent: second parent's chromosome
        :param crossover_pt: point(s) at which to perform the crossover
        :param offspring_number: whether it's the first or second offspring from a pair of parents.
        Important if there's different logic to be applied to each case.
        :return: the resulting offspring.
        """
        pass

    def mutate_population(self, population, n_mutations):
        """
        Mutates the population according to a given user defined rule.
        To be defined further in each child class. Each direct child class can call
        this super method to retrieve the mutation rows and mutations columns
        :param population: the population at a given iteration
        :param n_mutations: number of mutations to be performed. This number is
        calculated according to mutation_rate, but can be adjusted as needed inside this function
        :return: an array with the mutation_rows and mutation_cols
        """

        mutation_rows = np.random.choice(
            np.arange(1, self.pop_size), n_mutations, replace=True
        )

        mutation_cols = np.random.choice(
            self.allowed_mutation_genes, n_mutations, replace=True
        )
        return mutation_rows, mutation_cols

    def close_solve_logger(self):
        close_logger(self.logger)
