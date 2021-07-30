import datetime
from abc import abstractmethod
from typing import Sequence
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from navicatGA.exceptions import NoFitnessFunction, InvalidInput
from navicatGA.exception_messages import exception_messages
from navicatGA.progress_bars import set_progress_bars
from navicatGA.cache import set_lru_cache
from navicatGA.helpers import get_elapsed_time
from navicatGA.logger import configure_logger, close_logger


allowed_selection_strategies = {
    "roulette_wheel",
    "two_by_two",
    "random",
    "tournament",
    "boltzmann",
}


class GenAlgSolver:
    def __init__(
        self,
        n_genes: int,
        fitness_function=None,
        assembler=None,
        max_gen: int = 1000,
        max_conv: int = 100,
        pop_size: int = 100,
        mutation_rate: float = 0.15,
        selection_rate: float = 0.5,
        selection_strategy: str = "roulette_wheel",
        excluded_genes: Sequence = None,
        n_crossover_points: int = 1,
        random_state: int = None,
        lru_cache: bool = False,
        scalarizer=None,
        prune_duplicates=False,
        verbose: bool = True,
        show_stats: bool = False,
        plot_results: bool = False,
        to_stdout: bool = True,
        to_file: bool = True,
        logger_file: str = "output.log",
        logger_level: str = "INFO",
        progress_bars: bool = False,
        problem_type: str = "base",
    ):
        """Base solver class for the GA.
        Has the core methods and attributes that any GA run will require.
        However, it lacks a way to create new offspring
        evaluate fitness, and generate mutations.
        Those must be defined in a child class.

        Parameters:
        :param n_genes: number of genes (variables) to have in each chromosome
        :type n_genes: int
        :param fitness_function: a fitness function that takes assembler(chromosome) and returns one (or more) fitness scores
        :type fitness_function: object
        :param assembler: a function that takes a chromosome and returns an (ideally hashable) object to be evaluated
        :type assembler: object
        :param max_gen: maximum number of generations to perform the optimization
        :type max_gen: int
        :param max_conv: maximum number of generations with same max fitness until convergence is assumed
        :type max_conv : int
        :param pop_size: number of chromosomes in population
        :type pop_size: int
        :param mutation_rate: rate at which random mutations occur
        :type mutation_rate: float
        :param selection_rate: top percentage of the population to be selected for crossover
        :type selection_rate: float
        :param selection_strategy: strategy to use for selection, several available
        :type selection_strategy: string
        :param excluded_genes: indices of chromosomes that should not be changed during run
        :type excluded_genes: optional, array-like
        :param n_crossover_points: number of slices to make for the crossover
        :type n_crossover_points: int
        :param random_state: fixed the random seed for the run
        :type random_state: int, optional
        :param lru_cache: whether to use lru_cacheing, which is monkeypatched into the class. Requires that the fitness function is hashable.
        :type lru_cache: bool
        :param scalarizer: chimera scalarizer object initialized to work on the results of fitness function
        :type scalarizer: optional, object with a scalarize method that takes in a population fitness and rescales it
        :param prune_duplicates: whether to prune duplicates in each generation
        :type prune_duplicates: bool
        :param verbose: whether to print iterations status
        :type verbose: int
        :param show_stats: whether to print stats at the end
        :type show_stats: bool
        :param plot_results: whether to plot results of the run at the end
        :type plot_results: bool
        :param to_stdout: whether to write output to stdout
        :type to_stdout: bool
        :param to_file: whether to write output to file
        :type to_file: bool
        :param logger_file: name of the file where output will be written if to_file is True
        :type logger_file: string
        :param progess_bars: whether to monkeypatch progress bars for monitoring run
        :type progress_bars: bool
        :param problem_type: passing a simple flag from child class for some in built hashable fitness functions.
        :type problem_type: string
        """

        if isinstance(random_state, int):
            np.random.seed(random_state)

        self.logger = configure_logger(
            logger_file=logger_file,
            logger_level=logger_level,
            to_stdout=to_stdout,
            to_file=to_file,
        )
        self.generations_ = 0
        self.best_individual_ = None
        self.best_fitness_ = 0
        self.best_pfitness_ = 0
        self.population_ = None
        self.fitness_ = None
        self.printable_fitness = None
        self.mean_fitness_ = None
        self.max_fitness_ = None
        self.n_genes = n_genes
        self.allowed_mutation_genes = np.arange(self.n_genes)
        self.assembler = assembler
        self.check_input_base(
            fitness_function, selection_strategy, pop_size, excluded_genes
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
        self.runtime_ = 0.0
        self.problem_type = problem_type
        self.prune_duplicates = prune_duplicates
        self.temperature = 100

        if progress_bars:
            self.logger.info("Setting up progress bars through monkeypatching.")
            set_progress_bars(self)
        if lru_cache:
            self.logger.info("Setting up lru cache through monkeypatching.")
            set_lru_cache(self)

    def check_input_base(
        self, fitness_function, selection_strategy, pop_size: int, excluded_genes
    ):
        """
        Function to check that the main arguments have been passed to the GenAlgSolver instance.

        Parameters:
        :param fitness_function: a fitness function that takes a chromosome and returns a fitness
        :param selection_strategy: a selection strategy string that can be recognized by this class
        :param pop_size: the number of chromosomes
        :param excluded_genes: a sequence of genes that should not change or mutate
        """

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

    def solve(self, niter=None):
        """
        Performs the genetic algorithm optimization according to the parameters
        loaded in __init__. Will run for max_gen or until it
        converges for max_conv iterations, or for min(niter,max_gen) iterations if niter
        is an integer. Will start using previous state if available.

        Parameters:
        :param niter: the number of generations to run
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
            self.logger.info("Initializing population.")
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
                population[-1 - ix[i], :] = self.create_offspring(
                    population[ma[i], :], population[pa[i], :], xp[i], "first"
                )
                population[-1 - ix[i] - 1, :] = self.create_offspring(
                    population[pa[i], :], population[ma[i], :], xp[i], "second"
                )

            population = self.mutate_population(population, self.n_mutations)
            if self.prune_duplicates:
                pruned_pop = np.zeros(shape=(1, self.n_genes), dtype=object)
                pruned_pop[0, :] = population[0, :]
                self.logger.debug(
                    f"Pruned pop set as {pruned_pop} and population set as {population}"
                )
                for i in range(1, self.pop_size):
                    try:
                        if not list(population[i]) == list(pruned_pop[-1]):
                            pruned_pop = np.vstack((pruned_pop, population[i]))
                    except Exception as m:
                        self.logger.debug(
                            f"Population comparison for pruning failed: {m}"
                        )
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
                printable_fitness[i] = rest_printable_fitness[i]
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
                self.logger.info("Best fitness result: {0}".format(self.best_pfitness_))
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
        self.printable_fitness = printable_fitness
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
        Calculates the fitness of the population using the defined fitness function.

        Parameters:
        :param population: population (array of chromosomes)

        Returns:
        :return fitness: scalarized fitness of the current population, will be used
        :return pfitness: not-scalarized fitness of the current population, for printing
        """
        if self.scalarizer is None:
            nvals = 1
            fitness = np.zeros(shape=(population.shape[0], nvals), dtype=float)
            for i in range(population.shape[0]):
                fitness[i, :] = self.fitness_function(self.assembler(population[i]))
            fitness = np.squeeze(fitness)
            pfitness = fitness
        else:
            nvals = len(self.scalarizer.goals)
            fitness = np.zeros(shape=(population.shape[0], nvals), dtype=float)
            for i in range(population.shape[0]):
                fitness[i, :] = self.fitness_function(self.assembler(population[i]))
            pfitness = fitness
            fitness = self.scalarizer.scalarize(fitness)
        return fitness, pfitness

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

        Parameters:
        :param fitness: the fitness values of the whole population at a given iteration

        Returns:
        :return  (ma, pa):  a tuple containing the selected 2 parents for each mating
        """

        ma, pa = None, None

        if (self.selection_strategy == "roulette_wheel") or (
            self.selection_strategy == "random"
        ):

            self.logger.trace(
                f"Selection probabilities for kept population are {self.prob_intervals}."
            )

            ma = np.apply_along_axis(
                self.interval_selection, 1, np.random.rand(self.n_matings, 1)
            )
            pa = np.apply_along_axis(
                self.interval_selection, 1, np.random.rand(self.n_matings, 1)
            )

        elif self.selection_strategy == "boltzmann":

            self.prob_intervals = self.get_boltzmann_probabilities(fitness)
            self.logger.trace(
                f"Selection probabilities for kept population are {self.prob_intervals}."
            )

            ma = np.apply_along_axis(
                self.interval_selection, 1, np.random.rand(self.n_matings, 1)
            )
            pa = np.apply_along_axis(
                self.interval_selection, 1, np.random.rand(self.n_matings, 1)
            )

        elif self.selection_strategy == "two_by_two":

            range_max = self.n_matings * 2

            ma = np.arange(range_max)[::2]
            pa = np.arange(range_max)[1::2]

            if ma.shape[0] > pa.shape[0]:
                ma = ma[:-1]

        elif self.selection_strategy == "tournament":

            range_max = self.n_matings * 2

            ma = self.tournament_selection(fitness, range_max)
            pa = self.tournament_selection(fitness, range_max)

        return ma, pa

    def interval_selection(self, value):
        """
        Select based on self.prob_intervals, which are given by the selection strategy.

        Parameters:
        :param value: random value defining which individual is selected from the probability intervals

        Returns:
        :return: the selected individual from the population
        """
        return np.argmin(value > self.prob_intervals) - 1

    def tournament_selection(self, fitness, range_max):
        """
        Performs tournament selection.

        Parameters:
        :param fitness: the fitness values of the population at a given iteration
        :param range_max: range of individuals that can be selected for the tournament

        Returns:
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
        from a pool of candidate individuals.
        """

        individuals_fitness = fitness[selected_individuals]

        return selected_individuals[np.argmax(individuals_fitness)]

    def get_selection_probabilities(self):
        """
        Calculates selection probabilities either randomly or scaled by position.
        """

        if self.selection_strategy == "roulette_wheel":
            mating_prob = (
                np.arange(1, self.pop_keep + 1) / np.arange(1, self.pop_keep + 1).sum()
            )[::-1]
            return np.array([0, *np.cumsum(mating_prob[: self.pop_keep + 1])])

        elif self.selection_strategy == "random":
            return np.linspace(0, 1, self.pop_keep + 1)

    def get_boltzmann_probabilities(self, fitness):
        """
        Calculates selection probabilities according to a fitness Boltzmann distribution with an increasing temperature.
        """

        nfit = fitness[0 : self.pop_keep + 1]
        self.logger.trace(f"Boltzmann initial preserved fitnesses: {nfit}")
        sfit = 1 / ((nfit - nfit.min()) / (nfit.max() - nfit.min()) + 1e-6)
        self.logger.trace(f"Boltzmann initial scaled fitnesses: {sfit}")
        mating_prob = np.exp(-sfit * (1 / (self.temperature)))
        self.logger.trace(f"Pre-normalized probabilities: {mating_prob}")
        C = np.sum(mating_prob)
        mating_prob = mating_prob * (1 / C)
        self.logger.trace(f"Normalized probabilities: {mating_prob}")
        self.temperature += 0.1 * self.temperature
        self.logger.debug(f"Temperature increased to {self.temperature}.")
        return np.array([0, *np.cumsum(mating_prob[: self.pop_keep])])

    def get_number_mutations(self):
        """Returns the number of mutations that need to be performed."""
        return int(np.ceil((self.pop_size - 1) * self.n_genes * self.mutation_rate))

    @staticmethod
    def sort_by_fitness(fitness, population, printable_fitness):
        """
        Sorts fitness, population and printable fitness according to fitness.
        """

        sorted_fitness = np.argsort(fitness)[::-1]
        population = population[sorted_fitness, :]
        fitness = fitness[sorted_fitness]
        pfitness = np.empty_like(printable_fitness)
        for i in sorted_fitness:
            pfitness[i] = printable_fitness[i]
        return fitness, population, pfitness

    def get_crossover_points(self):
        """Retrieves random crossover points."""
        crossover_points = np.random.choice(
            np.arange(len(self.allowed_mutation_genes)),
            self.n_crossover_points,
            replace=False,
        )
        return np.asarray(crossover_points).sort()

    @staticmethod
    def plot_fitness_results(mean_fitness, max_fitness, iterations: int):
        """
        Plots the evolution of the mean and max fitness of the population using matplotlib.

        Parameters:
        :param mean_fitness: mean fitness array for each generation
        :param max_fitness: max fitness array for each generation
        :param iterations: total number of generations
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
        Prints the statistics of the optimization run.
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
        self.logger.info(f"Best fitness: {self.best_pfitness_}")
        self.logger.info(f"Best individual: {self.best_individual_}")

    @abstractmethod
    def initialize_population(self):
        """
        Initializes the population of the problem. To be implemented in each child class.
        """
        pass

    @staticmethod
    def create_offspring(first_parent, sec_parent, crossover_pt, offspring_number):
        """
        Creates an offspring from 2 parent chromosomes. It uses the crossover point(s)
        to determine how to perform the crossover. To be implemented on each child class.
        Must return the resulting offspring chromosome.
        """
        pass

    def mutate_population(self, population, n_mutations):
        """
        Mutates the population according to a given user defined rule.
        To be defined further in each child class. Each direct child class can call
        this super method to retrieve the mutation rows and mutations columns in population.
        """

        mutation_rows = np.random.choice(
            np.arange(1, self.pop_size), n_mutations, replace=True
        )

        mutation_cols = np.random.choice(
            self.allowed_mutation_genes, n_mutations, replace=True
        )
        return mutation_rows, mutation_cols

    def close_solve_logger(self):
        """
        Closes the logger of this solver. This avoid multiple loggers stacking when another solver is created.
        """
        close_logger(self.logger)
