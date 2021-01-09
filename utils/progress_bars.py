import datetime
import logging
import types
import numpy as np
import alive_progress
from alive_progress import alive_bar

from utils.helpers import get_elapsed_time

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    try:
        import alive_progress
        from alive_progress import alive_bar

        logger.info("Monkeypatching the solve method to add progress bars.")
    except ImportError as m:
        logger.info(
            "The progress package is not installed. Not monkeypatching the solve method."
        )
        logger.debug(m)


def solve_progress(self):
    """
    Performs the genetic algorithm optimization according to the parameters
    provided at initialization.
    :return: None
    """

    start_time = datetime.datetime.now()

    mean_fitness = np.ndarray(shape=(1, 0))
    max_fitness = np.ndarray(shape=(1, 0))

    # initialize the population
    population = self.initialize_population()

    fitness = self.calculate_fitness(population)

    fitness, population = self.sort_by_fitness(fitness, population)

    gen_interval = max(round(self.max_gen / 10), 1)

    with alive_bar(self.max_gen) as bar:
        for gen_n in range(self.max_gen):

            if self.verbose and gen_n % gen_interval == 0:
                self.logger.info("Iteration: {0}".format(gen_n))
                self.logger.info(f"Best fitness: {fitness[0]}")
                self.logger.info(f"Best individual: {population[0,:]}")
                self.logger.debug(
                    "Population at generation: {0}: {1}".format(gen_n, population)
                )

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

            fitness = np.hstack((fitness[0], self.calculate_fitness(population[1:, :])))

            fitness, population = self.sort_by_fitness(fitness, population)

            bar()

        self.generations_ = gen_n
        self.best_individual_ = population[0, :]
        self.best_fitness_ = fitness[0]
        self.population_ = population
        self.fitness_ = fitness

        if self.plot_results:
            self.plot_fitness_results(mean_fitness, max_fitness, gen_n)

        if self.show_stats:
            end_time = datetime.datetime.now()
            time_str = get_elapsed_time(start_time, end_time)
            self.print_stats(time_str)

        self.close_solve_logger()


def set_progress_bars(self):
    self.solve = types.MethodType(solve_progress, self)
