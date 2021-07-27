import datetime
import logging
import types
import numpy as np
import alive_progress
from alive_progress import alive_bar

from simpleGA.helpers import get_elapsed_time

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    try:
        import alive_progress
        from alive_progress import alive_bar

        print("Monkeypatching the solve method to add progress bars.")
    except ImportError as m:
        print(m)


def solve_progress(self, niter=None):
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
        population = self.initialize_population()
    else:
        self.logger.info("Continuing run with previous population in memory.")
        population = self.initialize_population()

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

    with alive_bar(niter) as bar:
        for counter in range(niter):
            gen_n = counter + 1
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


def set_progress_bars(self):
    self.solve = types.MethodType(solve_progress, self)
