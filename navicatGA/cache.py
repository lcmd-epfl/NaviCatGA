import numpy as np
import types
import logging
from functools import lru_cache

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    try:
        from functools import lru_cache

        print("Monkeypatching the calculate_fitness method to add cacheing.")
    except ImportError as m:
        print(m)


def calculate_fitness_cache(self, population):
    """
    Calculates the fitness of the population using a hashable fitness function.

    Parameters:
    :param population: population state at a given iteration
    :return: the fitness of the current population
    """
    if self.scalarizer is None:
        nvals = 1
    else:
        nvals = len(self.scalarizer.goals)
    fitness = np.zeros(shape=(population.shape[0], nvals), dtype=float)
    for i in range(population.shape[0]):
        chromosome = population[i][0 : self.n_genes]
        hashable = self.assembler(chromosome)
        fitness[i, :] = calculate_one_fitness_cache(hashable, self.fitness_function)
    logger.trace(calculate_one_fitness_cache.cache_info())
    if self.scalarizer is None:
        return np.squeeze(fitness), np.squeeze(fitness)
    else:
        return np.ones((population.shape[0])) - self.scalarizer.scalarize(fitness), (
            fitness
        )


@lru_cache(maxsize=64)
def calculate_one_fitness_cache(hashable, fitness_function):
    return fitness_function(hashable)


def set_lru_cache(self):
    """
    Monkeypatches the calculate_fitness method of the base solver class in order to use a lru cache.
    If a specific wrapper exists for a given solver, it will try to use the unique expression of genes
    given by that wrapper to generate a hashable fitness function. If not, it will require
    a hashable fitness function given by the user AND expect the given fitness_function to generate
    a unique hash from a gene.
    """
    self.calculate_fitness = types.MethodType(calculate_fitness_cache, self)
