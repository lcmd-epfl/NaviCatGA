import numpy as np
import types
import logging
from functools import lru_cache
from simpleGA.chemistry import get_selfie_chars
from simpleGA.wrappers import sc2selfies

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    try:
        from functools import lru_cache

        print("Monkeypatching the calculate_fitness method to add cacheing.")
    except ImportError as m:
        print(m)


def calculate_fitness_cache(self, population):
    """
    Calculates the fitness of the population
    :param population: population state at a given iteration
    :return: the fitness of the current population
    """
    fitness = np.zeros(shape=population.shape[0], dtype=float)
    logger.trace("Evaluating fitness individually with cache.")

    for i in range(population.shape[0]):
        chromosome = population[i][0 : self.n_genes]
        selfies = sc2selfies(chromosome)
        fitness[i] = calculate_one_fitness_cache(
            selfies, self.n_genes, self.fitness_function
        )

    logger.trace(calculate_one_fitness_cache.cache_info())
    return fitness


@lru_cache(maxsize=128)
def calculate_one_fitness_cache(selfies, n_genes, fitness_function):
    return fitness_function(get_selfie_chars(selfies, n_genes))


def set_lru_cache(self):
    self.calculate_fitness = types.MethodType(calculate_fitness_cache, self)
