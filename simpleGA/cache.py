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
    Calculates the fitness of the population
    :param population: population state at a given iteration
    :return: the fitness of the current population
    """
    fitness = np.zeros(shape=self.pop_size, dtype=float)

    @lru_cache(maxsize=256)
    def calculate_one_fitness_cache(chromosome, fitness_function):
        return fitness_function(chromosome)

    for i in range(self.pop_size):
        fitness[i] = calculate_one_fitness_cache(
            population[i][0 : self.n_genes], self.fitness_function
        )
    logger.info(calculate_one_fitness_cache.cache_info())
    return fitness


def set_lru_cache(self):
    self.calculate_fitness = types.MethodType(calculate_fitness_cache, self, population)
