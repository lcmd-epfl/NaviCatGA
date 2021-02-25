import numpy as np
import types
import logging
from functools import lru_cache
from simpleGA.chemistry_selfies import get_selfie_chars
from simpleGA.wrappers_selfies import sc2selfies

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    try:
        from functools import lru_cache

        print("Monkeypatching the calculate_fitness method to add cacheing.")
    except ImportError as m:
        print(m)


def calculate_fitness_cache_selfies(self, population):
    """
    Calculates the fitness of the population
    :param population: population state at a given iteration
    :return: the fitness of the current population
    """
    fitness = np.zeros(shape=population.shape[0], dtype=float)
    logger.debug("Evaluating fitness individually with cache.")

    for i in range(population.shape[0]):
        chromosome = population[i][0 : self.n_genes]
        selfies = sc2selfies(chromosome)
        fitness[i] = calculate_one_fitness_cache_selfies(
            selfies, self.n_genes, self.fitness_function
        )

    logger.trace(calculate_one_fitness_cache_selfies.cache_info())
    return fitness


def calculate_fitness_cache_xyz(self, population):
    """
    Calculates the fitness of the population
    :param population: population state at a given iteration
    :return: the fitness of the current population
    """
    fitness = np.zeros(shape=population.shape[0], dtype=float)
    logger.debug("Evaluating fitness individually with cache.")
    for i in range(population.shape[0]):
        chromosome = population[i][0 : self.n_genes]
        geom = gl2geom(chromosome)[1]
        fitness[i] = calculate_one_fitness_cache_xyz(
            geom,
            self.hashable_fitness_function,
        )
    logger.trace(calculate_one_fitness_cache_xyz.cache_info())
    return fitness


@lru_cache(maxsize=128)
def calculate_one_fitness_cache_selfies(selfies, n_genes, fitness_function):
    return fitness_function(get_selfie_chars(selfies, n_genes))


@lru_cache(maxsize=128)
def calculate_one_fitness_cache_xyz(geom, hashable_fitness_function):
    return hashable_fitness_function(geom)


def set_lru_cache(self):
    if self.problem_type == "selfies":
        self.calculate_fitness = types.MethodType(calculate_fitness_cache_selfies, self)
    if self.problem_type == "xyz":
        self.calculate_fitness = types.MethodType(calculate_fitness_cache_xyz, self)
