from typing import Sequence
import logging
import random
import numpy as np

from simpleGA.genetic_algorithm_base import GenAlgSolver
from simpleGA.chemistry_smiles import (
    sanitize_multiple_smiles,
    get_smiles_chars,
    randomize_smiles,
)
from simpleGA.wrappers_smiles import check_smiles_chars, sc2depictions
from simpleGA.exceptions import InvalidInput
from simpleGA.exception_messages import exception_messages
from rdkit import rdBase


logger = logging.getLogger(__name__)


class SmilesGenAlgSolver(GenAlgSolver):
    def __init__(
        self,
        n_genes: int,
        starting_smiles: list = [""],
        starting_random: bool = False,
        substituent_list: list = ["C", "N", "P"],
        fitness_function=None,
        max_gen: int = 500,
        pop_size: int = 100,
        mutation_rate: float = 0.05,
        selection_rate: float = 0.25,
        selection_strategy: str = "tournament",
        hashable_fitness_function=None,
        verbose: bool = True,
        show_stats: bool = False,
        plot_results: bool = False,
        excluded_genes: Sequence = [0],
        variables_limits: dict = None,
        n_crossover_points: int = 1,
        max_counter: int = 10,
        random_state: int = None,
        logger_file: str = "output.log",
        logger_level: str = "INFO",
        to_stdout: bool = True,
        to_file: bool = True,
        progress_bars: bool = False,
        lru_cache: bool = False,
        problem_type="smiles",
    ):

        GenAlgSolver.__init__(
            self,
            fitness_function=fitness_function,
            hashable_fitness_function=hashable_fitness_function,
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
            lru_cache=lru_cache,
            problem_type=problem_type,
        )
        self.alphabet = list(sorted(substituent_list))

        if not isinstance(starting_smiles, list):
            raise (InvalidInput(exception_messages["StartingSmilesNotAList"]))
        if not self.alphabet:
            raise (InvalidInput(exception_messages["AlphabetIsEmpty"]))
        if self.n_crossover_points > self.n_genes:
            raise (InvalidInput(exception_messages["TooManyCrossoverPoints"]))
        if self.n_crossover_points < 1:
            raise (InvalidInput(exception_messages["TooFewCrossoverPoints"]))

        if len(starting_smiles) < self.pop_size:
            n_patch = self.pop_size - len(starting_smiles)
            for i in range(n_patch):
                starting_smiles.append(np.random.choice(starting_smiles, size=1)[0])
        elif len(starting_smiles) > self.pop_size:
            n_remove = len(starting_smiles) - self.pop_size
            for i in range(n_remove):
                starting_smiles.remove(np.random.choice(starting_smiles, size=1)[0])
        assert len(starting_smiles) == self.pop_size
        self.starting_smiles = starting_smiles
        self.max_counter = int(max_counter)
        self.starting_random = starting_random

    def initialize_population(self):
        """
        Initializes the population of the problem according to the
        population size and number of genes and according to the problem
        type (smiles string elements here).
        :return: a numpy array with a sanitized initialized population
        """

        population = np.zeros(shape=(self.pop_size, self.n_genes), dtype=object)

        for i in range(self.pop_size):
            chromosome = get_smiles_chars(self.starting_smiles[i], self.n_genes)
            if self.starting_random:
                for j in range(1, self.n_genes):
                    chromosome[j] = np.random.choice(self.alphabet, size=1)[0]
            assert check_smiles_chars(chromosome)
            population[i][:] = chromosome[0 : self.n_genes]

        self.logger.debug("Initial population: {0}".format(population))
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
        Creates an offspring from 2 parents.
        """
        beta = np.random.rand(1)[0]
        gamma = np.random.rand(1)[0]
        backup_sec_parent = sec_parent
        backup_first_parent = first_parent
        if self.allowed_mutation_genes is not None:
            mask_allowed = np.zeros(first_parent.size, dtype=bool)
            mask_forbidden = np.ones(first_parent.size, dtype=bool)
            mask_allowed[self.allowed_mutation_genes] = True
            mask_forbidden[self.allowed_mutation_genes] = False
            full_offspring = np.empty_like(first_parent, dtype=object)
            full_offspring[mask_forbidden] = first_parent[mask_forbidden]
            first_parent = first_parent[mask_allowed]
            sec_parent = sec_parent[mask_allowed]

        offspring = np.empty_like(first_parent, dtype=object)
        valid_smiles = False

        # if beta < 0.5:
        #    first_parent = first_parent[::-1]
        # if gamma > 0.5:
        #    sec_parent = sec_parent[::-1]

        if offspring_number == "first":
            while not valid_smiles:
                offspring = np.empty_like(first_parent, dtype=object)
                counter = 0
                c = int(0)
                offspring[c : crossover_pt[0]] = first_parent[c : crossover_pt[0]]
                offspring[crossover_pt[0] :] = sec_parent[crossover_pt[0] :]
                for ci, cj in zip(crossover_pt[::2], crossover_pt[1::2]):
                    offspring[c:ci] = first_parent[c:ci]
                    offspring[ci:] = sec_parent[ci:]
                    c = cj
                if self.allowed_mutation_genes is not None:
                    full_offspring[mask_allowed] = offspring[:]
                    offspring = full_offspring
                logger.trace(
                    "Offspring chromosome attempt {0}: {1}".format(counter, offspring)
                )
                valid_smiles = check_smiles_chars(offspring)
                crossover_pt = self.get_crossover_points()
                counter += 1
                if counter > self.max_counter:
                    logger.trace(
                        "Counter in create offspring exceeded {0}, using default.".format(
                            self.max_counter
                        )
                    )
                    valid_smiles = True
                    offspring = backup_first_parent
            logger.trace("Final offspring chromosome: {0}".format(offspring))
            return offspring

        if offspring_number == "second":
            while not valid_smiles:
                offspring = np.empty_like(first_parent, dtype=object)
                counter = 0
                c = int(0)
                offspring[c : crossover_pt[0]] = sec_parent[c : crossover_pt[0]]
                offspring[crossover_pt[0] :] = first_parent[crossover_pt[0] :]
                for ci, cj in zip(crossover_pt[::2], crossover_pt[1::2]):
                    if not ci:
                        ci = crossover_pt[0]
                    offspring[c:ci] = sec_parent[c:ci]
                    offspring[ci:] = first_parent[ci:]
                    c = cj
                if self.allowed_mutation_genes is not None:
                    full_offspring[mask_allowed] = offspring[:]
                    offspring = full_offspring
                logger.trace(
                    "Offspring chromosome attempt {0}: {1}".format(counter, offspring)
                )
                valid_smiles = check_smiles_chars(offspring)
                crossover_pt = self.get_crossover_points()
                counter += 1
                if counter > self.max_counter:
                    logger.debug(
                        "Counter in create offspring exceeded {0}, using default.".format(
                            self.max_counter
                        )
                    )
                    valid_smiles = True
                    offspring = backup_sec_parent
            logger.trace("Final offspring chromosome: {0}".format(offspring))
            return offspring

    def mutate_population(self, population, n_mutations):
        """
        Mutates the population by randomizing specific positions of the
        population individuals.
        :param population: the population at a given iteration
        :param n_mutations: number of mutations to be performed.
        :return: the mutated population
        """

        valid_smiles = False
        alpha = np.random.rand(1)[0]
        mutation_rows, mutation_cols = super(
            SmilesGenAlgSolver, self
        ).mutate_population(population, n_mutations)
        for i, j in zip(mutation_rows, mutation_cols):
            backup_gene = population[i, j]
            counter = 0
            while not valid_smiles:
                population[i, j] = np.random.choice(self.alphabet, size=1)[0]
                logger.trace(
                    "Mutated chromosome attempt {0}: {1}".format(
                        counter, population[i, :]
                    )
                )
                if alpha < self.mutation_rate:
                    population[i, 0] = np.random.choice(self.starting_smiles, size=1)[0]
                valid_smiles = check_smiles_chars(population[i, :])
                counter += 1
                if counter > self.max_counter:
                    logger.debug(
                        "Counter in mutate exceeded {0}, using default.".format(
                            self.max_counter
                        )
                    )
                    population[i, j] = backup_gene
                    valid_smiles = True
            valid_smiles = False

        return population

    def write_population(self, basename="chromosome"):
        """
        Print xyz for all the population at the current state.
        """
        for i, j in zip(range(self.pop_size), self.fitness_):
            sc2depictions(
                self.population_[i][:],
                "{0}_{1}_{2}".format(basename, i, np.round(j, 4)),
            )
