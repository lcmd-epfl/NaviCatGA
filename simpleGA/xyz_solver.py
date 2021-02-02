from typing import Sequence
import logging
import numpy as np

from simpleGA.genetic_algorithm_base import GenAlgSolver
from simpleGA.chemistry_xyz import (
    get_starting_xyz_fromfile,
    get_starting_xyz_fromsmi,
    pad_xyz_list,
    check_xyz,
    get_default_dictionary,
    get_dictionary_from_path,
)
from simpleGA.exceptions import InvalidInput
from simpleGA.exception_messages import exception_messages
from rdkit import rdBase


logger = logging.getLogger(__name__)


class XYZGenAlgSolver(GenAlgSolver):
    def __init__(
        self,
        n_genes: int,
        starting_scaffolds: list = [],
        starting_xyz: list = [],
        starting_random: bool = False,
        starting_stoned: bool = False,
        alphabet_choice: str = "default",
        fitness_function=None,
        max_gen: int = 5,
        pop_size: int = 5,
        mutation_rate: float = 0.05,
        selection_rate: float = 0.25,
        selection_strategy: str = "tournament",
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
    ):
        if starting_scaffolds:
            starting_xyz = get_starting_xyz_fromfile(starting_scaffolds)
        if alphabet_choice == "default":
            self.alphabet = get_default_dictionary()
        else:
            self.alphabet = get_dictionary_from_path(alphabet_choice)

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
            lru_cache=lru_cache,
        )

        if variables_limits is not None:
            pass  # TBD

        if not isinstance(starting_xyz, list):
            raise (InvalidInput(exception_messages["StartingXYZNotAList"]))
        if not self.alphabet:
            raise (InvalidInput(exception_messages["AlphabetIsEmpty"]))
        if self.n_crossover_points > self.n_genes:
            raise (InvalidInput(exception_messages["TooManyCrossoverPoints"]))
        if self.n_crossover_points < 1:
            raise (InvalidInput(exception_messages["TooFewCrossoverPoints"]))
        if starting_random and starting_stoned:
            raise (InvalidInput(exception_messages["ConflictedRandomStoned"]))
        if starting_stoned and (len(starting_xyz) != 1):
            raise (InvalidInput(exception_messages["ConflictedStonedStarting"]))

        if starting_stoned:
            pass  # TBD
            # starting_xyz = randomize_xyz(
            #     starting_selfies[0], num_random=self.pop_size
            # )
        if len(starting_xyz) < self.pop_size:
            n_patch = self.pop_size - len(starting_xyz)
            for i in range(n_patch):
                starting_xyz.append(
                    np.random.choice(starting_xyz, size=1)[0]
                )  # OKAYISH
        elif len(starting_xyz) > self.pop_size:
            n_remove = len(starting_xyz) - self.pop_size
            for i in range(n_remove):
                starting_xyz.remove(
                    np.random.choice(starting_xyz, size=1)[0]
                )  # MIGHT BE IMPROVABLE
        assert len(starting_xyz) == self.pop_size
        self.starting_random = starting_random
        self.starting_xyz = starting_xyz
        self.max_counter = int(max_counter)

    def initialize_population(self):
        """
        Initializes the population of the problem according to the
        population size and number of genes and according to the problem
        type (XYZ fragmenta here).
        :return: a numpy array with a sanitized initialized population
        """

        population = np.zeros(shape=(self.pop_size, self.n_genes), dtype=object)

        for i in range(self.pop_size):
            logger.debug("Getting scaffold from:\n{0}".format(self.starting_xyz[i]))
            chromosome = pad_xyz_list(self.starting_xyz[i], self.n_genes)
            if self.starting_random:
                for j in range(1, self.n_genes):
                    chromosome[j] = np.random.choice(self.alphabet, size=1)[0]
            assert check_xyz(chromosome)
            population[i][:] = chromosome[0 : self.n_genes]

        self.logger.debug("Initial population:\n{0}".format(population))
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
        valid = False

        if beta < 0.5:
            first_parent = first_parent[::-1]
        if gamma > 0.5:
            sec_parent = sec_parent[::-1]

        if offspring_number == "first":
            while not valid:
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
                    "Offspring chromosome attempt {0}:\n{1}".format(counter, offspring)
                )
                valid = check_xyz(offspring)
                crossover_pt = self.get_crossover_points()
                counter += 1
                if counter > self.max_counter:
                    logger.trace(
                        "Counter in create offspring exceeded {0}, using default.".format(
                            self.max_counter
                        )
                    )
                    valid = True
                    offspring = backup_first_parent
            logger.trace("Final offspring chromosome:\n{0}".format(offspring))
            return offspring

        if offspring_number == "second":
            while not valid:
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
                    "Offspring chromosome attempt {0}:\n{1}".format(counter, offspring)
                )
                valid = check_xyz(offspring)
                crossover_pt = self.get_crossover_points()
                counter += 1
                if counter > self.max_counter:
                    logger.debug(
                        "Counter in create offspring exceeded {0}, using default.".format(
                            self.max_counter
                        )
                    )
                    valid = True
                    offspring = backup_sec_parent
            logger.trace("Final offspring chromosome:\n{0}".format(offspring))
            return offspring

    def mutate_population(self, population, n_mutations):
        """
        Mutates the population by randomizing specific positions of the
        population individuals.
        :param population: the population at a given iteration
        :param n_mutations: number of mutations to be performed.
        :return: the mutated population
        """

        valid = False
        mutation_rows, mutation_cols = super(XYZGenAlgSolver, self).mutate_population(
            population, n_mutations
        )
        for i, j in zip(mutation_rows, mutation_cols):
            backup_gene = population[i, j]
            counter = 0
            while not valid:
                population[i, j] = np.random.choice(self.alphabet, size=1)[0]
                logger.trace(
                    "Mutated chromosome attempt {0}:\n{1}".format(
                        counter, population[i, :]
                    )
                )
                valid = check_xyz(population[i, :])
                counter += 1
                if counter > self.max_counter:
                    logger.debug(
                        "Counter in mutate exceeded {0}, using default.".format(
                            self.max_counter
                        )
                    )
                    population[i, j] = backup_gene
                    valid = True
            valid = False

        return population


def test_cyclohexanes_xyz():
    from simpleGA.fitness_functions_xyz import fitness_function_xyz

    starting_smiles = ["C1CCCCC1"]
    starting_xyz = get_starting_xyz_fromsmi(starting_smiles)
    solver = XYZGenAlgSolver(
        n_genes=13,
        pop_size=5,
        max_gen=10,
        mutation_rate=0.10,
        fitness_function=fitness_function_xyz(1),
        starting_xyz=starting_xyz,
        starting_random=True,
        logger_level="INFO",
        n_crossover_points=1,
        verbose=True,
        progress_bars=True,
        to_file=False,
        to_stdout=True,
    )
    solver.solve()


if __name__ == "__main__":
    test_cyclohexanes_xyz()
