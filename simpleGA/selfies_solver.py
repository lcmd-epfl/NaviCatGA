from typing import Sequence
import logging
import random
import numpy as np

from selfies import (
    get_alphabet_from_selfies,
    get_semantic_robust_alphabet,
    set_semantic_constraints,
)
from simpleGA.base_solver import GenAlgSolver
from simpleGA.chemistry_selfies import (
    sanitize_multiple_smiles,
    get_selfie_chars,
    check_selfie_chars,
    randomize_selfies,
)
from simpleGA.exceptions import InvalidInput
from simpleGA.exception_messages import exception_messages
from rdkit import rdBase


logger = logging.getLogger(__name__)


class SelfiesGenAlgSolver(GenAlgSolver):
    def __init__(
        self,
        starting_selfies: list = ["[nop]"],
        starting_random: bool = False,
        starting_stoned: bool = False,
        alphabet_list: list = list(get_semantic_robust_alphabet()),
        branching: bool = False,
        variables_limits: bool = False,
        max_counter: int = 10,
        # Parameters for base class
        n_genes: int = 1,
        fitness_function=None,
        max_gen: int = 500,
        max_conv: int = 100,
        pop_size: int = 100,
        mutation_rate: float = 0.05,
        selection_rate: float = 0.25,
        selection_strategy: str = "tournament",
        excluded_genes: Sequence = None,
        n_crossover_points: int = 1,
        random_state: int = None,
        lru_cache: bool = False,
        hashable_fitness_function=None,
        scalarizer=None,
        prune_duplicates=False,
        # Verbosity and printing options
        verbose: bool = True,
        show_stats: bool = False,
        plot_results: bool = False,
        to_stdout: bool = True,
        to_file: bool = True,
        logger_file: str = "output.log",
        logger_level: str = "INFO",
        progress_bars: bool = False,
        problem_type="selfies",
    ):
        """Example child solver class for the GA.
        This child solver class is an example meant for a particular purpose,
        which also shows how to use the GA with SELFIES as a core molecular representation.
        It might require heavy modification for other particular usages.
        Only the parameters specific for this child class are covered here.

        Parameters:
        :param starting_selfies: list containing the starting SELFIES elements for all chromosomes; overridden by starting_random=True
        :type starting_selfies: list
        :param starting_random: whether to initialize all chromosomes with random elements from alphabet; overrides starting_selfies
        :type starting_random: bool
        :param starting_stoned: whether to use the STONED methodology to generate a chemical subspace from starting_selfies; incompatible with starting_random
        :type starting_stoned: bool
        :param alphabet_list: list containing the alphabets for the individual genes; or a single alphabet for all
        :type alphabet_list: list
        :param branching: whether to add random branches covering all possible branching possibilities to the alphabets 
        :type branching: bool
        :param variable_limits: will set semantic constraints on alphabets if set True
        :type variable_limits: bool
        :param max_counter: maximum number of times a wrong structure will try to be corrected before skipping
        :type max_counter: int
        """
        GenAlgSolver.__init__(
            self,
            fitness_function=fitness_function,
            hashable_fitness_function=hashable_fitness_function,
            scalarizer=scalarizer,
            n_genes=n_genes,
            max_gen=max_gen,
            max_conv=max_conv,
            pop_size=pop_size,
            mutation_rate=mutation_rate,
            selection_rate=selection_rate,
            selection_strategy=selection_strategy,
            verbose=verbose,
            show_stats=show_stats,
            plot_results=plot_results,
            excluded_genes=excluded_genes,
            prune_duplicates=prune_duplicates,
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

        if all(isinstance(i, list) for i in alphabet_list):
            if not (len(alphabet_list) == n_genes):
                raise (InvalidInput(exception_messages["AlphabetDimensions"]))
            self.alphabet = alphabet_list
            self.multi_alphabet = True
            if excluded_genes is not None:
                raise (InvaludInput(exception_messages["MultiDictExcluded"]))
            if equivalences is None:
                equivalences = []
                for j in range(n_genes):
                    equivalences.append([j])
                    for k in range(n_genes):
                        if list(self.alphabet[j]) == list(self.alphabet[k]):
                            equivalences[j].append(k)
                tpls = [tuple(x) for x in equivalences]
                dct = list(dict.fromkeys(tpls))
                equivalences = [list(x) for x in dct]
                self.equivalences = equivalences
                logger.debug(f"Equivalence classes are {equivalences}")
            else:
                if len(equivalences) > n_genes:
                    raise (InvalidInput(exception_messages["EquivalenceDimensions"]))
        else:
            self.multi_alphabet = False
            sorted_list = list(sorted(alphabet_list))
            self.alphabet = [sorted_list] * n_genes
            assert len(self.alphabet) == n_genes

        if variables_limits:
            set_semantic_constraints(variables_limits)

        if branching and not self.multi_alphabet:
            tuples = [
                (i + 1, j + 1) for i in range(self.n_genes) for j in range(self.n_genes)
            ]
            for i in tuples:
                if i[0] == i[1]:
                    pass
                else:
                    alphabet.add("[Branch{0}_{1}]".format(i[0], i[1]))
                    pass

        if not isinstance(starting_selfies, list):
            raise (InvalidInput(exception_messages["StartingSelfiesNotAList"]))
        if not self.alphabet:
            raise (InvalidInput(exception_messages["AlphabetIsEmpty"]))
        if self.n_crossover_points > self.n_genes:
            raise (InvalidInput(exception_messages["TooManyCrossoverPoints"]))
        if self.n_crossover_points < 1:
            raise (InvalidInput(exception_messages["TooFewCrossoverPoints"]))
        if starting_random and starting_stoned:
            raise (InvalidInput(exception_messages["ConflictedRandomStoned"]))
        if starting_stoned and (len(starting_selfies) != 1):
            raise (InvalidInput(exception_messages["ConflictedStonedStarting"]))

        if starting_random:
            logger.warning(
                "Randomizing starting population. Any starting chromosomes will be overwritten."
            )
            starting_selfies = list([""] * self.pop_size)
            for i in range(self.pop_size):
                for j in range(random.randint(1, self.n_genes)):
                    starting_selfies[i] += np.random.choice(self.alphabet[j], size=1)[0]
        elif starting_stoned:
            starting_selfies = randomize_selfies(
                starting_selfies[0], num_random=self.pop_size
            )
        if len(starting_selfies) < self.pop_size:
            n_patch = self.pop_size - len(starting_selfies)
            for i in range(n_patch):
                starting_selfies.append(np.random.choice(starting_selfies, size=1)[0])
        elif len(starting_selfies) > self.pop_size:
            n_remove = len(starting_selfies) - self.pop_size
            for i in range(n_remove):
                starting_selfies.remove(np.random.choice(starting_selfies, size=1)[0])
        assert len(starting_selfies) == self.pop_size
        self.starting_selfies = starting_selfies
        self.max_counter = int(max_counter)

    def initialize_population(self):
        """
        Initializes the population of the problem according to the
        population size and number of genes and according to the problem
        type (either integers or floats).
        
        Returns:
        :return: a numpy array with initialized population
        """

        population = np.zeros(shape=(self.pop_size, self.n_genes), dtype=object)

        for i in range(self.pop_size):
            logger.debug(
                "Getting selfie chars from {0}".format(self.starting_selfies[i])
            )
            chromosome = get_selfie_chars(self.starting_selfies[i], self.n_genes)
            assert check_selfie_chars(chromosome)
            population[i][:] = chromosome[0 : self.n_genes]

        self.logger.debug("Initial population: {0}".format(population))
        return population

    def refill_population(self, nrefill=0):

        assert nrefill > 0
        ref_pop = np.zeros(shape=(nrefill, self.n_genes), dtype=object)

        for i in range(nrefill):
            chromosome = get_selfie_chars(self.starting_selfies[i], self.n_genes)
            for j in range(1, self.n_genes):
                chromosome[j] = np.random.choice(self.alphabet[j], size=1)[0]
            assert check_selfie_chars(chromosome)
            ref_pop[i][:] = chromosome[0 : self.n_genes]

        self.logger.debug("Refill subset for population:\n{0}".format(ref_pop))
        return ref_pop

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

        if beta < 0.5:
            first_parent = first_parent[::-1]
        if gamma > 0.5:
            sec_parent = sec_parent[::-1]

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
                valid_smiles = check_selfie_chars(offspring)
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
                valid_smiles = check_selfie_chars(offspring)
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
        mutation_rows, mutation_cols = super(
            SelfiesGenAlgSolver, self
        ).mutate_population(population, n_mutations)
        for i, j in zip(mutation_rows, mutation_cols):
            backup_gene = population[i, j]
            counter = 0
            while not valid_smiles:
                population[i, j] = np.random.choice(self.alphabet[j], size=1)[0]
                logger.trace(
                    "Mutated chromosome attempt {0}: {1}".format(
                        counter, population[i, :]
                    )
                )
                valid_smiles = check_selfie_chars(population[i, :])
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


def test_benzene_selfies():
    from simpleGA.fitness_functions_selfies import fitness_function_selfies

    starting_selfies = ["[C][C=][C][C=][C][C=][Ring1][Branch1_2]"]
    solver = SelfiesGenAlgSolver(
        n_genes=16,
        pop_size=5,
        max_gen=10,
        fitness_function=fitness_function_selfies(2),
        starting_selfies=starting_selfies,
        excluded_genes=[0, 1, 2, 3, 4, 5, 6, 7, 8],
        logger_level="INFO",
        n_crossover_points=2,
        verbose=False,
        progress_bars=True,
        to_file=False,
        to_stdout=True,
    )
    solver.solve()


if __name__ == "__main__":
    test_benzene_selfies()
