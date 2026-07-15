from typing import Sequence
import logging
import numpy as np

from navicatGA.alphabet_solver import AlphabetGenAlgSolver
from navicatGA.helpers import check_error, concatenate_list
from navicatGA.chemistry_smiles import draw_smiles
from navicatGA.exceptions import InvalidInput
from navicatGA.exception_messages import exception_messages

logger = logging.getLogger(__name__)


class SmilesGenAlgSolver(AlphabetGenAlgSolver):
    # Defaults for the GenAlgSolver (base-class) parameters not otherwise
    # named below; override any of them via **base_kwargs, e.g.
    # SmilesGenAlgSolver(..., pop_size=50, lru_cache=True).
    _BASE_DEFAULTS = dict(
        max_gen=500,
        max_conv=100,
        pop_size=100,
        mutation_rate=0.05,
        selection_rate=0.25,
        selection_strategy="tournament",
        n_crossover_points=1,
        random_state=None,
        lru_cache=False,
        scalarizer=None,
        prune_duplicates=False,
        verbose=True,
        show_stats=False,
        plot_results=False,
        to_stdout=True,
        to_file=True,
        logger_file="output.log",
        logger_level="INFO",
        progress_bars=False,
        problem_type="smiles",
    )

    def __init__(
        self,
        starting_population: list = [[""]],
        starting_random: bool = False,
        alphabet_list: list = ["C", "N", "P", "O", "[Si]", "F", "[H]"],
        chromosome_to_smiles=concatenate_list(),
        multi_alphabet: bool = False,
        equivalences: Sequence = None,
        max_counter: int = 10,
        n_genes: int = 1,
        fitness_function=None,
        excluded_genes: Sequence = None,
        **base_kwargs,
    ):
        """Example child solver class for the GA.
        This child solver class is an example meant for a particular purpose,
        which also shows how to use the GA with SMILES as a core molecular representation.
        It might require heavy modification for other particular usages.
        Only the parameters specific for this child class are covered here.

        Parameters:
        :param starting_population: list containing the starting SMILES elements for all chromosomes; overridden by starting_random=True
        :type starting_selfies: list
        :param starting_random: whether to initialize all chromosomes with random elements from alphabets; overrides starting_selfies
        :type starting_random: bool
        :param alphabet_list: list containing the alphabets for the individual genes; or a single alphabet for all
        :type alphabet_list: list
        :param chromosome_to_smiles: object that can take a chromosome and generate a smiles string
        :type chromosome_to_smiles: object
        :param multi_alphabet: whether alphabet_list contains a single alphabet or a list of n_genes alphabet
        :type multi_alphabet: bool
        :param equivalences: list of integers that set the equivalent genes of a chromosome; see examples for clarification
        :type equivalences: list
        :param max_counter: maximum number of times a wrong structure will try to be corrected before skipping
        :type max_counter: int
        :param base_kwargs: any GenAlgSolver parameter (max_gen, pop_size, mutation_rate, ...); see GenAlgSolver for the full list and defaults
        """

        if chromosome_to_smiles is None:
            raise (InvalidInput("No smiles builder provided."))
        super().__init__(
            fitness_function=fitness_function,
            assembler=chromosome_to_smiles,
            n_genes=n_genes,
            excluded_genes=excluded_genes,
            **{**self._BASE_DEFAULTS, **base_kwargs},
        )
        self._setup_alphabet(alphabet_list, n_genes, excluded_genes, equivalences)

        if not isinstance(starting_population, list):
            raise (InvalidInput(exception_messages["StartingPopulationNotAList"]))
        if not self.alphabet:
            raise (InvalidInput(exception_messages["AlphabetIsEmpty"]))
        if self.n_crossover_points > self.n_genes:
            raise (InvalidInput(exception_messages["TooManyCrossoverPoints"]))
        if self.n_crossover_points < 1:
            raise (InvalidInput(exception_messages["TooFewCrossoverPoints"]))

        self.starting_population = self._pad_or_trim_to_pop_size(
            starting_population, self.pop_size
        )
        self.max_counter = int(max_counter)
        self.starting_random = starting_random

    def create_offspring(
        self, first_parent, sec_parent, crossover_pt, offspring_number
    ):
        """
        Creates an offspring from 2 parents.
        """
        beta = np.random.rand(1)[0]
        gamma = np.random.rand(1)[0]
        if not self.multi_alphabet:
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
            if beta < 0.5:
                first_parent = first_parent[::-1]
            if gamma > 0.5:
                sec_parent = sec_parent[::-1]
        else:
            beta = np.random.rand(1)[0]
            gamma = np.random.rand(1)[0]
            backup_sec_parent = sec_parent
            backup_first_parent = first_parent
            for group in self.equivalences:
                mask_allowed = np.zeros(backup_first_parent.size, dtype=bool)
                mask_forbidden = np.ones(backup_first_parent.size, dtype=bool)
                mask_allowed[group] = True
                mask_forbidden[group] = False
                full_offspring = np.empty_like(backup_first_parent, dtype=object)
                full_offspring[mask_forbidden] = backup_first_parent[mask_forbidden]
                first_parent = backup_first_parent[mask_allowed]
                sec_parent = backup_sec_parent[mask_allowed]
                if beta < 0.5:
                    first_parent = first_parent[::-1]
                if gamma > 0.5:
                    sec_parent = sec_parent[::-1]
                backup_first_parent[mask_allowed] = first_parent
                backup_sec_parent[mask_allowed] = sec_parent
            sec_parent = backup_sec_parent
            first_parent = backup_first_parent

        offspring = np.empty_like(first_parent, dtype=object)
        valid_smiles = False

        if offspring_number == "first":
            counter = 0
            while not valid_smiles:
                offspring = np.empty_like(first_parent, dtype=object)
                c = int(0)
                offspring[c : crossover_pt[0]] = first_parent[c : crossover_pt[0]]
                offspring[crossover_pt[0] :] = sec_parent[crossover_pt[0] :]
                for ci, cj in zip(crossover_pt[::2], crossover_pt[1::2]):
                    offspring[c:ci] = first_parent[c:ci]
                    offspring[ci:] = sec_parent[ci:]
                    c = cj
                if not self.multi_alphabet:
                    full_offspring[mask_allowed] = offspring[:]
                    offspring = full_offspring
                logger.trace(
                    "Offspring chromosome attempt {0}: {1}".format(counter, offspring)
                )
                valid_smiles = check_error(self.assembler, offspring)
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
            counter = 0
            while not valid_smiles:
                offspring = np.empty_like(first_parent, dtype=object)
                c = int(0)
                offspring[c : crossover_pt[0]] = sec_parent[c : crossover_pt[0]]
                offspring[crossover_pt[0] :] = first_parent[crossover_pt[0] :]
                for ci, cj in zip(crossover_pt[::2], crossover_pt[1::2]):
                    if not ci:
                        ci = crossover_pt[0]
                    offspring[c:ci] = sec_parent[c:ci]
                    offspring[ci:] = first_parent[ci:]
                    c = cj
                if not self.multi_alphabet:
                    full_offspring[mask_allowed] = offspring[:]
                    offspring = full_offspring
                logger.trace(
                    "Offspring chromosome attempt {0}: {1}".format(counter, offspring)
                )
                valid_smiles = check_error(self.assembler, offspring)
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

    def write_population(self, basename="chromosome"):
        """
        Print xyz for all the population at the current state.
        """
        for i, j in zip(range(self.pop_size), self.fitness_):
            smiles = self.assembler(self.population_[i][:])
            draw_smiles(smiles, f"{basename}_{i}_{np.round(j,4)}")
