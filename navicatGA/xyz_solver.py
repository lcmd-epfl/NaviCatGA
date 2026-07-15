from typing import Sequence
import logging
import numpy as np

from navicatGA.alphabet_solver import AlphabetGenAlgSolver
from navicatGA.chemistry_xyz import (
    get_default_alphabet,
    get_alphabet_from_path,
    random_merge_xyz,
    draw_xyz,
)
from navicatGA.helpers import check_error
from navicatGA.exceptions import InvalidInput
from navicatGA.exception_messages import exception_messages


logger = logging.getLogger(__name__)


class XYZGenAlgSolver(AlphabetGenAlgSolver):
    # Defaults for the GenAlgSolver (base-class) parameters not otherwise
    # named below; override any of them via **base_kwargs, e.g.
    # XYZGenAlgSolver(..., pop_size=20, lru_cache=True).
    _BASE_DEFAULTS = dict(
        max_gen=15,
        max_conv=100,
        pop_size=5,
        mutation_rate=0.10,
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
        problem_type="xyz",
    )

    def __init__(
        self,
        starting_population: list = [[None]],
        starting_random: bool = False,
        alphabet_list: list = get_default_alphabet(),
        chromosome_to_xyz=random_merge_xyz(),
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
        which also shows how to use the GA with xyz coordinate fragments using AaronTools.py.
        It might require heavy modification for other particular usages.
        Only the parameters specific for this child class are covered here.

        Parameters:
        :param starting_population: list containing the starting AaronTools.py interpretable elements for all chromosomes; overridden by starting_random=True
        :type starting_population: list
        :param starting_random: whether to initialize all chromosomes with random elements from alphabet; overrides starting_selfies
        :type starting_random: bool
        :param alphabet_list: list containing the alphabets for the individual genes; or a single alphabet for all; can be a path directing to a directory containing xyz files
        :type alphabet_list: list
        :param chromosome_to_xyz: object that can take a chromosome and generate an AaronTools.py geometry object
        :type chromosome_to_xyz: object
        :param multi_alphabet: whether alphabet_list contains a single alphabet or a list of n_genes alphabet
        :type multi_alphabet: bool
        :param equivalences: list of integers that set the equivalent genes of a chromosome; see examples for clarification
        :type equivalences: list
        :param max_counter: maximum number of times a wrong structure will try to be corrected before skipping
        :type max_counter: int
        :param base_kwargs: any GenAlgSolver parameter (max_gen, pop_size, mutation_rate, ...); see GenAlgSolver for the full list and defaults
        """
        if chromosome_to_xyz is None:
            raise (InvalidInput("No xyz builder provided."))
        super().__init__(
            fitness_function=fitness_function,
            assembler=chromosome_to_xyz,
            n_genes=n_genes,
            excluded_genes=excluded_genes,
            **{**self._BASE_DEFAULTS, **base_kwargs},
        )
        if isinstance(alphabet_list, str):
            alphabet_list = get_alphabet_from_path(alphabet_list)
        elif not isinstance(alphabet_list, list):
            raise (
                InvalidInput("The alphabet_list provided is neither a path nor a list.")
            )

        # AaronTools geometries aren't orderable, unlike SMILES/SELFIES tokens.
        self._setup_alphabet(
            alphabet_list, n_genes, excluded_genes, equivalences, sort_alphabet=False
        )

        if not isinstance(starting_population, list):
            raise (InvalidInput(exception_messages["StartingPopulationNotAList"]))
        if self.n_crossover_points > self.n_genes:
            raise (InvalidInput(exception_messages["TooManyCrossoverPoints"]))
        if self.n_crossover_points < 1:
            raise (InvalidInput(exception_messages["TooFewCrossoverPoints"]))

        self.starting_random = starting_random
        self.starting_population = self._pad_or_trim_to_pop_size(
            starting_population, self.pop_size
        )
        self.max_counter = int(max_counter)

    def write_population(self, basename="chromosome"):
        """
        Print  xyz for all the population at the current state.
        """
        for i, j in zip(range(self.pop_size), self.fitness_):
            xyz = self.assembler(self.population_[i][:])
            draw_xyz(xyz, f"{basename}_{i}_{np.round(j,4)}")

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
                valid = check_error(self.assembler, offspring)
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
                valid = check_error(self.assembler, offspring)
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
