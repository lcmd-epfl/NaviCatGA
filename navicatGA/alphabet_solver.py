import logging

import numpy as np

from navicatGA.base_solver import GenAlgSolver
from navicatGA.helpers import check_error
from navicatGA.exceptions import InvalidInput
from navicatGA.exception_messages import exception_messages

logger = logging.getLogger(__name__)


class AlphabetGenAlgSolver(GenAlgSolver):
    """Shared base for solvers whose chromosome genes are drawn from a
    per-gene alphabet (SMILES fragments, SELFIES tokens, XYZ fragments).

    Factors out the alphabet/equivalences setup, population init/refill,
    crossover-point sampling, and mutation logic that
    SmilesGenAlgSolver/SelfiesGenAlgSolver/XYZGenAlgSolver used to each
    reimplement independently. Subclasses still own `create_offspring`
    (crossover behaviour genuinely differs between them) and
    `write_population` (different depiction backend per representation).
    """

    def _setup_alphabet(
        self, alphabet_list, n_genes, excluded_genes, equivalences, sort_alphabet=True
    ):
        """Populates self.alphabet, self.multi_alphabet and self.equivalences."""

        if all(isinstance(i, list) for i in alphabet_list):
            if not (len(alphabet_list) == n_genes):
                raise (InvalidInput(exception_messages["AlphabetDimensions"]))
            self.alphabet = alphabet_list
            self.multi_alphabet = True
            if excluded_genes is not None:
                raise (InvalidInput(exception_messages["MultiDictExcluded"]))
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
                self.equivalences = equivalences
        else:
            self.multi_alphabet = False
            sorted_list = list(sorted(alphabet_list)) if sort_alphabet else list(
                alphabet_list
            )
            self.alphabet = [sorted_list] * n_genes
            assert len(self.alphabet) == n_genes

    @staticmethod
    def _pad_or_trim_to_pop_size(starting_population, pop_size):
        """Randomly duplicates or drops entries so len(starting_population) == pop_size."""

        if not isinstance(starting_population, list):
            raise (InvalidInput(exception_messages["StartingPopulationNotAList"]))

        if len(starting_population) < pop_size:
            n_patch = pop_size - len(starting_population)
            for _ in range(n_patch):
                j = np.random.choice(range(len(starting_population)), size=1)[0]
                starting_population.append(starting_population[j])
        elif len(starting_population) > pop_size:
            n_remove = len(starting_population) - pop_size
            for _ in range(n_remove):
                j = np.random.choice(range(len(starting_population)), size=1)[0]
                starting_population.remove(starting_population[j])
        assert len(starting_population) == pop_size
        return starting_population

    def initialize_population(self):
        """
        Initializes the population according to the population size and
        number of genes, seeding from self.starting_population and
        optionally randomizing per self.starting_random.

        Returns:
        :return: a numpy array with initialized population
        """

        population = np.zeros(shape=(self.pop_size, self.n_genes), dtype=object)

        for i in range(self.pop_size):
            chromosome = self.chromosomize(self.starting_population[i])
            if self.starting_random:
                logger.debug("Randomizing starting chromosome.")
                for n, j in enumerate(range(self.n_genes)):
                    if n in self.allowed_mutation_genes:
                        chromosome[j] = np.random.choice(self.alphabet[j], size=1)[0]
            assert check_error(self.assembler, chromosome)
            population[i][:] = chromosome[0 : self.n_genes]

        self.logger.debug("Initial population: {0}".format(population))
        self.starting_population = population
        return population

    def refill_population(self, nrefill=0):
        assert nrefill > 0
        ref_pop = np.zeros(shape=(nrefill, self.n_genes), dtype=object)

        for i in range(nrefill):
            chromosome = self.chromosomize(self.starting_population[i])
            for n, j in enumerate(range(self.n_genes)):
                if n in self.allowed_mutation_genes:
                    chromosome[j] = np.random.choice(self.alphabet[j], size=1)[0]
            assert check_error(self.assembler, chromosome)
            ref_pop[i][:] = chromosome[0 : self.n_genes]

        self.logger.debug("Refill subset for population:\n{0}".format(ref_pop))
        return ref_pop

    def get_crossover_points(self):
        """
        Retrieves random crossover points.
        :return: a numpy array with the crossover points
        """

        return np.sort(
            np.random.choice(
                np.arange(self.n_genes), self.n_crossover_points, replace=False
            )
        )

    def mutate_population(self, population, n_mutations):
        """
        Mutates the population by randomizing specific positions of the
        population individuals, retrying until self.assembler accepts the
        result or self.max_counter attempts are exhausted.

        :param population: the population at a given iteration
        :param n_mutations: number of mutations to be performed.
        :return: the mutated population
        """

        valid = False
        mutation_rows, mutation_cols = super().mutate_population(
            population, n_mutations
        )
        for i, j in zip(mutation_rows, mutation_cols):
            backup_gene = population[i, j]
            counter = 0
            while not valid:
                population[i, j] = np.random.choice(self.alphabet[j], size=1)[0]
                logger.trace(
                    "Mutated chromosome attempt {0}: {1}".format(
                        counter, population[i, :]
                    )
                )
                valid = check_error(self.assembler, population[i, :])
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

    def chromosomize(self, str_list):
        """Pad or truncate a starting-population entry to conform to n_genes."""
        logger.debug(f"Chromosomizing {str_list} to conform to n_genes {self.n_genes}")
        if not isinstance(str_list, (list, np.ndarray)):
            raise (InvalidInput("Starting population is not a list of lists."))

        chromosome = np.empty(self.n_genes, dtype=object)
        for i in range(min(self.n_genes, len(str_list))):
            chromosome[i] = str_list[i]
        if len(str_list) > self.n_genes:
            logger.debug("Exceedingly long entry produced. Will be truncated.")
        if len(str_list) < self.n_genes:
            logger.debug("Exceedingly short entry produced. Will be randomly completed.")
            for i in range(1, self.n_genes - len(str_list) + 1):
                chromosome[-i] = np.random.choice(self.alphabet[-i], size=1)[0]
        return chromosome
