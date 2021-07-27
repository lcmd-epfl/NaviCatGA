exception_messages = {
    "InvalidSelectionStrategy": lambda selection_strategy, allowed_selection_strategies: f"{selection_strategy} is not a valid selection strategy. "
    f"Available options are {', '.join(allowed_selection_strategies)}.",
    "InvalidPopulationSize": "The population size must be larger than 2",
    "InvalidExcludedGenes": lambda excluded_genes: f"{excluded_genes} is not a valid input for excluded_genes",
    "StartingSelfiesNotAList": "starting_selfies must be a list even if it only contains one string",
    "StartingSmilesNotAList": "starting_smiles must be a list even if it only contains one string",
    "StartingXYZNotAList": "starting_XYZ must be a list even if it only contains one string",
    "ConflictedRandomStoned": "starting_random is not compatible with starting_stoned",
    "ConflictedStonedStarting": "there must be exactly one item in starting_selfies for starting_stoned to work",
    "AlphabetIsEmpty": "at least one element is required in the alphabet for mutation and randomization purposes",
    "AlphabetDimensions": "the input alphabet seems to be a nested list, from which an alphabet list for each gene is expected",
    "EquivalenceDimensions": "the input equivalences define more subgroups than genes per chromosome, which is wrong",
    "MultiDictExcluded": "multiple dictionaries and excluded genes are not compatible. Set up one element dictionaries for exclusion",
    "TooManyCrossoverPoints": "n_crossover_points must be smaller than n_genes",
    "TooFewCrossoverPoints": "n_crossover_points must be at least 1 for the genetic algorithm to work",
}
