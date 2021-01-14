exception_messages = {
    "InvalidSelectionStrategy": lambda selection_strategy, allowed_selection_strategies: f"{selection_strategy} is not a valid selection strategy. "
    f"Available options are {', '.join(allowed_selection_strategies)}.",
    "InvalidPopulationSize": "The population size must be larger than 2",
    "InvalidExcludedGenes": lambda excluded_genes: f"{excluded_genes} is not a valid input for excluded_genes",
    "ConflictedRandomStoned": "starting_random is not compatible with starting_stoned",
    "ConflictedStonedStarting": "there must be exactly one item in starting_selfies for starting_stoned to work",
    "AlphabetIsEmpty": "at least one element is required in the alphabet for mutation and randomization purposes",
    "TooManyCrossoverPoints": "n_crossover_points must be smaller than n_genes",
    "TooFewCrossoverPoints": "n_crossover_points must be at least 1 for the genetic algorithm to work",
}
