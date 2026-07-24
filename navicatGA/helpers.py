from collections import namedtuple

import numpy as np


# Convenience return value for GenAlgSolver.solve(): everything solve()
# already stores as solver.best_individual_/solver.best_fitness_/etc,
# bundled into one object so callers don't have to pull attributes off the
# solver by hand.
GAResult = namedtuple(
    "GAResult",
    [
        "best_individual",
        "best_fitness",
        "best_pfitness",
        "population",
        "fitness",
        "generations",
        "runtime",
    ],
)


def get_input_dimensions(lst, n_dim=0):
    if isinstance(lst, (list, tuple)):
        return get_input_dimensions(lst[0], n_dim + 1) if len(lst) > 0 else 0
    else:
        return n_dim


def check_error(func, *args, **kw):
    try:
        func(*args, **kw)
        return True
    except Exception as m:
        return False


def concatenate_list():
    """Default chromosome manipulator: concatenates all elements of a list."""

    def sc2str(chromosome):
        """Generates a single string from a chromosome (list of arbitrary strings)."""
        string = "".join(str(gene) for gene in chromosome)
        return string

    return sc2str


def make_array():
    """Default chromosome manipulator: turns list into array."""

    def sc2ndarray(chromosome):
        """Generates a float vector from a chromosome (list of arbitrary floats)."""
        return chromosome

    return sc2ndarray


def get_elapsed_time(start_time, end_time):
    runtime = (end_time - start_time).seconds

    hours, remainder = np.divmod(runtime, 3600)
    minutes, seconds = np.divmod(remainder, 60)

    time_str = ""

    if hours:
        time_str += f"{hours} hours, "

    if minutes:
        time_str += f"{minutes} minutes, "

    if seconds:
        time_str += f"{seconds} seconds"

    return runtime, time_str


def track_evaluations(iterable, total, desc, enabled):
    """
    Wraps the per-individual fitness loop in a tqdm bar when progress_bars is on.

    Fitness evaluation is the slow part of a generation (an external QM call can
    take minutes per individual), so the bar counts individuals, not generations.
    tqdm writes to stderr, leaving the stdout/file logs clean.

    Parameters:
    :param iterable: the iterable driving the fitness loop
    :param total: number of individuals to be evaluated
    :param desc: label shown to the left of the bar
    :param enabled: whether to show the bar at all

    Returns:
    :return: iterable, wrapped in a tqdm bar if enabled
    """
    if not enabled:
        return iterable
    from tqdm import tqdm

    return tqdm(iterable, total=total, desc=desc, unit="ind", leave=False)
