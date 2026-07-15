"""Template for the two functions every navicatGA run needs.

Reference them from your config.yaml (see the *.yaml templates in this
directory) by dotted path, e.g.:

    solver:
      chromosome_to_smiles: your_project.assembler_and_fitness:assemble
      fitness_function: your_project.assembler_and_fitness:score

Copy this file into your project and rename/fill in the two functions - it's
a shape to start from, not something to keep importing from navicatGA.
"""


def assemble(chromosome):
    """Turn a raw chromosome (array/list of genes) into the object to be
    scored - a SMILES/SELFIES string, an AaronTools geometry, a plain
    np.array, etc, depending on your solver type.

    Must raise on an invalid/unbuildable chromosome: navicatGA relies on
    this (via helpers.check_error) to retry mutations/crossovers rather than
    accept a broken result.
    """
    raise NotImplementedError


def score(assembled_object):
    """Score the assembled object.

    Return a single float to optimize directly, or a tuple of floats if
    you're combining several objectives via a 'scalarizer' block in the
    YAML config (e.g. chimera.Chimera - see templates/smiles.yaml).
    """
    raise NotImplementedError
