import logging

from simpleGA.score_modifiers import score_modifier
from simpleGA.wrappers_xyz import (
    gl2bond,
    gl2angle,
    gl2dihedral,
)


logger = logging.getLogger(__name__)


def fitness_function_target_property(
    target, function_number=1, score_modifier_number=1, parameter=1, **kwargs
):

    if function_number == 1:  # gl2bond

        return lambda chromosome: score_modifier(
            gl2bond(chromosome, kwargs("a1"), kwargs("a2")),
            target,
            score_modifier_number,
            parameter,
        )


def fitness_function_selfies(function_number=1, **kwargs):

    if function_number == 1:  # gl2bond

        return lambda chromosome: gl2bond(chromosome, kwargs("a1"), kwargs("a2"))
