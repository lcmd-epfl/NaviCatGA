import logging

from simpleGA.score_modifiers import score_modifier
from simpleGA.wrappers_selfies import (
    sc2logp,
    sc2ilogp,
    sc2mw,
    sc2mv,
    sc2nmw,
    sc2mwilogp,
    sc2levenshtein_to_target,
    sc2tanimoto_to_target,
    sc2krr,
)
from simpleGA.quantum_wrappers_selfies import sc2gap, sc2ehomo, sc2elumo


logger = logging.getLogger(__name__)


def fitness_function_target_property(
    target, function_number=1, score_modifier_number=1, parameter=1
):

    if function_number == 1:  # sc2logp logp

        return lambda chromosome: score_modifier(
            sc2logp(chromosome), target, score_modifier_number, parameter
        )

    if function_number == 3:  # sc2mw molecular weight

        return lambda chromosome: score_modifier(
            sc2mw(chromosome), target, score_modifier_number, parameter
        )

    if function_number == 6:  # sc2mv molecular volume

        return lambda chromosome: score_modifier(
            sc2mv(chromosome), target, score_modifier_number, parameter
        )

    if function_number == 9:  # sc2gap homo-lumo gap

        return lambda chromosome: score_modifier(
            sc2gap(chromosome), target, score_modifier_number, parameter
        )


def fitness_function_target_selfies(target_selfie, function_number=1):

    if function_number == 1:  # Tanimoto distance

        return lambda chromosome: sc2tanimoto_to_target(chromosome, target_selfie)

    if function_number == 2:  # Levenshtein distance

        return lambda chromosome: sc2levenshtein_to_target(chromosome, target_selfie)


def fitness_function_selfies(function_number=1):

    if function_number == 1:  # sc2logp logp

        return lambda chromosome: sc2logp(chromosome)

    if function_number == 2:  # sc2ilogp inverse logp

        return lambda chromosome: sc2ilogp(chromosome)

    if function_number == 3:  # sc2mw molecular weight

        return lambda chromosome: sc2mw(chromosome)

    if function_number == 4:  # sc2nmw negative molecular weight to avoid singularity

        return lambda chromosome: sc2nmw(chromosome)

    if function_number == 5:  # sc2mwilogp product of mw and inverse logp

        return lambda chromosome: sc2mwilogp(chromosome)

    if function_number == 6:  # sc2mv molecular volume

        return lambda chromosome: sc2mv(chromosome)

    if function_number == 7:  # sc2ehomo homo energy

        return lambda chromosome: sc2ehomo(chromosome)

    if function_number == 8:  # sc2elumo lumo energy

        return lambda chromosome: sc2elumo(chromosome)

    if function_number == 9:  # sc2gap gap

        return lambda chromosome: sc2gap(chromosome)
