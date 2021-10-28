import numpy as np
import logging

logger = logging.getLogger(__name__)


def GaussianModifier(score, target, parameter=1) -> float:
    if not np.isclose(target, 0.0):
        sigma = parameter * (abs(target) / 2)
    else:
        sigma = parameter
    try:
        score = np.exp(-0.5 * np.power((score - target) / sigma, 2.0))
    except Exception as m:
        logger.warning("Gaussian-modified score could not be evaluated for chromosome.")
        logger.debug(m)
        score = -1e6
    return np.around(score, decimals=4)


def AbsoluteModifier(score, target, parameter=1) -> float:
    score = 1.0 - (parameter * np.abs(target - score))
    return np.around(score, decimals=4)


def SquaredModifier(score, target, parameter=1) -> float:
    score = 1.0 - (parameter * np.square(target - score))
    return np.around(score, decimals=4)


def score_modifier(score, target, score_modifier_number=1, parameter=1) -> float:

    if score_modifier_number == 1:

        return GaussianModifier(score, target, parameter)

    if score_modifier_number == 2:

        return AbsoluteModifier(score, target, parameter)

    if score_modifier_number == 3:

        return SquaredModifier(score, target, parameter)
