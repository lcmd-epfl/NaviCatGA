import numpy as np
import logging

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs.cDataStructs import TanimotoSimilarity
from selfies import decoder

from chemistry.evo import sanitize_smiles, get_selfie_chars
from chemistry.wrappers import (
    sc2logp,
    sc2ilogp,
    sc2mw,
    sc2mv,
    sc2nmw,
    sc2mwilogp,
    sc2krr,
)


logger = logging.getLogger(__name__)


def get_ECFP4(mol):
    return AllChem.GetMorganFingerprint(mol, 2)


def levenshtein(chromosome, target_selfie):
    reward = 0
    sc1 = chromosome
    l1 = len(sc1)
    sc2 = get_selfie_chars(target_selfie, maxchars=l1)
    l2 = len(sc2)
    iterations = max(l1, l2)

    for i in range(iterations):

        if i + 1 > len(sc1) or i + 1 > len(sc2):
            return reward

        if sc1[i] == sc2[i]:
            reward += 1

    return reward


def tanimoto(chromosome, target_selfie):
    selfie = "".join(x for x in list(chromosome))
    smi1 = decoder(selfie)
    smi2 = decoder(target_selfie)
    mol1, smi_canon1, done1 = sanitize_smiles(smi1)
    mol2, smi_canon2, done2 = sanitize_smiles(smi2)
    fp1 = get_ECFP4(mol1)
    fp2 = get_ECFP4(mol2)
    return TanimotoSimilarity(fp1, fp2)


def GaussianModifier(score, target, sigma):
    try:
        score = np.exp(-0.5 * np.power((score - target) / sigma, 2.0))
    except:
        score = 0.0

    return score


def fitness_function_target_selfies(target_selfie, function_number=1):

    if function_number == 1:  # Tanimoto distance

        return lambda chromosome: tanimoto(chromosome, target_selfie)

    if function_number == 2:  # Levenshtein distance

        return lambda chromosome: levenshtein(chromosome, target_selfie)


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
