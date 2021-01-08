import numpy as np
import logging

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw, Descriptors
from rdkit.Chem import MolFromSmiles as smi2mol
from rdkit.Chem import MolToSmiles as mol2smi
from rdkit.DataStructs.cDataStructs import TanimotoSimilarity
from selfies import decoder

from chemistry.evo import sanitize_smiles, get_selfie_chars, sc2mol_structure


logger = logging.getLogger(__name__)


def get_ECFP4(mol):
    return AllChem.GetMorganFingerprint(mol, 2)


def sc2logp(chromosome):
    selfie = "".join(x for x in list(chromosome))
    smiles = decoder(selfie)
    mol, smiles_canon, done = sanitize_smiles(smiles)
    if smiles_canon == "" or not done or mol is None:
        return -1e6
    logp = Descriptors.MolLogP(mol)
    return logp


def sc2ilogp(chromosome):
    selfie = "".join(x for x in list(chromosome))
    smiles = decoder(selfie)
    tol = 1e-6
    mol, smiles_canon, done = sanitize_smiles(smiles)
    if smiles_canon == "" or not done or mol is None:
        return -1e6
    logp = Descriptors.MolLogP(mol)
    ilogp = 1 / (logp + tol)
    return ilogp


def sc2mw(chromosome):
    selfie = "".join(x for x in list(chromosome))
    smiles = decoder(selfie)
    mol, smiles_canon, done = sanitize_smiles(smiles)
    if smiles_canon == "" or not done or mol is None:
        return -1e6
    mw = np.round(Descriptors.HeavyAtomMolWt(mol))
    return mw


def sc2nmw(chromosome):
    selfie = "".join(x for x in list(chromosome))
    smiles = decoder(selfie)
    mol, smiles_canon, done = sanitize_smiles(smiles)
    if smiles_canon == "" or not done or mol is None:
        return -1e6
    mw = np.round(Descriptors.HeavyAtomMolWt(mol))
    nmw = -mw
    return nmw


def sc2mwilogp(chromosome):
    selfie = "".join(x for x in list(chromosome))
    smiles = decoder(selfie)
    tol = 1e-6
    mol, smiles_canon, done = sanitize_smiles(smiles)
    if smiles_canon == "" or not done or mol is None:
        return -1e6
    logp = Descriptors.MolLogP(mol)
    mw = np.round(Descriptors.HeavyAtomMolWt(mol))
    mwilogp = mw / (np.abs(logp) + tol)
    return mwilogp


def sc2mv(chromosome):
    try:
        mol = sc2mol_structure(chromosome)
        mv = AllChem.ComputeMolVolume(mol, lot=0)
    except Exception as m:
        logger.warning(
            "Fitness could not be evaluated for chromosome. SMILES : {0}".format(
                decoder("".join(x for x in list(chromosome)))
            )
        )
        logger.debug(m)
        mv = -1e6
    return mv


def sc2krr(chromosome):
    return 50


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
