import numpy as np
import logging

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw, Descriptors
from rdkit.Chem import MolFromSmiles as smi2mol
from rdkit.Chem import MolToSmiles as mol2smi

from selfies import decoder

from chemistry.evo import sanitize_smiles, sc2mol_structure


logger = logging.getLogger(__name__)


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
        mv = AllChem.ComputeMolVolume(mol)
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


def fitness_function_selfies(function_number):

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
