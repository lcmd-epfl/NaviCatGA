import logging
import numpy as np
from simpleGA.evo import (
    sanitize_smiles,
    timed_decoder,
    get_structure_ff,
    get_selfie_chars,
    get_interatomic_distances,
    get_ECFP4,
)
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Draw, Descriptors
from rdkit.DataStructs.cDataStructs import TanimotoSimilarity
from rdkit.Chem.rdmolfiles import MolToPDBFile as mol2pdb
from rdkit.Chem.rdmolfiles import MolToXYZFile as mol2xyz

logger = logging.getLogger(__name__)
lg = RDLogger.logger()
lg.setLevel(RDLogger.ERROR)
RDLogger.DisableLog("rdApp.*")


def sc2selfies(chromosome):
    """Generate a selfies string from a list of selfies characters."""
    selfie = "".join(x for x in list(chromosome))
    return selfie


def sc2smiles(chromosome):
    """Generate a canonical smiles string from a list of selfies characters."""
    selfie = "".join(x for x in list(chromosome))
    smiles = timed_decoder(selfie)
    mol, smi_canon, check = sanitize_smiles(smiles)
    if check:
        return smi_canon
    else:
        return None


def sc2depictions(chromosome, root_name="output", lot=0):
    """Generate 2D and 3D depictions from a list of selfies characters."""
    mol_structure = sc2mol_structure(chromosome, lot=lot)
    mol2pdb(mol_structure, "{0}.pdb".format(root_name))
    mol2xyz(mol_structure, "{0}.xyz".format(root_name))
    Draw.MolToFile(mol_structure, "{0}.png".format(root_name))
    logger.info("Generated depictions with root name {0}".format(root_name))


def mol_structure2depictions(mol_structure, root_name="output"):
    """Generate 2D and 3D depictions from an rdkit.mol object with 3D coordinates."""
    mol2pdb(mol_structure, "{0}.pdb".format(root_name))
    mol2xyz(mol_structure, "{0}.xyz".format(root_name))
    Draw.MolToFile(mol_structure, "{0}.png".format(root_name))


def sc2mol_structure(chromosome, lot=0):
    """Generates a rdkit.mol object with 3D coordinates from a list of selfies characters."""
    selfie = "".join(x for x in list(chromosome))
    smiles = timed_decoder(selfie)
    mol, smi_canon, check = sanitize_smiles(smiles)
    if not check:
        logger.exception("SMILES {0} cannot be sanitized".format(smiles))
    if lot == 0:
        return get_structure_ff(mol, n_confs=5)
    if lot == 1:
        exit()


def sc2logp(chromosome):
    selfie = "".join(x for x in list(chromosome))
    smiles = timed_decoder(selfie)
    mol, smiles_canon, done = sanitize_smiles(smiles)
    if smiles_canon == "" or not done or mol is None:
        return -1e6
    logp = Descriptors.MolLogP(mol)
    return logp


def sc2ilogp(chromosome):
    selfie = "".join(x for x in list(chromosome))
    smiles = timed_decoder(selfie)
    tol = 1e-6
    mol, smiles_canon, done = sanitize_smiles(smiles)
    if smiles_canon == "" or not done or mol is None:
        return -1e6
    logp = Descriptors.MolLogP(mol)
    ilogp = 1 / (logp + tol)
    return ilogp


def sc2mw(chromosome):
    selfie = "".join(x for x in list(chromosome))
    smiles = timed_decoder(selfie)
    mol, smiles_canon, done = sanitize_smiles(smiles)
    if smiles_canon == "" or not done or mol is None:
        return -1e6
    mw = np.round(Descriptors.HeavyAtomMolWt(mol))
    return mw


def sc2nmw(chromosome):
    selfie = "".join(x for x in list(chromosome))
    smiles = timed_decoder(selfie)
    mol, smiles_canon, done = sanitize_smiles(smiles)
    if smiles_canon == "" or not done or mol is None:
        return -1e6
    mw = np.round(Descriptors.HeavyAtomMolWt(mol))
    nmw = -mw
    return nmw


def sc2mwilogp(chromosome):
    selfie = "".join(x for x in list(chromosome))
    smiles = timed_decoder(selfie)
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
            "Molecular Volume not be evaluated for chromosome. SMILES : {0}".format(
                timed_decoder("".join(x for x in list(chromosome)))
            )
        )
        logger.debug(m)
        mv = -1e6
    return mv


def sc2levenshtein_to_target(chromosome, target_selfie):
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


def sc2tanimoto_to_target(chromosome, target_selfie):
    selfie = "".join(x for x in list(chromosome))
    smi1 = timed_decoder(selfie)
    smi2 = timed_decoder(target_selfie)
    mol1, smi_canon1, done1 = sanitize_smiles(smi1)
    mol2, smi_canon2, done2 = sanitize_smiles(smi2)
    fp1 = get_ECFP4(mol1)
    fp2 = get_ECFP4(mol2)
    return TanimotoSimilarity(fp1, fp2)


def sc2cm(chromosome, order="C"):
    try:
        mol = sc2mol_structure(chromosome)
        m = mol_structure2cm(mol)
        if order == "C":
            cm = m.flatten(order="C")
        elif order == "F":
            cm = m.flatten(order="F")
    except Exception as m:
        logger.warning(
            "Fitness could not be evaluated for chromosome. SMILES : {0}".format(
                timed_decoder("".join(x for x in list(chromosome)))
            )
        )
        logger.debug(m)
        cm = None
    return cm


def mol_structure2cm(mol):
    n_atoms = mol.GetNumAtoms()
    z = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    d = get_interatomic_distances(mol)
    m = np.zeros((n_atoms, n_atoms))
    for mol_structure in mol.GetConformers():
        for i in range(mol.GetNumAtoms()):
            for j in range(mol.GetNumAtoms()):
                if i == j:
                    m[i, j] = 0.5 * z[i] ** 2.4
                elif i < j:
                    m[i, j] = (z[i] * z[j]) / d[i, j]
                    m[j, i] = m[i, j]
                else:
                    continue
    return m


def sc2krr(chromosome):
    return 50
