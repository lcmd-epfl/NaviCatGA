import logging
import numpy as np
from navicatGA.chemistry_smiles import (
    sanitize_smiles,
    get_structure_ff,
    get_interatomic_distances,
    get_ECFP4,
)
from rdkit import RDLogger
from rdkit.Chem import AllChem, Draw, Descriptors
from rdkit.DataStructs.cDataStructs import TanimotoSimilarity
from rdkit.Chem.rdmolfiles import MolToPDBFile as mol2pdb
from rdkit.Chem.rdmolfiles import MolToXYZFile as mol2xyz

logger = logging.getLogger(__name__)
lg = RDLogger.logger()
lg.setLevel(RDLogger.ERROR)
RDLogger.DisableLog("rdApp.*")


def check_smiles_chars(chromosome):
    """Checks if a chromosome corresponds to a proper SMILES."""
    smiles = sc2smiles(chromosome)
    logger.debug("Checking SMILES {0} from chromosome {1}".format(smiles, chromosome))
    return sanitize_smiles(smiles)[2]


def chromosome_to_smiles():
    """Wrapper function for simplicity."""

    def sc2smi(chromosome):
        """Generate a SMILES string from a list of SMILES characters. To be customized."""
        silyl = "([Si]([C])([C])([C]))"
        core = chromosome[0]
        phosphine_1 = (
            "(P(" + chromosome[1] + ")(" + chromosome[2] + ")(" + chromosome[3] + "))"
        )
        phosphine_2 = (
            "(P(" + chromosome[4] + ")(" + chromosome[5] + ")(" + chromosome[6] + "))"
        )
        smiles = "{0}{1}{2}{3}".format(core, phosphine_1, phosphine_2, silyl)
        return smiles

    return sc2smi


def sc2smiles(chromosome):
    """Generate a SMILES string from a list of SMILES characters. To be customized."""
    silyl = "([Si]([C])([C])([C]))"
    core = chromosome[0]
    phosphine_1 = (
        "(P(" + chromosome[1] + ")(" + chromosome[2] + ")(" + chromosome[3] + "))"
    )
    phosphine_2 = (
        "(P(" + chromosome[4] + ")(" + chromosome[5] + ")(" + chromosome[6] + "))"
    )
    smiles = "{0}{1}{2}{3}".format(core, phosphine_1, phosphine_2, silyl)
    logger.debug("Chromosome transformed to SMILES {0}".format(smiles))
    return smiles


def sc2depictions(chromosome, root_name="output", lot=0):
    """Generate 2D and 3D depictions from a chromosome."""
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
    smiles = sc2smiles(chromosome)
    mol, smi_canon, check = sanitize_smiles(smiles)
    if not check:
        logger.exception("SMILES {0} cannot be sanitized".format(smiles))
    if lot == 0:
        return get_structure_ff(mol, n_confs=5)
    if lot == 1:
        return get_structure_ff(mol, n_confs=10)


def smiles2mol_structure(smiles, lot=0):
    """Generates a rdkit.mol object with 3D coordinates from a SMILES."""
    mol, smi_canon, check = sanitize_smiles(smiles)
    if not check:
        logger.exception("SMILES {0} cannot be sanitized".format(smiles))
    if lot == 0:
        return get_structure_ff(mol, n_confs=5)
    if lot == 1:
        return get_structure_ff(mol, n_confs=10)


def sc2logp(chromosome):
    smiles = sc2smiles(chromosome)
    mol, smiles_canon, done = sanitize_smiles(smiles)
    if smiles_canon == "" or not done or mol is None:
        return -1e6
    logp = Descriptors.MolLogP(mol)
    return logp


def sc2ilogp(chromosome):
    smiles = sc2smiles(chromosome)
    tol = 1e-6
    mol, smiles_canon, done = sanitize_smiles(smiles)
    if smiles_canon == "" or not done or mol is None:
        return -1e6
    logp = Descriptors.MolLogP(mol)
    ilogp = 1 / (logp + tol)
    return ilogp


def sc2mw(chromosome):
    smiles = sc2smiles(chromosome)
    mol, smiles_canon, done = sanitize_smiles(smiles)
    if smiles_canon == "" or not done or mol is None:
        return -1e6
    mw = np.round(Descriptors.HeavyAtomMolWt(mol))
    return mw


def sc2nmw(chromosome):
    smiles = sc2smiles(chromosome)
    mol, smiles_canon, done = sanitize_smiles(smiles)
    if smiles_canon == "" or not done or mol is None:
        return -1e6
    mw = np.round(Descriptors.HeavyAtomMolWt(mol))
    nmw = -mw
    return nmw


def sc2mwilogp(chromosome):
    smiles = sc2smiles(chromosome)
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
        logger.debug(m)
        mv = -1e6
    return mv


def sc2cm(chromosome, order="C"):
    try:
        mol = sc2mol_structure(chromosome)
        m = mol_structure2cm(mol)
        if order == "C":
            cm = m.flatten(order="C")
        elif order == "F":
            cm = m.flatten(order="F")
    except Exception as m:
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
