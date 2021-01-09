import logging
from chemistry.evo import sanitize_smiles, timed_decoder, get_structure_ff
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.rdmolfiles import MolToPDBFile as mol2pdb
from rdkit.Chem.rdmolfiles import MolToXYZFile as mol2xyz

logger = logging.getLogger(__name__)
lg = RDLogger.logger()
lg.setLevel(RDLogger.ERROR)
RDLogger.DisableLog("rdApp.*")


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
