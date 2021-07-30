import logging
import numpy as np
from navicatGA.timeout import timeout
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.rdmolfiles import MolToSmiles as mol2smi
from rdkit.Chem.rdmolfiles import MolFromSmiles as smi2mol
from rdkit.Chem.rdmolfiles import MolToPDBFile as mol2pdb
from rdkit.Chem.rdmolfiles import MolToXYZFile as mol2xyz

logger = logging.getLogger(__name__)
lg = RDLogger.logger()
lg.setLevel(RDLogger.ERROR)
RDLogger.DisableLog("rdApp.*")


def sanitize_smiles(smiles):  # Problems with C1C=CC=CC=1[P-1]=[P-1][P-1] for instance
    """Return a canonical smile representation of smi.
    If there are metals, it will try to fix the bonds as dative.

    Parameters:
    :param smi: smile string to be canonicalized
    :type smi: str

    Returns:
    :return mol: rdkit mol object, None if exception caught
    :return smi_canon: canonicalized smile representation of smi, None if exception caught.
    :return conversion_successful: True if no exception caught, False if exception caught.
    """
    try:
        mol, smi_canon, conversion_successful = timed_sanitizer(smiles)
        return (mol, smi_canon, conversion_successful)
    except:
        logger.debug(
            "Smiles {0} probably became stuck in rdkit smi2mol.".format(smiles)
        )
        return (None, None, False)


def sanitize_multiple_smiles(smiles_list):
    """Calls sanitize_smiles for every item in a list.

    Parameters:
    :param smiles_list: list of smile strings to be sanitized.

    Returns:
    :return sanitized_smiles: list of sanitized smile strings or None in case a smiles led to error
    """
    sanitized_smiles = []
    for smi in smiles_list:
        smi_converted = sanitize_smiles(smi)
        sanitized_smiles.append(smi_converted[1])
        if not smi_converted[2]:
            logger.exception("Invalid SMILES encountered : ", smi)
    return sanitized_smiles


def timed_sanitizer(smiles):
    """Convert smiles string to rdkit.mol, call exception and return None if it takes more than 10 seconds to run."""
    with timeout(seconds=10):
        mol = smi2mol(smiles, sanitize=False)
        if mol is not None:
            if has_transition_metals(mol):
                mol = set_dative_bonds(mol)
            smi_canon = mol2smi(mol, isomericSmiles=False, canonical=True)
            mol = smi2mol(smi_canon, sanitize=True)
            return (mol, smi_canon, True)
        else:
            logger.debug("Smiles {0} could not be understood by rdkit.".format(smiles))
            return (None, None, False)


def get_structure_ff(mol, n_confs=5):
    """Generates a reasonable set of 3D structures
    using forcefields for a given rdkit.mol object.
    It will try several 3D generation approaches in rdkit.
    It will try to sample several conformations and get the minima.

    Parameters:
    :param mol: an rdkit mol object
    :type mol: rdkit.mol
    :param n_confs: number of conformations to sample
    :type n_confs: int

    Returns:
    :return mol_structure: mol with 3D coordinate information set
    """
    Chem.SanitizeMol(mol)
    mol = Chem.AddHs(mol)
    coordinates_added = False
    if not coordinates_added:
        try:
            conformer_ids = AllChem.EmbedMultipleConfs(
                mol,
                numConfs=n_confs,
                useExpTorsionAnglePrefs=True,
                useBasicKnowledge=True,
                pruneRmsThresh=1.25,
                enforceChirality=True,
            )
        except:
            logger.debug("Method 1 failed to generate conformations.")
        else:
            if all([conformer_id >= 0 for conformer_id in conformer_ids]):
                coordinates_added = True

    if not coordinates_added:
        try:
            params = params = AllChem.srETKDGv3()
            params.useSmallRingTorsions = True
            conformer_ids = AllChem.EmbedMultipleConfs(
                mol, numConfs=n_confs, params=params
            )
        except:
            logger.debug("Method 2 failed to generate conformations.")
        else:
            if all([conformer_id >= 0 for conformer_id in conformer_ids]):
                coordinates_added = True

    if not coordinates_added:
        try:
            conformer_ids = AllChem.EmbedMultipleConfs(
                mol,
                numConfs=n_confs,
                useRandomCoords=True,
                useBasicKnowledge=True,
                maxAttempts=250,
                pruneRmsThresh=1.25,
                ignoreSmoothingFailures=True,
            )
        except:
            logger.debug("Method 3 failed to generate conformations.")
        else:
            if all([conformer_id >= 0 for conformer_id in conformer_ids]):
                coordinates_added = True
        finally:
            if not coordinates_added:
                diagnose_mol(mol)

    if not coordinates_added:
        logger.exception(
            "Could not embed the molecule. SMILES {0}".format(mol2smi(mol))
        )
        return mol
    else:
        mol_structure = get_confs_ff(mol, maxiters=250)
        return mol_structure


def draw_smiles(smiles, root_name):
    mol, smi_canon, ok = timed_sanitizer(smiles)
    mol_structure = get_structure_ff(mol)
    mol2pdb(mol_structure, "{0}.pdb".format(root_name))
    mol2xyz(mol_structure, "{0}.xyz".format(root_name))
    Draw.MolToFile(mol_structure, "{0}.png".format(root_name))
    logger.info("Generated depictions with root name {0}".format(root_name))


def get_confs_ff(mol, maxiters=250):
    mol_structure = Chem.Mol(mol)
    mol_structure.RemoveAllConformers()
    if Chem.rdForceFieldHelpers.MMFFHasAllMoleculeParams(mol):
        AllChem.MMFFSanitizeMolecule(mol)
        energies = AllChem.MMFFOptimizeMoleculeConfs(
            mol, maxIters=maxiters, nonBondedThresh=15.0
        )
        energies_list = [e[1] for e in energies]
        min_e_index = energies_list.index(min(energies_list))
        mol_structure.AddConformer(mol.GetConformer(min_e_index))
        return mol_structure
    elif Chem.rdForceFieldHelpers.UFFHasAllMoleculeParams(mol):
        energies = AllChem.UFFOptimizeMoleculeConfs(
            mol, maxIters=maxiters, vdwThresh=15.0
        )
        energies_list = [e[1] for e in energies]
        min_e_index = energies_list.index(min(energies_list))
        mol_structure.AddConformer(mol.GetConformer(min_e_index))
        return mol_structure
    else:
        logger.debug("Could not do complete FF typing. SMILES {0}".format(mol2smi(mol)))
        return mol


def prune_mol_conformers(mol, energies_list):
    if mol.GetNumConformers() <= 1:
        return mol
    energies = np.asarray(energies_list, dtype=float)
    rmsd = get_conformer_rmsd(mol)
    sort = np.argsort(energies)
    keep = []
    for i in sort:
        if len(keep) == 0:
            keep.append(i)
            continue
        this_rmsd = rmsd[i][np.asarray(keep, dtype=int)]
        if np.all(this_rmsd >= 1.25):
            keep.append(i)
    mol_structure = Chem.Mol(mol)
    mol_structure.RemoveAllConformers()
    conf_ids = [conf.GetId() for conf in mol.GetConformers()]
    for i in keep:
        conf = mol.GetConformer(conf_ids[i])
        mol_structure.AddConformer(conf, assignId=True)
    return mol_structure


def get_conformer_rmsd(mol):
    rmsd = np.zeros((mol.GetNumConformers(), mol.GetNumConformers()), dtype=float)
    for i, ref_conf in enumerate(mol.GetConformers()):
        for j, fit_conf in enumerate(mol.GetConformers()):
            if i >= j:
                continue
            rmsd[i, j] = AllChem.GetBestRMS(
                mol, mol, ref_conf.GetId(), fit_conf.GetId()
            )
            rmsd[j, i] = rmsd[i, j]
    return rmsd


def get_interatomic_distances(conf):
    n_atoms = conf.GetNumAtoms()
    coords = [conf.GetAtomPosition(i) for i in range(n_atoms)]
    d = np.zeros((n_atoms, n_atoms), dtype=float)
    for i in range(n_atoms):
        for j in range(n_atoms):
            if i < j:
                d[i, j] = coords[i].Distance(coords[j])
                d[j, i] = d[i, j]
            else:
                continue
    return d


def get_ECFP4(mol):
    return AllChem.GetMorganFingerprint(mol, 2)


def has_transition_metals(mol):
    """Returns True if the rdkit.mol object passed as argument has a (transition)-metal atom, False if else."""
    if any([is_transition_metal(at) for at in mol.GetAtoms()]):
        return True
    else:
        return False


def is_transition_metal(at):
    """Returns True if the rdkit.Atom object passed as argument is a transition metal, False if else."""
    n = at.GetAtomicNum()
    return (n >= 22 and n <= 29) or (n >= 40 and n <= 47) or (n >= 72 and n <= 79)


def set_dative_bonds(mol, fromAtoms=(7, 8, 15, 16)):
    """Tries to replace bonds with metal atoms by dative bonds, while keeping valence rules enforced. Adapted from G. Landrum."""
    pt = Chem.GetPeriodicTable()
    rwmol = Chem.RWMol(mol)
    rwmol.UpdatePropertyCache(strict=False)
    metals = [at for at in rwmol.GetAtoms() if is_transition_metal(at)]
    for metal in metals:
        for nbr in metal.GetNeighbors():
            if (
                nbr.GetAtomicNum() in fromAtoms
                and nbr.GetExplicitValence() > pt.GetDefaultValence(nbr.GetAtomicNum())
                and rwmol.GetBondBetweenAtoms(
                    nbr.GetIdx(), metal.GetIdx()
                ).GetBondType()
                == Chem.BondType.SINGLE
            ):
                rwmol.RemoveBond(nbr.GetIdx(), metal.GetIdx())
                rwmol.AddBond(nbr.GetIdx(), metal.GetIdx(), Chem.BondType.DATIVE)
    return rwmol


def diagnose_mol(mol):
    """Tries to identify and print to logger whatever was or is wrong with the chemistry_selfies of an rdkit.mol object."""
    problems = Chem.DetectChemistryProblems(mol)
    if len(problems) >= 1:
        for problem in problems:
            logger.warning(problem.GetType())
            logger.warning(problem.Message())


def randomize_smiles(mol):
    if not mol:
        return None

    Chem.Kekulize(mol)
    return mol2smi(
        mol, canonical=False, doRandom=True, isomericSmiles=False, kekuleSmiles=True
    )
