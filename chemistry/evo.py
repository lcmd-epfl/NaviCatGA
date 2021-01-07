import numpy as np
import logging
from utils.timeout import handler, timer_alarm
from selfies import decoder, encoder, split_selfies
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Draw, rdDistGeom
from rdkit.Chem.rdmolfiles import MolToSmiles as mol2smi
from rdkit.Chem.rdmolfiles import MolFromSmiles as smi2mol
from rdkit.Chem.rdmolfiles import MolToPDBFile as mol2pdb
from rdkit.Chem.rdmolfiles import MolToXYZFile as mol2xyz
import rdkit.DistanceGeometry as DG

logger = logging.getLogger(__name__)
lg = RDLogger.logger()
lg.setLevel(RDLogger.ERROR)
RDLogger.DisableLog("rdApp.*")


def sanitize_smiles(smiles):
    """Return a canonical smile representation of smi

    Parameters:
    smi (string) : smile string to be canonicalized

    Returns:
    mol (rdkit.Chem.rdchem.Mol) : RdKit mol object                          (None if invalid smile string smi)
    smi_canon (string)          : Canonicalized smile representation of smi (None if invalid smile string smi)
    conversion_successful (bool): True/False to indicate if conversion was  successful
    """
    try:
        mol = smi2mol(smiles, sanitize=False)
        if has_transition_metals(mol):
            mol = set_dative_bonds(mol)
        smi_canon = mol2smi(mol, isomericSmiles=False, canonical=True)
        mol = smi2mol(smi_canon, sanitize=True)
        return (mol, smi_canon, True)
    except Exception as m:
        logger.exception(m)
        return (None, None, False)


def sanitize_multiple_smiles(smiles_list):
    sanitized_smiles = []
    for smi in smiles_list:
        smi_converted = sanitize_smiles(smi)
        sanitized_smiles.append(smi_converted[1])
        if smi_converted[2] == False:
            logger.exception("Invalid SMILES encountered. Value =", smi)
    return sanitized_smiles


def encode_smiles(smiles_list):
    selfies_list = []
    for smi in smiles_list:
        selfie = encoder(smi)
        selfies_list.append(selfie)
    return selfies_list


def timed_decoder(selfie):  # This is basically a wrapper around selfies.decoder
    timer_alarm(
        90, handler
    )  # If this does not finish within 90 seconds, exception pops
    try:
        selfie = selfie.replace("[nop]", "")
        smiles = decoder(selfie)
    except:
        smiles = selfie  # Will lead to a crash later on...
    return smiles


def check_selfie_chars(chromosome):
    selfie = "".join(x for x in list(chromosome))
    smiles = timed_decoder(selfie)
    logger.debug(
        "Checking SELFIE {0} which was decoded to SMILES {1}".format(selfie, smiles)
    )
    return sanitize_smiles(smiles)[2]


def get_selfie_chars(selfie, maxchars):
    """Obtain an ordered list of all selfie characters in string selfie
    padded to maxchars with [nop]s

    Parameters:
    selfie (string) : A selfie string - representing a molecule

    Example:
    >>> get_selfie_chars('[C][=C][C][=C][C][=C][Ring1][Branch1_1]')
    ['[C]', '[=C]', '[C]', '[=C]', '[C]', '[=C]', '[Ring1]', '[Branch1_1]']

    Returns:
    chars_selfie: list of selfie characters present in molecule selfie
    """
    chars_selfie = []  # A list of all SELFIE sybols from string selfie
    while selfie != "":
        chars_selfie.append(selfie[selfie.find("[") : selfie.find("]") + 1])
        selfie = selfie[selfie.find("]") + 1 :]
    if len(chars_selfie) > maxchars:
        raise Exception("Very long SELFIES produced. Value =", chars_selfie)
    if len(chars_selfie) < maxchars:
        chars_selfie += ["[nop]"] * (maxchars - len(chars_selfie))
    return chars_selfie


def count_selfie_chars(selfie):
    chars_selfie = []  # A list of all SELFIE sybols from string selfie
    while selfie != "":
        chars_selfie.append(selfie[selfie.find("[") : selfie.find("]") + 1])
        selfie = selfie[selfie.find("]") + 1 :]
    return len(chars_selfie)


def sc2depictions(chromosone, root_name="output", lot=0):
    mol_structure = sc2mol_structure(chromosome, lot=lot)
    mol2pdb(mol_structure, "{0}.pdb".format(root_name))
    mol2xyz(mol_structure, "{0}.xyz".format(root_name))
    Draw.MolToFile(mol_structure, "{0}.png".format(root_name))
    logger.info("Generated depictions with root name {0}".format(root_name))


def mol_structure2depictions(mol_structure, root_name="output"):
    mol2pdb(mol_structure, "{0}.pdb".format(root_name))
    mol2xyz(mol_structure, "{0}.xyz".format(root_name))
    Draw.MolToFile(mol_structure, "{0}.png".format(root_name))


def sc2smiles(chromosome):
    selfie = "".join(x for x in list(chromosome))
    smiles = timed_decoder(selfie)
    mol, smi_canon, check = sanitize_smiles(smiles)
    if check:
        return smi_canon
    else:
        return None


def sc2mol_structure(chromosome, lot=0):
    selfie = "".join(x for x in list(chromosome))
    smiles = timed_decoder(selfie)
    mol, smi_canon, check = sanitize_smiles(smiles)
    if not check:
        logger.exception("SMILES {0} cannot be sanitized".format(smiles))
    if lot == 0:
        return get_structure_ff(mol, n_confs=5)
    if lot == 1:
        exit()


def get_structure_ff(mol, n_confs):
    Chem.SanitizeMol(mol)
    mol = Chem.AddHs(mol)
    mol_structure = Chem.Mol(mol)
    coordinates_added = False
    if not coordinates_added:
        try:
            conformer_ids = AllChem.EmbedMultipleConfs(
                mol,
                numConfs=n_confs,
                useExpTorsionAnglePrefs=True,
                useBasicKnowledge=True,
                pruneRmsThresh=1.5,
                enforceChirality=True,
            )
        except:
            logger.warning("Method 1 failed to generate conformations.")
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
            logger.warning("Method 2 failed to generate conformations.")
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
                pruneRmsThresh=1.5,
                ignoreSmoothingFailures=True,
            )
        except:
            logger.warning("Method 3 failed to generate conformations.")
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

    if Chem.rdForceFieldHelpers.MMFFHasAllMoleculeParams(mol):
        energies = AllChem.MMFFOptimizeMoleculeConfs(
            mol, maxIters=250, nonBondedThresh=15.0
        )
    elif Chem.rdForceFieldHelpers.UFFHasAllMoleculeParams(mol):
        energies = AllChem.UFFOptimizeMoleculeConfs(mol, maxIters=250, vdwThresh=15.0)
    else:
        logger.exception(
            "Could not generate structures using FF and rdkit typing. SMILES {0}".format(
                mol2smi(mol)
            )
        )

    energies_list = [e[1] for e in energies]
    min_e_index = energies_list.index(min(energies_list))
    mol_structure.AddConformer(mol.GetConformer(min_e_index))

    return mol_structure


def has_transition_metals(mol):
    if any([is_transition_metal(at) for at in mol.GetAtoms()]):
        return True
    else:
        return False


def is_transition_metal(at):
    n = at.GetAtomicNum()
    return (n >= 22 and n <= 29) or (n >= 40 and n <= 47) or (n >= 72 and n <= 79)


def set_dative_bonds(mol, fromAtoms=(7, 8, 15, 16)):
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
    problems = Chem.DetectChemistryProblems(mol)
    if len(problems) >= 1:
        for problem in problems:
            logger.warning(problem.GetType())
            logger.warning(problem.Message())
