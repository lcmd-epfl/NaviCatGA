import logging
from utils.timeout import handler, timer_alarm
from selfies import decoder, encoder
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.rdmolfiles import MolToSmiles as mol2smi
from rdkit.Chem.rdmolfiles import MolFromSmiles as smi2mol

logger = logging.getLogger(__name__)
lg = RDLogger.logger()
lg.setLevel(RDLogger.ERROR)
RDLogger.DisableLog("rdApp.*")


def sanitize_smiles(smiles):  # Problems with C1C=CC=CC=1[P-1]=[P-1][P-1] for instance
    """Return a canonical smile representation of smi.
    If there are metals, it will try to fix the bonds as dative.

    Parameters:
    smi (string) : smile string to be canonicalized

    Returns:
    mol (rdkit.Chem.rdchem.Mol) : rdkit mol object, None if exception caught.
    smi_canon (string)          : Canonicalized smile representation of smi, None if exception caught.
    conversion_successful (bool): True if no exception caught, False if exception caught.
    """
    return timed_sanitizer(smiles)


def sanitize_multiple_smiles(smiles_list):
    """Calls sanitize_smiles for every item in a list.

    Parameters:
    smiles_list (list) : list of smile strings to be sanitized.

    Returns:
    sanitized_smiles (list) : list of sanitized smile strings with None in errors.
    """
    sanitized_smiles = []
    for smi in smiles_list:
        smi_converted = sanitize_smiles(smi)
        sanitized_smiles.append(smi_converted[1])
        if not smi_converted[2]:
            logger.exception("Invalid SMILES encountered. Value =", smi)
    return sanitized_smiles


def encode_smiles_list(smiles_list):
    """Encode a list of smiles to a list of selfies using selfies.encoder."""
    selfies_list = []
    for smi in smiles_list:
        selfie = encoder(smi)
        selfies_list.append(selfie)
    return selfies_list


def timed_sanitizer(smiles):
    """Convert smiles string to rdkit.mol, call exception and return None if it takes more than 10 seconds to run. """
    timer_alarm(
        10, handler
    )  # If this does not finish within 90 seconds, exception pops
    try:
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
    except:
        logger.debug(
            "Smiles {0} probably became stuck in rdkit smi2mol.".format(smiles)
        )
        return (None, None, False)


def timed_decoder(selfie):
    """Decode a selfies string to smiles using selfies.decoder, call exception and return None if decoder takes more than 10 seconds to run. """
    timer_alarm(
        10, handler
    )  # If this does not finish within 90 seconds, exception pops
    try:
        selfie = selfie.replace("[nop]", "")
        smiles = decoder(selfie)
    except:
        smiles = None
    return smiles


def check_selfie_chars(chromosome):
    """Check if a list of selfies characters leads to a valid smiles string. Uses sanitize_smiles to check the smiles string from selfies.decoder.

    Parameters:
    chromosome (list) : list of selfie characters.

    Returns:
    True if the smiles string is deemed valid by sanitize_smiles, False otherwise.
    """
    selfie = "".join(x for x in list(chromosome))
    smiles = timed_decoder(selfie)
    logger.debug(
        "Checking SELFIE {0} which was decoded to SMILES {1}".format(selfie, smiles)
    )
    return sanitize_smiles(smiles)[2]


def get_selfie_chars(selfie, maxchars):
    """Obtain an ordered list of all selfie characters in string selfie
    padded to maxchars with [nop]s.

    Parameters:
    selfie (string) : A selfie string - representing a molecule.
    maxchars (int) : Maximum number of elements in the list.

    Example:
    >>> get_selfie_chars('[C][=C][C][=C][C][=C][Ring1][Branch1_1]')
    ['[C]', '[=C]', '[C]', '[=C]', '[C]', '[=C]', '[Ring1]', '[Branch1_1]']

    Returns:
    chars_selfie (list): list of selfie characters present in molecule selfie.
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
    """Count the number of selfie characters in a selfie string. Returns the number."""
    chars_selfie = []
    while selfie != "":
        chars_selfie.append(selfie[selfie.find("[") : selfie.find("]") + 1])
        selfie = selfie[selfie.find("]") + 1 :]
    return len(chars_selfie)


def get_structure_ff(mol, n_confs):
    """Generates a reasonable set of 3D structures
    using forcefields for a given rdkit.mol object.
    It will try several 3D generation approaches in rdkit.
    It will try to sample several conformations and get the minima.

    Parameters:
    mol (rdkit.mol) : An rdkit mol object.
    n_confs (int) : The number of conformations to sample.

    Returns:
    mol_structure (rdkit.mol) : The same rdkit mol with 3D coordinates.

    """
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
        energies_list = [e[1] for e in energies]
        min_e_index = energies_list.index(min(energies_list))
        mol_structure.AddConformer(mol.GetConformer(min_e_index))
    elif Chem.rdForceFieldHelpers.UFFHasAllMoleculeParams(mol):
        energies = AllChem.UFFOptimizeMoleculeConfs(mol, maxIters=250, vdwThresh=15.0)
        energies_list = [e[1] for e in energies]
        min_e_index = energies_list.index(min(energies_list))
        mol_structure.AddConformer(mol.GetConformer(min_e_index))
    else:
        logger.warning(
            "Could not generate structures using FF and rdkit typing. SMILES {0}".format(
                mol2smi(mol)
            )
        )

    return mol_structure


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
    """Tries to identify and print to logger whatever was or is wrong with the chemistry of an rdkit.mol object."""
    problems = Chem.DetectChemistryProblems(mol)
    if len(problems) >= 1:
        for problem in problems:
            logger.warning(problem.GetType())
            logger.warning(problem.Message())
