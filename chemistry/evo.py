import numpy as np
from selfies import decoder, encoder, split_selfies
from rdkit import Chem
from rdkit.Chem import MolToSmiles as mol2smi
from rdkit.Chem import MolFromSmiles as smi2mol


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
       mol = smi2mol(smiles, sanitize=True)
       smi_canon = mol2smi(mol, isomericSmiles=False, canonical=True)
       return (mol, smi_canon, True)
    except:
       return (None, None, False)


def sanitize_multiple_smiles(smiles_list):
    sanitized_smiles = []
    for smi in smiles_list:
        smi_converted = sanitize_smiles(smi)
        sanitized_smiles.append(smi_converted[1])
        if smi_converted[2] == False or smi_converted[1] == "":
            raise Exception("Invalid SMILE encountered. Value =", smi)
    return sanitized_smiles


def encode_smiles(smiles_list):
    selfies_list = []
    for smi in smiles_list:
        selfie = encoder(smi)
        selfies_list.append(selfie)
    return selfies_list


def get_selfie_chars(selfie, maxchars):
    """Obtain a list of all selfie characters in string selfie
    
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
        raise Exception("Very long SELFIES produced from SMILES. Value =", selfie)
    if len(chars_selfie) < maxchars:
        chars_selfie += ['[nop]'] * (maxchars - len(chars_selfie))
    return chars_selfie



