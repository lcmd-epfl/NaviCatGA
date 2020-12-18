import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import MolFromSmiles as smi2mol
from rdkit.Chem import MolToSmiles as mol2smi
from rdkit.Chem import Descriptors
from selfies import decoder
from chemistry.evo import sanitize_smiles


def sc2logp(chromosome):
    selfie = "".join(x for x in list(chromosome))
    smiles = decoder(selfie)
    mol, smiles_canon, done = sanitize_smiles(smiles)
    if smiles_canon == "" or not done:
        print(smiles)
        # raise Exception("Inadequete SELFIES was generated. Value =", selfie)
        return 0
    logp = Descriptors.MolLogP(mol)
    return logp


def sc2wt(chromosome):
    selfie = "".join(x for x in list(chromosome))
    smiles = decoder(selfie)
    mol, smiles_canon, done = sanitize_smiles(smiles)
    if smiles_canon == "" or not done:
        print(smiles)
        # raise Exception("Inadequete SELFIES was generated. Value =", selfie)
        return 0
    chgwt = np.round(Descriptors.HeavyAtomMolWt(mol))
    return chgwt

def sc2krr(chromosome):
    return 50


def fitness_function_selfies(function_number):

    if function_number == 1:

        return lambda chromosome: sc2logp(chromosome)

    if function_number == 2:

        return lambda chromosome: sc2wt(chromosome)
