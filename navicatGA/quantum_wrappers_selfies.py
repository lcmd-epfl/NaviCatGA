import logging
import numpy as np
from rdkit import Chem, RDLogger
from pyscf import gto, dft, semiempirical
from navicatGA.wrappers_selfies import sc2mol_structure
from navicatGA.chemistry_selfies import timed_decoder

logger = logging.getLogger(__name__)
lg = RDLogger.logger()
lg.setLevel(RDLogger.ERROR)
RDLogger.DisableLog("rdApp.*")


def sc2gap(chromosome, lot=0):
    try:
        mol_structure = sc2mol_structure(chromosome, lot=lot)
        pyscfmol, mf = mol_structure2pyscf(mol_structure, lot=lot)
        idx = np.argsort(mf.mo_energy)
        e_sort = mf.mo_energy[idx]
        occ_sort = mf.mo_occ[idx].astype(int)
        assert len(e_sort) == len(occ_sort)
        for i, occ in enumerate(occ_sort):
            if occ >= 1:
                continue
            if occ < 1:
                e_lumo = e_sort[i]
                e_homo = e_sort[i - 1]
                break
        assert e_lumo > e_homo
    except Exception as m:
        logger.warning(
            "E(LUMO)-E(HOMO) not be evaluated for chromosome. SMILES : {0}".format(
                timed_decoder("".join(x for x in list(chromosome)))
            )
        )
        logger.debug(m)
        e_homo = 1e6
        e_lumo = 0
    return e_lumo - e_homo


def sc2ehomo(chromosome, lot=0):
    try:
        mol_structure = sc2mol_structure(chromosome, lot=lot)
        pyscfmol, mf = mol_structure2pyscf(mol_structure, lot=lot)
        idx = np.argsort(mf.mo_energy)
        e_sort = mf.mo_energy[idx]
        occ_sort = mf.mo_occ[idx].astype(int)
        assert len(e_sort) == len(occ_sort)
        for i, occ in enumerate(occ_sort):
            if occ >= 1:
                continue
            if occ < 1:
                e_homo = e_sort[i - 1]
                break
        assert 0 > e_homo
    except Exception as m:
        logger.warning(
            "E(HOMO) not be evaluated for chromosome. SMILES : {0}".format(
                timed_decoder("".join(x for x in list(chromosome)))
            )
        )
        logger.debug(m)
        e_homo = -1e6
    return e_homo


def sc2elumo(chromosome, lot=0):
    try:
        mol_structure = sc2mol_structure(chromosome, lot=lot)
        pyscfmol, mf = mol_structure2pyscf(mol_structure, lot=lot)
        idx = np.argsort(mf.mo_energy)
        e_sort = mf.mo_energy[idx]
        occ_sort = mf.mo_occ[idx].astype(int)
        assert len(e_sort) == len(occ_sort)
        for i, occ in enumerate(occ_sort):
            if i >= 1:
                continue
            if i < 1:
                e_lumo = e_sort[i]
                break
        assert 0 < e_lumo
    except Exception as m:
        logger.warning(
            "E(LUMO) not be evaluated for chromosome. SMILES : {0}".format(
                timed_decoder("".join(x for x in list(chromosome)))
            )
        )
        logger.debug(m)
    return e_lumo


def mol_structure2pyscf(mol, lot=0):
    pyscfmol = gto.Mole()
    pyscfmol.atom = ""
    pyscfmol.charge = Chem.GetFormalCharge(mol)
    radical_electrons = [a.GetNumRadicalElectrons() for a in mol.GetAtoms()]
    pyscfmol.spin = sum(radical_electrons)
    symbols = [a.GetSymbol() for a in mol.GetAtoms()]
    for mol_structure in mol.GetConformers():
        for i, symbol in enumerate(symbols):
            p = mol_structure.GetAtomPosition(i)
            pyscfmol.atom += """{0} {1} {2} {3}\n""".format(symbol, p.x, p.y, p.z)
    if lot == 0:
        pyscfmol.build()
        mf = semiempirical.RMINDO3(pyscfmol)
        mf.verbose = 1
        mf.run(conv_tol=1e-6)  # UMINDO3 is an option as well
    if lot == 1:
        pyscfmol.basis = "MINAO"
        pyscfmol.build()
        mf = dft.ROKS(pyscfmol)
        mf.verbose = 1
        mf.xc = "pbe,pbe"
        mf = mf.density_fit().run(conv_tol=1e-6)
    if lot == 2:
        pyscfmol.basis = "pcseg0"
        pyscfmol.build()
        mf = dft.RKS(pyscfmol)
        mf.verbose = 1
        mf.xc = "b3lypg"
        mf = mf.density_fit().run(conv_tol=1e-7)
    return pyscfmol, mf
