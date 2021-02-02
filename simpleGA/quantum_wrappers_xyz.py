import logging
import numpy as np
from pyscf import gto, dft, semiempirical
from simpleGA.wrappers_xyz import gl2geom

logger = logging.getLogger(__name__)


def gl2gap(chromosome, lot=0):
    geom = gl2geom(chromosome)[1]
    try:
        pyscfmol, mf = geom2pyscf(geom, lot=lot)
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
            "E(LUMO)-E(HOMO) could not be evaluated for chromosome with geometry:\n{0}".format(
                geom
            )
        )
        logger.debug(m)
        e_homo = 1e6
        e_lumo = 0
    return e_lumo - e_homo


def gl2ehomo(chromosome, lot=0):
    geom = gl2geom(chromosome)[1]
    try:
        pyscfmol, mf = geom2pyscf(geom, lot=lot)
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
            "E(HOMO) could not be evaluated for chromosome with geometry:\n{0}".format(
                geom
            )
        )
        logger.debug(m)
        e_homo = -1e6
    return e_homo


def gl2elumo(chromosome, lot=0):
    geom = gl2geom(chromosome)[1]
    try:
        pyscfmol, mf = geom2pyscf(geom, lot=lot)
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
            "E(LUMO) could not be evaluated for chromosome with geometry:\n{0}".format(
                geom
            )
        )
        logger.debug(m)
    return e_lumo


def geom2pyscf(geom, lot=0):
    pyscfmol = gto.Mole()
    pyscfmol.atom = ""
    pyscfmol.charge = 0
    pyscfmol.spin = 0
    for atom in geom:
        symbol = atom.element
        p = atom.coords
        pyscfmol.atom += """{0} {1} {2} {3}\n""".format(symbol, p[0], p[1], p[2])
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
