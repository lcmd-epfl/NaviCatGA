import logging
import numpy as np
from pyscf import gto, dft, semiempirical
from simpleGA.wrappers_xyz import gl2geom
from AaronTools.atoms import Atom
from AaronTools.geometry import Geometry


logger = logging.getLogger(__name__)


def gl2gap(chromosome, lot=0):
    ok, geom = gl2geom(chromosome)
    if not ok:
        logger.debug("No molecule generated from genes.")
    return geom2gap(geom, lot)


def geom2gap(geom, lot=0):
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
        logger.debug(m)
        logger.warning("E(LUMO)-E(HOMO) could not be evaluated for chromosome.")
        logger.debug("Geometry :\n{0}".format(geom))
        e_homo = 1e6
        e_lumo = 0
    return e_lumo - e_homo


def gl2ehl(chromosome, lot=0):
    ok, geom = gl2geom(chromosome)
    if not ok:
        logger.debug("No molecule generated from genes.")
    return geom2ehl(geom, lot)


def geom2ehl(geom, lot=0):
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
        logger.debug(m)
        logger.warning("E(HOMO) and E(LUMO) could not be evaluated for chromosome.")
        logger.debug("Geometry :\n{0}".format(geom))
        e_homo = 1e6
        e_lumo = 0
    return [e_homo, e_lumo]


def geom2ehomo(geom, lot=0):
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
        logger.debug(m)
        logger.warning("E(HOMO) could not be evaluated for chromosome.")
        logger.debug("Geometry :\n{0}".format(geom))
        e_homo = -1e6
    return e_homo


def geom2elumo(geom, lot=0):
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
        logger.debug(m)
        logger.warning("E(LUMO) could not be evaluated for chromosome.")
        logger.debug("Geometry :\n{0}".format(geom))
    return e_lumo


def gl2elumo(chromosome, lot=0):
    ok, geom = gl2geom(chromosome)
    if not ok:
        logger.debug("No molecule generated from genes.")
    return geom2elumo(geom, lot)


def gl2opt(chromosome, lot=0):
    ok, geom = gl2geom(chromosome)
    if not ok:
        logger.warning("No molecule generated from genes.")
    return geom2opt(geom, lot)


def gl2ehomo(chromosome, lot=0):
    ok, geom = gl2geom(chromosome)
    if not ok:
        logger.debug("No molecule generated from genes.")
    return geom2ehomo(geom, lot)


def geom2opt(geom, lot=0):
    try:
        pyscfmol, mf = geom2pyscf(geom, lot=lot)
        pyscfmol, mf = opt(pyscfmol, mf)
        geom = pyscf2geom(pyscfmol, geom)
    except Exception as m:
        logger.debug(m)
        logger.warning("Could not optimize geometry.")
        logger.debug("Geometry :\n{0}".format(geom))
    return geom


def opt(pyscfmol, mf):
    from pyscf.geomopt.berny_solver import optimize

    conv_params = {  # They are default settings
        "gradientmax": 3e-4,  # Eh/Angstrom
        "gradientrms": 3e-4,  # Eh/Angstrom
        "stepmax": 2e-3,  # Angstrom
        "steprms": 2e-3,  # Angstrom
    }
    opt_pyscfmol = optimize(mf, **conv_params)
    return opt_pyscfmol, mf


def pyscf2geom(pyscfmol, geom):
    atom_list = []
    for i in range(pyscfmol.natm):
        s = pyscfmol.atom_symbol(i)
        r = pyscfmol.atom_coord(i)
        atom = Atom(element=s, coords=r, name=str(i))
        atom_list.append(atom)
    geom.update_geometry(atom_list)
    geom.detect_substituents()
    return geom


def geom2pyscf(geom, lot=0):
    pyscfmol = gto.Mole()
    pyscfmol.atom = ""
    nelectron = pyscfmol.atom_charges().sum()
    pyscfmol.spin = 0
    if nelectron % 2 == 0:
        pyscfmol.charge = 0
    else:
        pyscfmol.charge = -1
    for atom in geom:
        symbol = atom.element
        p = atom.coords
        pyscfmol.atom += """{0} {1} {2} {3}\n""".format(symbol, p[0], p[1], p[2])
    if lot == 0:
        pyscfmol.build()
        mf = semiempirical.RMINDO3(pyscfmol)
        mf.verbose = 1
        mf.conv_tol = 1e-6
        mf = mf.run()
    if lot == 1:
        pyscfmol.basis = "pcseg0"
        pyscfmol.build()
        mf = dft.ROKS(pyscfmol)
        mf.verbose = 1
        mf.xc = "pbe,pbe"
        mf.conv_tol = 1e-6
        mf = mf.density_fit().run()
    if lot == 2:
        pyscfmol.basis = "def2svp"
        pyscfmol.build()
        mf = dft.ROKS(pyscfmol)
        mf.verbose = 1
        mf.xc = "b97d"
        mf.conv_tol = 1e-7
        mf = mf.density_fit().run()
    return pyscfmol, mf
