import logging
import numpy as np
import os
from copy import deepcopy
from AaronTools.geometry import Geometry
from AaronTools.substituent import Substituent

logger = logging.getLogger(__name__)

queue_env_var = os.getenv("QUEUE_TYPE")
COORD_THRESHOLD = 0.2
USER = os.getenv("USER")
if queue_env_var is not None:
    QUEUE_TYPE = queue_env_var.upper()
else:
    os.environ["QUEUE_TYPE"] = "NOQUEUE"


class Hashable_Geometry(Geometry):
    """This is a modified version of the Geometry class in AaronTools.py, which is identical but hashable."""

    def __init__(self, *args, **kw):
        object.__setattr__(self, "_hashed", False)
        super().__init__(*args, **kw)

    def __repr__(self):
        """string representation"""
        s = ""
        for a in self:
            s += a.__repr__() + "\n"
        return s

    def __eq__(self, other):
        """
        two geometries equal if:
            same number of atoms
            same numbers of elements
            coordinates of atoms similar
        """
        if id(self) == id(other):
            return True
        if len(self.atoms) != len(other.atoms):
            return False

        self_eles = [atom.element for atom in self.atoms]
        other_eles = [atom.element for atom in other.atoms]
        self_counts = {ele: self_eles.count(ele) for ele in set(self_eles)}
        other_counts = {ele: other_eles.count(ele) for ele in set(other_eles)}
        if self_counts != other_counts:
            return False

        rmsd = self.RMSD(other, sort=False)
        return rmsd < COORD_THRESHOLD

    def __setattr__(self, attr, val):
        if (
            (attr == "_hashed" and val)
            or (attr != "_hashed" and attr.startswith("_"))
            or not self._hashed
        ):
            object.__setattr__(self, attr, val)
        else:
            object.__setattr__(self, attr, val)

    def __delattr__(self, attr):
        if not self._hashed:
            object.__del__(self, attr)
        else:
            object.__del__(self, attr)

    def __hash__(self):
        coords = self.coords
        coords -= self.COM()
        _, ax = self.get_principle_axes()
        coords = np.dot(coords, ax)

        t = []
        for atom, coord in zip(self.atoms, coords):
            t.append(
                (int(atom.get_neighbor_id()), tuple([int(x * 1e3) for x in coord]))
            )
            if not isinstance(atom.coords, tuple):
                atom.coords = tuple(atom.coords)
            if not isinstance(atom.connected, frozenset):
                atom.connected = frozenset(atom.connected)
            atom._hashed = True

        if not isinstance(self.atoms, tuple):
            self.atoms = tuple(self.atoms)
        self._hashed = True

        return hash(tuple(t))


def chromosome_to_xyz():
    """Wrapper function for simplicity."""

    def sc2xyz(chromosome):
        """Generate a XYZ string from a list particular geometry objects. To be customized."""
        h_positions = "19-20"
        scaffold = deepcopy(chromosome[0])
        target_list = []
        for gene in chromosome[1:]:
            if gene is not None:
                deepgene = deepcopy(Substituent(gene))
                target_list.append(deepgene)
            if gene is None:
                target_list.append(None)
        h_positions = scaffold.find("H", h_positions)
        assert len(h_positions) >= len(target_list)
        for i, j in enumerate(target_list):
            if j is not None:
                scaffold.substitute(j, h_positions[i], minimize=True)
        geom = Hashable_Geometry(scaffold)
        geom.minimize()
        return geom

    return sc2xyz


def gl2geom(chromosome, h_positions="19-20"):
    """Check if a chromosome (list of geometries) can lead to a valid structure."""
    logger.trace("Checking chromosome using scaffold:\n{0}".format(chromosome[0]))
    scaffold = deepcopy(chromosome[0])
    target_list = []
    for gene in chromosome[1:]:
        if gene is not None:
            deepgene = deepcopy(Substituent(gene))
            target_list.append(deepgene)
        if gene is None:
            target_list.append(None)
    h_positions = scaffold.find("H", h_positions)

    assert len(h_positions) >= len(target_list)
    try:
        for i, j in enumerate(target_list):
            if j is not None:
                scaffold.substitute(j, h_positions[i], minimize=True)
        geom = Hashable_Geometry(scaffold)
        geom.minimize()
        ok = True
    except Exception as m:
        logger.debug(m)
        geom = None
        ok = False
    logger.trace("Final geometry from scaffold:\n{0}".format(geom))
    return (ok, geom)


def geom2bond(geom, a1, a2):
    a1 = geom.find(a1)[0]
    a2 = geom.find(a2)[0]
    return a1.dist(a2)


def geom2angle(geom, a1, a2, a3):
    a1 = geom.find(a1)[0]
    a2 = geom.find(a2)[0]
    a3 = geom.find(a3)[0]
    return geom.angle(a1, a2, a3)


def geom2dihedral(geom, a1, a2, a3, a4):
    a1 = geom.find(a1)[0]
    a2 = geom.find(a2)[0]
    a3 = geom.find(a3)[0]
    a4 = geom.find(a4)[0]
    return geom.dihedral(a1, a2, a3, a4)


def geom2sub_sterimol(geom, pos, parameter):
    try:
        sub = geom.find_substituent(pos)
        val = sub.sterimol(parameter=parameter, return_vector=False, radii="bondi")
    except Exception as m:
        logger.debug(m)
        logger.warning("Could not evaluate Sterimol, perhaps there is no substituent.")
        logger.debug("Geometry :\n{0}".format(geom))
        val = 1.0
    return val
