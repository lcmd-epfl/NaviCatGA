import logging
import os
import numpy as np
from glob import glob
from copy import deepcopy
from AaronTools.fileIO import FileReader
from AaronTools.geometry import Geometry
from AaronTools.substituent import Substituent

logger = logging.getLogger(__name__)


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
        return rmsd < 0.2

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


def random_merge_xyz():
    """Default chromosome manipulator: randomly generates an XYZ structure from fragments."""

    def merge(chromosome):
        """Generates an XYZ geometry from fragments randomly."""
        chromosome = [i for i in chromosome if i]
        geom_list = np.empty(len(chromosome), dtype=object)
        target_list = np.empty(len(chromosome), dtype=object)
        try:
            for i, gene in enumerate(chromosome):
                geom_list[i] = deepcopy(Geometry(gene))
            for i, gene in enumerate(geom_list):
                target_list[i] = Substituent(sub=geom_list[i])
            av_h = np.zeros_like(geom_list, dtype=int)
            for i, geom in enumerate(geom_list):
                try:
                    nh = len(geom.find("H"))
                except Exception as m:
                    nh = 0
                av_h[i] = nh
            ok_h = np.where(av_h >= len(chromosome) - 1)[0].flatten()
            apt_id = np.random.choice(ok_h, size=1)[0]
            scaffold = geom_list[apt_id]
            target_list = np.delete(geom_list, apt_id, 0)
            h_pos = scaffold.find("H")
            np.random.shuffle(h_pos)
            for i, geom in enumerate(target_list):
                print(type(scaffold), type(geom), type(h_pos[i]))
                scaffold.substitute(geom, h_pos[i], minimize=True)
            mgeom = Hashable_Geometry(scaffold)
            mgeom.minimize()
        except Exception as m:
            print(f"Random merger of xyz structures failed: {m}")
        return mgeom

    return merge


def get_starting_xyz_from_path(path="scaffolds"):
    reference = False
    xyz_list = []
    alphabet_directory = os.path.join(path, "*.xyz")
    for i in glob(alphabet_directory):
        geom = Geometry(FileReader(i))
        if not reference:
            ref_geom = geom
            reference = True
        else:
            geom.RMSD(ref_geom, align=True, heavy_only=True, sort=True)
        xyz_list.append(geom)
    return xyz_list


def get_starting_xyz_from_file(file_list):
    reference = False
    xyz_list = []
    for i in file_list:
        geom = Geometry(FileReader(i))
        if not reference:
            ref_geom = geom
            reference = True
        else:
            geom.RMSD(ref_geom, align=True, heavy_only=True, sort=True)
        xyz_list.append(geom)
    assert len(file_list) == len(xyz_list)
    return xyz_list


def get_starting_xyz_from_smi(smiles_list):
    reference = False
    xyz_list = []
    for i in smiles_list:
        geom = Geometry.from_string(i)
        if not reference:
            ref_geom = geom
            reference = True
        else:
            geom.RMSD(ref_geom, align=True, heavy_only=True, sort=True)
        xyz_list.append(geom)
    assert len(smiles_list) == len(xyz_list)
    return xyz_list


def get_default_alphabet():
    alphabet = []
    dir1 = os.path.join(Substituent.AARON_LIBS, "*.xyz")
    dir2 = os.path.join(Substituent.BUILTIN, "*.xyz")
    for i in glob(dir1) + glob(dir2):
        geom = Geometry(FileReader(i))
        alphabet.append(geom)
    return alphabet


def get_alphabet_from_path(path="alphabet/"):
    alphabet = []
    alphabet_directory = os.path.join(path, "*.xyz")
    for i in glob(alphabet_directory):
        geom = Geometry(FileReader(i))
        alphabet.append(geom)
    return alphabet


def check_xyz(chromosome):
    """Check if a list of Geometries can lead to a valid structure."""
    logger.trace(f"Checking chromosome {chromosome}")
    return random_merge_xyz(chromosome)


def pad_xyz_list(xyz, maxchars):
    """Pads chromosome with empty elements."""
    xyz_list = [xyz]
    if len(xyz_list) > maxchars:
        logger.warning("Exceedingly long list produced. Will be truncated.")
        xyz_list = xyz_list[0 : maxchars - 1]
    if len(xyz_list) < maxchars:
        xyz_list += [None] * (maxchars - len(xyz_list))
    return xyz_list


def draw_xyz(geom, outname):
    geom.write(outname)
