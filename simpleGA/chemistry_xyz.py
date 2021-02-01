import logging
import numpy as np
import os
from glob import glob
from AaronTools.geometry import Geometry
from AaronTools.fileIO import FileReader, read_types
from AaronTools.substituent import Substituent

logger = logging.getLogger(__name__)


def get_starting_xyz_fromfile(files):
    reference = False
    xyz_list = []
    for i in files:
        geom = Geometry(FileReader(i))
        if reference:
            geom.RMSD(ref_geom, align=True, heavy_only=True, sort=True)
        else:
            ref_geom = geom
            reference = True
        xyz_list.append(geom)
    assert len(files) == len(xyz_list)
    return xyz_list


def get_starting_xyz_fromsmi(smiles):
    reference = False
    xyz_list = []
    for i in files:
        geom = Geometry.from_string(i)
        if reference:
            geom.RMSD(ref_geom, align=True, heavy_only=True, sort=True)
        else:
            ref_geom = geom
            reference = True
        xyz_list.append(geom)
    assert len(smiles) == len(xyz_list)


def get_default_dictionary():
    dictionary = []
    for i in glob(Substituent.AARON_LIBS) + glob(Substituent.BUILTIN):
        geom = Geometry(FileReader(i))
        dictionary.append(geom)
    return dictionary


def get_dictionary_from_path(path="dictionary"):
    dictionary = []
    dictionary_directory = os.path.join(path, "*.xyz")
    for i in glob(dictionary_directory):
        geom = Geometry(FileReader(i))
        dictionary.append(geom)
    return dictionary


def check_xyz(chromosome):
    """Check if a list of Geometries can lead to a valid structure."""
    logger.debug("Checking chromosome using scaffold {0}".format(scaffold))
    try:
        scaffold = chromosome[0]
        target_list = [Substituent(i) for i in chromosome[1:]]
        h_positions = scaffold.find["H"]
        assert len(h_positions) <= len(target_list)
        for i, atom in enumerate(h_positions):
            scaffold.substitute(target_list[i], atom, minimize=True)
        scaffold.refresh_connected()
        ok = True
    except Exception as m:
        logger.debug(m)
        ok = False
    return ok


def pad_xyz_list(xyz, maxchars):
    """Pads chromosome with empty elements."""
    xyz_list = [xyz]
    if len(xyz_list) > maxchars:
        logger.warning("Exceedingly long list produced. Will be truncated.")
        xyz_list = xyz_list[0 : maxchars - 1]
    if len(xyz_list) < maxchars:
        xyz_list += [Geometry.from_string("[H]", form="smiles")] * (
            maxchars - len(xyz_list)
        )
    return xyz_list
