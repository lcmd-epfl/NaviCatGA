import logging
import os
from glob import glob
from AaronTools.geometry import Geometry
from AaronTools.fileIO import FileReader
from AaronTools.substituent import Substituent
from navicatGA.wrappers_xyz import gl2geom

logger = logging.getLogger(__name__)

h_positions = "19-20"


def get_starting_xyz_from_path(path="scaffolds"):
    reference = False
    xyz_list = []
    dictionary_directory = os.path.join(path, "*.xyz")
    for i in glob(dictionary_directory):
        geom = Geometry(FileReader(i))
        if reference:
            geom.RMSD(ref_geom, align=True, heavy_only=True, sort=True)
        else:
            ref_geom = geom
            reference = True
        xyz_list.append(geom)
    return xyz_list


def get_starting_xyz_from_file(file_list):
    reference = False
    xyz_list = []
    for i in file_list:
        geom = Geometry(FileReader(i))
        if reference:
            geom.RMSD(ref_geom, align=True, heavy_only=True, sort=True)
        else:
            ref_geom = geom
            reference = True
        xyz_list.append(geom)
    assert len(file_list) == len(xyz_list)
    return xyz_list


def get_starting_xyz_from_smi(smiles_list):
    reference = False
    xyz_list = []
    for i in smiles_list:
        geom = Geometry.from_string(i)
        if reference:
            geom.RMSD(ref_geom, align=True, heavy_only=True, sort=True)
        else:
            ref_geom = geom
            reference = True
        xyz_list.append(geom)
    assert len(smiles_list) == len(xyz_list)
    return xyz_list


def get_default_dictionary():
    dictionary = []
    dir1 = os.path.join(Substituent.AARON_LIBS, "*.xyz")
    dir2 = os.path.join(Substituent.BUILTIN, "*.xyz")
    for i in glob(dir1) + glob(dir2):
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
    logger.trace("Checking chromosome using scaffold:\n{0}".format(chromosome[0]))
    return gl2geom(chromosome, h_positions)[0]


def pad_xyz_list(xyz, maxchars):
    """Pads chromosome with empty elements."""
    xyz_list = [xyz]
    if len(xyz_list) > maxchars:
        logger.warning("Exceedingly long list produced. Will be truncated.")
        xyz_list = xyz_list[0 : maxchars - 1]
    if len(xyz_list) < maxchars:
        xyz_list += [None] * (maxchars - len(xyz_list))
    return xyz_list


def write_chromosome(outname, chromosome):
    ok, geom = gl2geom(chromosome, h_positions)
    if ok:
        geom.write(outname)
    else:
        logger.warning("Could not write {0} as {1}.".format(chromosome, outname))
