import logging
from copy import deepcopy
from AaronTools.geometry import Geometry
from AaronTools.substituent import Substituent

logger = logging.getLogger(__name__)


def gl2geom(chromosome):
    """Check if a list of Geometries can lead to a valid structure."""
    logger.debug("Checking chromosome using scaffold:\n{0}".format(chromosome[0]))
    scaffold = deepcopy(chromosome[0])
    target_list = []
    for gene in chromosome[1:]:
        if gene is not None:
            deepgene = deepcopy(gene)
            target_list.append(Substituent(deepgene))
        if gene is None:
            target_list.append(None)
    h_positions = scaffold.find("H")[0::2] + scaffold.find("H")[1::2]
    assert len(h_positions) >= len(target_list)
    try:
        for i, j in enumerate(target_list):
            if j is not None:
                scaffold.substitute(j, h_positions[i], minimize=True)
        geom = Geometry(scaffold)
        ok = True
    except Exception as m:
        logger.debug(m)
        geom = None
        ok = False
    logger.debug("Final geometry from scaffold:\n{0}".format(geom))
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
        logger.warning(
            "Could not evaluate Sterimol, perhaps there is no substituent:\n{0}".format(
                geom
            )
        )
        val = 1.0
    return val
