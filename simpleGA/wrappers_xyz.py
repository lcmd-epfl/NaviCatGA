import logging
import numpy as np
from copy import deepcopy
from AaronTools.geometry import Geometry
from AaronTools.fileIO import FileReader, read_types
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
