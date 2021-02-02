import logging
import numpy as np
from copy import deepcopy
from AaronTools.geometry import Geometry
from AaronTools.fileIO import FileReader, read_types
from AaronTools.substituent import Substituent
from AaronTools.theory import Theory, OptimizationJob

logger = logging.getLogger(__name__)


def gl2geom(chromosome):
    """Check if a list of Geometries can lead to a valid structure."""
    logger.debug("Checking chromosome using scaffold:\n{0}".format(chromosome[0]))
    scaffold = deepcopy(chromosome[0])
    target_list = []
    for gene in chromosome[1:]:
        if gene is not None:
            target_list.append(Substituent(gene))
        if gene is None:
            target_list.append(None)
    h_positions = scaffold.find("H")
    try:
        assert len(h_positions) >= len(target_list)
        for i, j in enumerate(target_list):
            if j is not None:
                scaffold.substitute(j, h_positions[i], minimize=True)
        scaffold.refresh_connected()
        geom = Geometry(scaffold)
        ok = True
    except Exception as m:
        logger.debug(m)
        geom = None
        ok = False
    return (ok, geom)
