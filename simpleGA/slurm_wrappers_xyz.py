import logging
import numpy as np
from AaronTools.geometry import Geometry
from AaronTools.fileIO import FileReader, read_types
from AaronTools.substituent import Substituent
from AaronTools.theory import Theory, OptimizationJob
from AaronTools.job_control import SubmitProcess
from simpleGA.wrappers_xyz import gl2geom

logger = logging.getLogger(__name__)


def geom2optgeom_slurm(chromosome, lot=0, idtag=0):
    ok, geom = gl2geom(chromosome)
    if ok:
        if lot == 0:
            lot_0 = Theory(method="PM6", processors=1, job_type=OptimizationJob())
            geom.write(outfile="opt_{0}.com".format(idtag), theory=lot_0)
            opt_job = SubmitProcess(
                fname="opt_{0}.com".format(idtag), walltime=1, processors=1, memory=8
            )
            opt_job.submit(wait=True)
            opt_geom = Geometry("opt_{0}.log".format(idtag))
        return opt_geom
    else:
        return None


def gl2dihedral_slurm(chromosome, a1, a2, a3, a4):
    geom = geom2optgeom_slurm(chromosome)
    a1 = geom.find(a1)[0]
    a2 = geom.find(a2)[0]
    a3 = geom.find(a3)[0]
    a4 = geom.find(a4)[0]
    val = geom.dihedral(a1, a2, a3, a4)
    val *= 180 / np.pi
    return val


def gl2bond_slurm(chromosome, a1, a2):
    geom = geom2optgeom_slurm(chromosome)
    a1 = geom.find(a1)[0]
    a2 = geom.find(a2)[0]
    val = a1.dist(a2)
    return val


def gl2angle_slurm(chromosome, a1, a2, a3):
    geom = geom2optgeom_slurm(chromosome)
    a1 = geom.find(a1)[0]
    a2 = geom.find(a2)[0]
    a3 = geom.find(a3)[0]
    val = geom.angle(a2, a2, a3)
    val *= 180 / np.pi
    return val