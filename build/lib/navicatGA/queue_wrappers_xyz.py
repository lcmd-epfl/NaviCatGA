import logging
import subprocess
import os
from time import sleep
from AaronTools.geometry import Geometry
from AaronTools.theory import Theory, OptimizationJob
from AaronTools.job_control import SubmitProcess

logger = logging.getLogger(__name__)

queue_env_var = os.getenv("QUEUE_TYPE")
COORD_THRESHOLD = 0.2
USER = os.getenv("USER")
if queue_env_var is not None:
    QUEUE_TYPE = queue_env_var.upper()
else:
    os.environ["QUEUE_TYPE"] = "NOQUEUE"


class FixedSubmitProcess(SubmitProcess):
    """This is a lightly modified version of the SubmitProcess class in AaronTools.py."""

    def submit(self, wait=False, quiet=True):

        job_file = os.path.join(self.directory, self.name + ".job")

        opts = {
            "name": self.name,
            "walltime": self.walltime,
            "processors": self.processors,
            "memory": self.memory,
        }

        tm = self.template.render(**opts)

        with open(job_file, "w") as f:
            f.write(tm)

        if QUEUE_TYPE == "LSF":
            args = ["bsub", "<", job_file]
        elif QUEUE_TYPE == "SLURM":
            args = ["sbatch", "--wait", job_file]
        elif QUEUE_TYPE == "PBS":
            args = ["qsub", job_file]
        elif QUEUE_TYPE == "SGE":
            args = ["qsub", job_file]
        else:
            raise NotImplementedError(
                "%s queues not supported, only LSF, SLURM, PBS, and SGE" % QUEUE_TYPE
            )

        proc = subprocess.Popen(
            args, cwd=self.directory, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

        self.submit_out, self.submit_err = proc.communicate()

        if len(self.submit_err) != 0:
            raise RuntimeError(
                "error with submitting job %s: %s"
                % (self.name, self.submit_err.decode("utf-8"))
            )

        if not quiet:
            print(self.submit_out.decode("utf-8").strip())

        if wait is not False:
            if wait is True:
                wait_time = 5
            else:
                wait_time = abs(wait)

            while len(self.unfinished_jobs_in_dir(self.directory)) != 0:
                sleep(wait_time)

        return


def geom2opt(geom, lot=2, idtag=0):
    outfile = "opt_{0}.com".format(idtag)
    logfile = "opt_{0}.log".format(idtag)
    job_type = OptimizationJob()
    if lot == 0:
        functional = "PM6"
        basis = ""
    if lot == 1:
        functional = "PBEPBE"
        basis = "def2SVP"
    if lot == 2:
        functional = "B97D"
        basis = "def2SVP/W06"
    lot_theory = Theory(
        method=functional, basis=basis, processors=24, memory=100, job_type=job_type
    )
    geom.write(
        outfile=outfile,
        theory=lot_theory,
        GAUSSIAN_ROUTE={"DensityFit": ["Coulomb", "NonIterative"], "SCF": ["YQC"]},
    )
    job = FixedSubmitProcess(fname=outfile, walltime=30, processors=24, memory=101)

    job.submit(
        wait=False, quiet=True
    )  # this does not work due to typo in AaronTools SubmitProcess, thus the child class

    try:
        opt_geom = Geometry(logfile)
        geom.update_geometry(opt_geom)
        geom.detect_substituents()
    except Exception as m:
        logger.warning("Could not optimize molecule.")
        logger.debug(m)
    return geom
