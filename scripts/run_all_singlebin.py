import ast
import contextlib

# can write astropy to h5
import copy
import glob
import os
import shutil
import sys
import traceback
import types
import warnings
from io import BytesIO
from pathlib import Path

import astropy.units as u
import h5py as h5
import matplotlib.pyplot as plt
import numpy as np
from astropy.cosmology import FlatLambdaCDM
from astropy.table import Column, QTable, Table, vstack
from joblib import Parallel, delayed, parallel_config
from tqdm import tqdm


from EXPANSE import (
    ResolvedGalaxy,
    MockResolvedGalaxy,
    run_bagpipes_wrapper,
    ResolvedGalaxies,
)
from EXPANSE.bagpipes.pipes_models import *

save_out = True  # Set to False if you don't want to save the output to h5

try:
    from mpi4py import MPI

    rank = MPI.COMM_WORLD.Get_rank()
    size = MPI.COMM_WORLD.Get_size()
    from mpi4py.futures import MPIPoolExecutor

    if size > 1:
        print("Running with mpirun/mpiexec detected.")
        save_out = False  # Avoids locking issues

except ImportError:
    rank = 0
    size = 1

cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

# Get path of this file
file_path = os.path.abspath(__file__)
db_dir = ""

if os.path.exists("/.singularity.d/Singularity"):
    computer = "singularity"
elif "nvme" in file_path:
    computer = "morgan"
elif "Users" in file_path:
    computer = "mac"
else:
    computer = "unknown"


if computer == "morgan":
    galaxies_dir = "/nvme/scratch/work/tharvey/EXPANSE/galaxies/"
    run_dir = "/nvme/scratch/work/tharvey/EXPANSE/pipes/"
elif computer == "singularity":
    galaxies_dir = "/mnt/galaxies/"
    run_dir = "/mnt/pipes/"

field = "JOF_psfmatched"
load_only = False
fit_photometry = "TOTAL_BIN"


galaxies_dir += "singlebin/"

try:
    n_jobs = int(sys.argv[1])
except:
    n_jobs = 6


galaxies = ResolvedGalaxy.init_all_field_from_h5(
    field, galaxies_dir, save_out=False
)

multiple_galaxies = ResolvedGalaxies(galaxies)

num = len(multiple_galaxies)


from EXPANSE.bagpipes.pipes_models import (
    continuity_dict,
    continuity_bursty_dict,
    create_dicts,
    delayed_dict,
    dpl_dict,
    lognorm_dict,
    cnst_dict,
    resolved_dict_cnst,
    resolved_dict_bursty,
)

override_meta = {
    "redshift": "eazy",
    "redshift_sigma": "eazy",
    "use_bpass": True,
}
delayed_dicts = create_dicts(
    delayed_dict, override_meta=override_meta, num=num
)
continuity_dicts = create_dicts(
    continuity_dict, override_meta=override_meta, num=num
)

continuity_bursty_dicts = create_dicts(
    continuity_bursty_dict, override_meta=override_meta, num=num
)

cnst_dicts = create_dicts(cnst_dict, override_meta=override_meta, num=num)


dpl_dicts = create_dicts(dpl_dict, override_meta=override_meta, num=num)
lognorm_dicts = create_dicts(
    lognorm_dict, override_meta=override_meta, num=num
)

override_meta_resolved = {
    "use_bpass": True,
}
resolved_dicts_cnst = create_dicts(
    resolved_dict_cnst, num=num, override_meta=override_meta_resolved
)

resolved_dicts_bursty = create_dicts(
    resolved_dict_bursty,
    num=num,
    override_meta=override_meta_resolved,
)


for dicts in [
    delayed_dicts,
    continuity_dicts,
    continuity_bursty_dicts,
    cnst_dicts,
    dpl_dicts,
    lognorm_dicts,
    resolved_dicts_cnst,
    resolved_dicts_bursty,
]:
    multiple_galaxies.run_bagpipes_parallel(
        dicts,
        n_jobs=n_jobs,
        fit_photometry=fit_photometry,
        run_dir=run_dir,
        load_only=load_only,
    )


# # -------------------------------------------------------------------------------------

# Do Resolved_cnst and resolved_bursty again for pixedfit_nomin
binmap_type = "pixedfit_nomin"

print("Running pixedfit_nomin binned SED fits")

override_meta_resolved = {
    "use_bpass": True,
    "run_name": "CNST_SFH_RESOLVED_NOMIN",
}
resolved_dicts_cnst = create_dicts(
    resolved_dict_cnst,
    num=num,
    override_meta=override_meta_resolved,
)

override_meta_resolved = {
    "use_bpass": True,
    "run_name": "BURSTY_SFH_RESOLVED_NOMIN",
}

resolved_dicts_bursty = create_dicts(
    resolved_dict_bursty,
    num=num,
    override_meta=override_meta_resolved,
)

for dicts in [
    resolved_dicts_cnst,
    resolved_dicts_bursty,
]:
    galaxies.run_bagpipes_parallel(
        dicts,
        n_jobs=n_jobs,
        fit_photometry=fit_photometry,
        run_dir=run_dir,
        load_only=load_only,
        override_binmap_type=binmap_type,
    )

# # -------------------------------------------------------------------------------------
# Do Resolved_cnst and resolved_bursty again for voronoi

binmap_type = "voronoi"

print("Running voronoi binned SED fits")

override_meta_resolved = {
    "use_bpass": True,
    "run_name": "CNST_SFH_RESOLVED_VORONOI",
}
resolved_dicts_cnst = create_dicts(
    resolved_dict_cnst,
    num=num,
    override_meta=override_meta_resolved,
)

override_meta_resolved = {
    "use_bpass": True,
    "run_name": "BURSTY_SFH_RESOLVED_VORONOI",
}

resolved_dicts_bursty = create_dicts(
    resolved_dict_bursty,
    num=num,
    override_meta=override_meta_resolved,
)

for dicts in [
    resolved_dicts_cnst,
    resolved_dicts_bursty,
]:
    galaxies.run_bagpipes_parallel(
        dicts,
        n_jobs=n_jobs,
        fit_photometry=fit_photometry,
        run_dir=run_dir,
        load_only=load_only,
        override_binmap_type=binmap_type,
    )
