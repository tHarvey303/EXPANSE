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
fit_photometry = "bin"
binmap_type = "voronoi"
psf_type = "webbpsf"
test_ids = ["13892", "1076", "4823", "14747", "13892", "13665", "12443", "9196", "6764", "5671"]

bands = [
    "F090W",
    "F115W",
    "F150W",
    "F162M",
    "F182M",
    "F200W",
    "F210M",
    "F250M",
    "F277W",
    "F300M",
    "F335M",
    "F356W",
    "F410M",
    "F444W",
]

try:
    n_jobs = int(sys.argv[1])
except:
    n_jobs = 6


galaxies = ResolvedGalaxy.init_all_field_from_h5(
    field,
    galaxies_dir,
    save_out=True,
    filter_ids=test_ids,
)

for galaxy in galaxies:
    # Hide ACS bands
    original_bands = copy.copy(galaxy.bands)
    galaxy.bands = copy.copy(bands)
    galaxy.use_psf_type = psf_type

    if (
        psf_type in galaxy.photometry_table.keys()
        and f"voronoi_{psf_type}" in galaxy.photometry_table[psf_type].keys()
    ):
        print(f"Skipping {galaxy.galaxy_id}")
        continue

    galaxy.get_webbpsf(
        PATH_LW_ENERGY="/nvme/scratch/work/tharvey/EXPANSE/psfs/Encircled_Energy_LW_ETCv2.txt",
        PATH_SW_ENERGY="/nvme/scratch/work/tharvey/EXPANSE/psfs/Encircled_Energy_SW_ETCv2.txt",
        overwrite=True,
        oversample=4,
    )
    galaxy.convolve_with_psf(psf_type, use_unmatched_data=True)

    galaxy.pixedfit_processing(gal_region_use="detection", override_psf_type=psf_type)
    # pixedfit binning
    galaxy.pixedfit_binning(name_out=f"pixedfit_{psf_type}", redc_chi2_limit=5.0, save_out=True)

    galaxy.pixedfit_processing(gal_region_use="detection", override_psf_type=psf_type)
    # pixedfit no min binning
    galaxy.pixedfit_binning(
        save_out=True,
        SNR_reqs=4,
        Dmin_bin=1,
        redc_chi2_limit=100.0,
        del_r=1.0,
        overwrite=False,
        name_out=f"pixedfit_nomin_{psf_type}",
    )

    # voronoi binning
    galaxy.voronoi_binning(
        SNR_reqs=7,
        galaxy_region="detection",
        overwrite=True,
        use_only_widebands=False,
        plot=True,
        quiet=False,
        ref_band="combined_average",
        override_psf_type=psf_type,
        name_out=f"voronoi_{psf_type}",
    )

    # give bands back for SED fitting
    # galaxy.bands = original_bands

    # Measure flux in bins
    galaxy.measure_flux_in_bins(f"pixedfit_{psf_type}", override_psf_type=psf_type)
    galaxy.measure_flux_in_bins(f"pixedfit_nomin_{psf_type}", override_psf_type=psf_type)
    galaxy.measure_flux_in_bins(f"voronoi_{psf_type}", override_psf_type=psf_type)


multiple_galaxies = ResolvedGalaxies(galaxies)

# Setup Bagpipes model

from EXPANSE.bagpipes.pipes_models import (
    create_dicts,
    resolved_dict_cnst,
)

num = len(galaxies)

# Run the bagpipes fitting
for binmap_type in ["pixedfit", "pixedfit_nomin", "voronoi"]:
    binmap_type = f"{binmap_type}_{psf_type}"

    override_meta_resolved = {
        "use_bpass": True,
        "name_append": f"_{binmap_type}_webbpsf",
    }
    resolved_dicts_cnst = create_dicts(
        resolved_dict_cnst, num=num, override_meta=override_meta_resolved
    )


    for dicts in [
        resolved_dicts_cnst,
    ]:
        multiple_galaxies.run_bagpipes_parallel(
            dicts,
            n_jobs=n_jobs,
            fit_photometry=fit_photometry,
            run_dir=run_dir,
            load_only=load_only,
            overwrite=False,
            dont_skip=True,
            override_binmap_type=binmap_type,
            override_psf_type=psf_type,
        )
