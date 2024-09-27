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
from joblib import Parallel, delayed
from tqdm import tqdm


from EXPANSE import (
    ResolvedGalaxy,
    MockResolvedGalaxy,
    run_bagpipes_wrapper,
    MultipleResolvedGalaxy,
)
from EXPANSE.bagpipes.pipes_models import *


try:
    from mpi4py import MPI

    rank = MPI.COMM_WORLD.Get_rank()
    size = MPI.COMM_WORLD.Get_size()
    from mpi4py.futures import MPIPoolExecutor

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


resolved_galaxy_dir = "../galaxies/"

if __name__ == "__main__":
    if computer == "morgan":
        catalog_path_selected = "/nvme/scratch/work/tharvey/EXPANSE/catalogs/JOF_psfmatched_MASTER_Sel-F277W+F356W+F444W_v11_total_selected.fits"
        cat_selected = Table.read(catalog_path_selected)
        ids = cat_selected["NUMBER"]
        cutout_size = cat_selected["CUTOUT_SIZE"]
        ids = ids  # For testing
        overwrite = False
        h5_folder = resolved_galaxy_dir
    elif computer == "singularity":
        overwrite = False
        h5_folder = "/mnt/galaxies/"
        # Get all ids in the folder
        ids = [
            int(os.path.basename(f).split("_")[-1].split(".")[0])
            for f in glob.glob(f"{h5_folder}/*.h5")
            if "mock" not in f and "temp" not in f
        ]
        # remove
        cutout_size = None  # Not used when loading from h5

    # Should speed it up?

    cat = None

    initial_load = False
    n_jobs = 32
    # Set to True if you want to load the data from the catalogue
    just_bagpipes_parallel = True

    if not just_bagpipes_parallel:
        galaxies = ResolvedGalaxy.init(
            list(ids),
            "JOF_psfmatched",
            "v11",
            already_psf_matched=True,
            cutout_size=cutout_size,
            h5_folder=h5_folder,
        )

    num_of_bins = 0
    num_of_single_bin = 0
    if rank == 0 and initial_load:
        for posi, galaxy in enumerate(galaxies):
            # Add original imaging back
            # print('Adding original imaging.')

            cat = galaxy.add_original_data(
                cat=cat, return_cat=True, overwrite=overwrite, crop_by=None
            )
            # Add total fluxes
            galaxy.add_flux_aper_total(
                catalogue_path="/raid/scratch/work/austind/GALFIND_WORK/Catalogues/v11/ACS_WFC+NIRCam/JOF_psfmatched/JOF_psfmatched_MASTER_Sel-F277W+F356W+F444W_v11_total.fits",
                overwrite=True,
            )

            # Add Detection data
            galaxy.add_detection_data(overwrite=overwrite)

            # Plot segmentation stamps
            # fig = galaxy.plot_seg_stamps()
            # fig.savefig(f'{h5_folder}/diagnostic_plots/{galaxy.galaxy_id}_seg_stamps.png', dpi=300, bbox_inches='tight')
            # plt.close()
            # Currently set to use segmentation map from detection image. May change this in future.
            if galaxy.gal_region in [None, {}] or overwrite:
                galaxy.pixedfit_processing(
                    gal_region_use="detection", overwrite=overwrite
                )  # Maybe seg map should be from detection image?

                # fig = galaxy.plot_gal_region()
                # fig.savefig(f'{h5_folder}/diagnostic_plots/{galaxy.galaxy_id}_gal_region.png', dpi=300, bbox_inches='tight')
                # plt.close()
            if galaxy.pixedfit_map is None or overwrite:
                galaxy.pixedfit_binning(overwrite=overwrite)

            if galaxy.photometry_table in [None, {}] or overwrite:
                galaxy.measure_flux_in_bins(overwrite=overwrite)

            if (
                not os.path.exists(
                    f"{h5_folder}/diagnostic_plots/JOF_psfmatched_{galaxy.galaxy_id}_overview.png"
                )
                or overwrite
            ):
                galaxy.plot_overview(save=True)

            import psutil

            process = psutil.Process(os.getpid())
            print(process.open_files())

            nbins = galaxy.get_number_of_bins()
            if nbins == 1:
                num_of_single_bin += 1

            num_of_bins += nbins

        print(f"Total number of bins to fit: {num_of_bins}")
        print(f"Number of galaxies with only one bin: {num_of_single_bin}")
        # Run Bagpipes in parallel

    from EXPANSE.bagpipes.pipes_models import (
        continuity_dict,
        create_dicts,
        delayed_dict,
        dpl_dict,
        lognorm_dict,
        resolved_dict,
    )

    override_meta = {
        "redshift": "eazy",
        "redshift_sigma": "eazy",
        "use_bpass": True,
    }
    delayed_dicts = create_dicts(
        delayed_dict, override_meta=override_meta, num=len(ids)
    )
    continuity_dicts = create_dicts(
        continuity_dict, override_meta=override_meta, num=len(ids)
    )

    dpl_dicts = create_dicts(
        dpl_dict, override_meta=override_meta, num=len(ids)
    )
    lognorm_dicts = create_dicts(
        lognorm_dict, override_meta=override_meta, num=len(ids)
    )
    resolved_dicts = create_dicts(resolved_dict, num=len(ids))

    # Not fitting resolved yet
    for run_dicts in [
        delayed_dicts,
        continuity_dicts,
        dpl_dicts,
        lognorm_dicts,
    ]:
        size = 1
        if size >= 2:
            # This option for running by with mpirun/mpiexec
            n_jobs = 6
        else:
            if computer == "morgan":
                n_jobs = 6
                backend = "loky"
            elif computer == "singularity":
                n_jobs = np.min([len(galaxies) + 1, 6])
                backend = "multiprocessing"
                backend = "threading"

        if n_jobs == 1:
            print("Running in serial.")
            for i in range(len(ids)):
                print(i)
                print(run_dicts[i])
                galaxies[i].run_bagpipes(run_dicts[i])
        else:
            print(f"Using {n_jobs} cores.")
            for i in range(len(ids)):
                print(run_dicts[i])

            Parallel(n_jobs=n_jobs)(
                delayed(run_bagpipes_wrapper)(
                    galaxy_id,
                    resolved_dict,
                    cutout_size=cutout_size,
                    h5_folder=h5_folder,
                    alert=True,
                )
                for galaxy_id, resolved_dict in zip(ids, run_dicts)
            )

    # Test the Galaxy class
    # galaxy = ResolvedGalaxy.init_from_galfind(645, 'NGDEEP2', 'v11', excl_bands = ['F435W', 'F775W', 'F850LP'])
    # galaxy2 = ResolvedGalaxy.init_from_h5('NGDEEP2_645')

    # Simple test Bagpipes fit_instructions
    """
    sfh = {
        'age_max': (0.03, 0.5), # Gyr 
        'age_min': (0, 0.5), # Gyr
        'metallicity': (1e-3, 2.5), # solar
        'massformed': (4, 12), # log mstar/msun
        }

    nebular = {}
    nebular["logU"] = -2.0 

    dust = {}
    dust["type"] = "Calzetti"
    dust["Av"] = (0, 5.0)

    fit_instructions = {"t_bc":0.01,
                    "constant":sfh,
                    "nebular":nebular,
                    "dust":dust,  
                    }
    meta = {'run_name':'initial_test_cnst_sfh', 'redshift':0.48466974}

    overall_dict = {'meta': meta, 'fit_instructions': fit_instructions}

    galaxy2.run_bagpipes(overall_dict)

    galaxy2.plot_bagpipes_results('initial_test_cnst_sfh')
    #galaxy2.pixedfit_processing()
    
    """


"""
TODO: This commented out code is for adding a galaxy to a blank region of the image.

    # Get the image
    im_data, im_header, seg_data, seg_header, mask = self.load_data(band, incl_mask = True)
    combined_mask = self.combine_seg_data_and_mask(seg_data = seg_data, mask = mask)

    if pos == 0:
        final_mask = combined_mask
    else:
        assert np.shape(final_mask) == np.shape(combined_mask), "Segmentation maps must have same shape"
        final_mask += combined_mask

# Renormalize the image
final_mask[final_mask > 0] = 1
# Dilate by 30 pixel ELLIPSE
from cv2 import dilate, getStructuringElement, MORPH_ELLIPSE
morph_size = cutout_size // 2 + 30
kernel = getStructuringElement(MORPH_ELLIPSE, (morph_size, morph_size))
final_mask = dilate(final_mask, kernel)

# Pick a random position in the image where mask is 0  

possible_pos = final_mask.nonzero()
pos = np.random.choice(len(possible_pos[0]))
y, x = possible_pos[0][pos], possible_pos[1][pos]

print('Selected position', x, y)
# Cutout the image in each band
bckg_cutouts = {}
bckg_err_cutouts = {}
data_with_bckg = {}

for pos, band in enumerate(bands):
    im_data, im_header, seg_data, seg_header, mask = self.load_data(band, incl_mask = True)
    err_data = self.load_rms_err(band)
    cutout = Cutout2D(im_data, (x, y), cutout_size)
    bckg_cutouts[band] = cutout.data
    cutout = Cutout2D(err_data, (x, y), cutout_size)
    bckg_err_cutouts[band] = cutout.data

    # Add the galaxy to the cutout
    data_with_bckg[band] = bckg_cutouts[band] + psf_imgs[band].arr
"""
