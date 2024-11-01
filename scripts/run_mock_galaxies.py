import astropy.units as u
import numpy as np
from astropy.cosmology import FlatLambdaCDM
from joblib import Parallel, delayed
import os
import sys
import time

from EXPANSE import MockResolvedGalaxy, run_bagpipes_wrapper, ResolvedGalaxies
import h5py
from EXPANSE.bagpipes import (
    calculate_bins,
    continuity_dict,
    continuity_bursty_dict,
    create_dicts,
    delayed_dict,
    dpl_dict,
    lognorm_dict,
    resolved_dict_cnst,
    resolved_dict_bursty,
)

file_path = os.path.abspath(__file__)

if os.path.exists("/.singularity.d/Singularity"):
    computer = "singularity"
elif "nvme" in file_path:
    computer = "morgan"
elif "Users" in file_path:
    computer = "mac"
else:
    computer = "unknown"

if computer == "morgan":
    galaxies_dir = "/nvme/scratch/work/tharvey/EXPANSE/galaxies/mock/"
    bagpipes_run_dir = "/nvme/scratch/work/tharvey/EXPANSE/pipes/"
elif computer == "singularity":
    galaxies_dir = "/mnt/galaxies/mock/"
    bagpipes_run_dir = "/mnt/pipes/"


if __name__ == "__main__":
    mock_field = "JOF_psfmatched"
    overwrite = False
    initial_creation = True
    cosmo = FlatLambdaCDM(H0=70, Om0=0.300)
    grid_name = "bpass-2.2.1-bin_chabrier03-0.1,300.0_cloudy-c17.03"
    grid_dir = "/nvme/scratch/work/tharvey/synthesizer/grids/"
    path = "/nvme/scratch/work/tharvey/EXPANSE/data/JOF_mock.h5"
    mock_rms_fit_path = "/nvme/scratch/work/tharvey/EXPANSE/data/mock_rms/JOF/"
    fit_photometry = "TOTAL_BIN"
    bagpipes_only = False
    load_only = False

    try:
        n_jobs = sys.argv[1]
    except:
        n_jobs = 6

    if not bagpipes_only:
        with h5py.File(path, "r") as hf:
            structure = {i: {} for i in hf.keys()}

            for zbin in hf.keys():
                for region in hf[zbin].keys():
                    structure[zbin][region] = list(hf[zbin][region].keys())

        num_of_galaxies = np.sum(
            [
                len(structure[zbin][region])
                for zbin in structure
                for region in structure[zbin]
            ]
        )

        # -------------------------------------------------------------------------------------
        # Setting up two Bagpipes fits - one for the unresolved photometry (delayed SFH) and one for the resolved photometry (constant SFH)
        # -------------------------------------------------------------------------------------

        print(f"Creating {num_of_galaxies} mock galaxies")

        galaxies = []
        for redshift_code in structure:
            for region in structure[redshift_code]:
                for gal_id in structure[redshift_code][region]:
                    print(
                        f"Doing {redshift_code} region {region}, galaxy {gal_id}"
                    )

                    try:
                        mock_galaxy = MockResolvedGalaxy.init(
                            mock_survey=mock_field,
                            redshift_code=redshift_code,
                            gal_region=region,
                            gal_id=gal_id,
                            grid_dir=grid_dir,
                            file_path=path,
                            overwrite=overwrite,
                            mock_rms_fit_path=mock_rms_fit_path,
                            debug=False,
                            h5_folder=galaxies_dir,
                        )
                        mock_galaxy.pixedfit_processing(
                            gal_region_use="detection",
                            overwrite=overwrite,
                            dir_images=galaxies_dir,
                        )
                        # Maybe seg map should be from detection image?
                        mock_galaxy.pixedfit_binning(overwrite=overwrite)
                        mock_galaxy.measure_flux_in_bins(overwrite=overwrite)
                        mock_galaxy.eazy_fit_measured_photometry(
                            "MAG_APER_0.32 arcsec", update_meta_properties=True
                        )
                        if not initial_creation:
                            galaxies.append(mock_galaxy)  # Clear memory
                        else:
                            del mock_galaxy

                    except AssertionError as e:
                        print(e)
                        print(f"Failed on {redshift_code} galaxy {gal_id}")
                        continue

        print("Finished creating mock galaxies.")
        stop = input("Press enter to continue.")

    # Doesn't preserve order otherwise - can't guarantee they will be run in the input order
    if bagpipes_only:
        galaxies = MockResolvedGalaxy.init_all_field_from_h5(
            mock_field, galaxies_dir, save_out=False
        )

    multiple_galaxies = ResolvedGalaxies(galaxies)

    override_meta = {"use_bpass": True, "redshift": "eazy"}
    override_cont_meta = {
        "use_bpass": True,
        "redshift": "eazy",
        "update_cont_bins": True,
    }
    # Different meta to allow different SFH bins per galaxy
    continuity_dicts = create_dicts(
        continuity_dict, len(galaxies), override_meta=override_cont_meta
    )

    continuity_bursty_dicts = create_dicts(
        continuity_bursty_dict, len(galaxies), override_meta=override_cont_meta
    )

    delayed_dicts = create_dicts(
        delayed_dict, len(galaxies), override_meta=override_meta
    )
    dpl_dicts = create_dicts(
        dpl_dict, len(galaxies), override_meta=override_meta
    )
    lognorm_dicts = create_dicts(
        lognorm_dict, len(galaxies), override_meta=override_meta
    )
    resolved_dicts_cnst = create_dicts(
        resolved_dict_cnst, len(galaxies), override_meta={"use_bpass": True}
    )

    resolved_dicts_bursty = create_dicts(
        resolved_dict_bursty, len(galaxies), override_meta={"use_bpass": True}
    )

    for dicts in [
        continuity_bursty_dicts,
        continuity_dicts,
        delayed_dicts,
        dpl_dicts,
        lognorm_dicts,
        resolved_dicts,
        resolved_dicts_bursty,
    ]:
        multiple_galaxies.run_bagpipes_parallel(
            dicts,
            n_jobs=n_jobs,
            fit_photometry=fit_photometry,
            run_dir=bagpipes_run_dir,
            load_only=load_only,
        )
