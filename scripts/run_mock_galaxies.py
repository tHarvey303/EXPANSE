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
    cnst_dict,
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
    overwrite = True
    bagpipes_only = False  # This is for running Bagpipes only if the galaxies have already been created
    load_only = False  # This is for running Bagpipes - whether to skip running fitting and load existing results
    only_new = (
        False  # This is whether to skip initial running of existing .h5 files.
    )
    cosmo = FlatLambdaCDM(H0=70, Om0=0.300)
    grid_name = "bpass-2.2.1-bin_chabrier03-0.1,300.0_cloudy-c17.03"
    grid_dir = "/nvme/scratch/work/tharvey/synthesizer/grids/"
    path = "/nvme/scratch/work/tharvey/EXPANSE/data/JOF_mock.h5"
    mock_rms_fit_path = "/nvme/scratch/work/tharvey/EXPANSE/data/mock_rms/JOF/"
    fit_photometry = "TOTAL_BIN"
    plot_overview = True

    try:
        n_jobs = int(sys.argv[1])
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

        start_now = False
        for redshift_code in structure:
            for region in structure[redshift_code]:
                for gal_id in structure[redshift_code][region]:
                    print(
                        f"Doing {redshift_code} region {region}, galaxy {gal_id}"
                    )
                    """
                    if start_now or (
                        redshift_code == "010_z005p000"
                        and region == "10"
                        and gal_id == "480"
                    ):
                        #    continue
                        start_now = True
                    else:
                        continue
                    """

                    if only_new and os.path.exists(
                        os.path.join(
                            galaxies_dir,
                            f"{mock_field}_{redshift_code}_{region}_{gal_id}_mock.h5",
                        )
                    ):
                        print("Galaxy already exists. Skipping.")
                        continue
                    try:
                        mock_galaxy = MockResolvedGalaxy.init(
                            mock_survey=mock_field,
                            redshift_code=redshift_code,
                            gal_region=region,
                            gal_id=gal_id,
                            grid_dir=grid_dir,
                            file_path=path,
                            overwrite=False,
                            mock_rms_fit_path=mock_rms_fit_path,
                            debug=False,
                            h5_folder=galaxies_dir,
                        )
                        # Use sep to process the detection image, get aperture fluxes etc using PSF matched imaging.
                        mock_galaxy.sep_process(
                            debug=False, overwrite=overwrite
                        )
                        # Save Synthesizer SED within the detection image region.
                        # if 'det_segmap_fnu' not in mock_galaxy.seds.keys():
                        mock_galaxy.save_new_synthesizer_sed(
                            regenerate_original=True, overwrite=overwrite
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
                            "MAG_APER_0.32 arcsec",
                            update_meta_properties=True,
                            overwrite=overwrite,
                        )

                        if plot_overview:
                            mock_galaxy.plot_overview(
                                bins_to_show=[
                                    "MAG_APER_TOTAL",
                                    "TOTAL_BIN",
                                    "1",
                                ],
                                rgb_q=5,
                                rgb_stretch=6e-5,
                                save=True,
                                show=False,
                            )

                        del mock_galaxy

                    except AssertionError as e:
                        print(e)
                        print(f"Failed on {redshift_code} galaxy {gal_id}")
                        continue

        print("Finished creating mock galaxies.")
        stop = input("Press enter to continue.")

    # Doesn't preserve order otherwise - can't guarantee they will be run in the input order

    galaxies = MockResolvedGalaxy.init_all_field_from_h5(
        mock_field, galaxies_dir, save_out=True, n_jobs=n_jobs
    )

    multiple_galaxies = ResolvedGalaxies(galaxies)

    override_meta = {
        "use_bpass": True,
        "redshift": "self",
        "min_redshift_sigma": 0,
        "redshift_sigma": 0,
        "name_append": "_zfix",
    }
    override_cont_meta = {
        "use_bpass": True,
        "redshift": "self",
        "name_append": "_zfix",
        "update_cont_bins": True,
        "min_redshift_sigma": 0,
        "redshift_sigma": 0,
    }
    resolved_meta = {
        "use_bpass": True,
        "redshift": "self",
        "name_append": "_zfix",
        "min_redshift_sigma": 0,
        "redshift_sigma": 0,
    }
    # Different meta to allow different SFH bins per galaxy
    continuity_dicts = create_dicts(
        continuity_dict, len(galaxies), override_meta=override_cont_meta
    )

    continuity_bursty_dicts = create_dicts(
        continuity_bursty_dict, len(galaxies), override_meta=override_cont_meta
    )

    cnst_dicts = create_dicts(
        cnst_dict, len(galaxies), override_meta=override_meta
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
        resolved_dict_cnst, len(galaxies), override_meta=resolved_meta
    )

    resolved_dicts_bursty = create_dicts(
        resolved_dict_bursty, len(galaxies), override_meta=resolved_meta
    )

    for dicts in [
        # lognorm_dicts,
        cnst_dicts,
        resolved_dicts_cnst,
        continuity_dicts,
        delayed_dicts,
        continuity_bursty_dicts,
        resolved_dicts_bursty,
        dpl_dicts,
    ]:
        multiple_galaxies.run_bagpipes_parallel(
            dicts,
            n_jobs=n_jobs,
            fit_photometry=fit_photometry,
            run_dir=bagpipes_run_dir,
            load_only=load_only,
        )
