import astropy.units as u
import numpy as np
from astropy.cosmology import FlatLambdaCDM
from joblib import Parallel, delayed

from EXPANSE import MockResolvedGalaxy, run_bagpipes_wrapper
import h5py
from EXPANSE.bagpipes import (
    calculate_bins,
    continuity_dict,
    create_dicts,
    delayed_dict,
    dpl_dict,
    lognorm_dict,
    resolved_dict,
)

if __name__ == "__main__":
    n_jobs = 8
    overwrite = False
    cosmo = FlatLambdaCDM(H0=70, Om0=0.300)
    grid_name = "bpass-2.2.1-bin_chabrier03-0.1,300.0_cloudy-c17.03"
    grid_dir = "/nvme/scratch/work/tharvey/synthesizer/grids/"
    path = "/nvme/scratch/work/tharvey/EXPANSE/data/JOF_mock.h5"
    mock_rms_fit_path = "/nvme/scratch/work/tharvey/EXPANSE/data/mock_rms/JOF/"
    h5_folder = "galaxies/mock/"

    with h5py.File(path, "r") as hf:
        structure = {i: {} for i in hf.keys()}

        for zbin in hf.keys():
            for region in hf[zbin].keys():
                structure[zbin][region] = list(hf[zbin][region].keys())

    run_ids = []
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

    redshifts = []
    for redshift_code in structure:
        for region in structure[redshift_code]:
            for gal_id in structure[redshift_code][region]:
                print(
                    f"Doing {redshift_code} region {region}, galaxy {gal_id}"
                )

                try:
                    mock_galaxy = MockResolvedGalaxy.init(
                        mock_survey="JOF_psfmatched",
                        redshift_code=redshift_code,
                        gal_region=region,
                        gal_id=gal_id,
                        grid_dir=grid_dir,
                        file_path=path,
                        overwrite=overwrite,
                        mock_rms_fit_path=mock_rms_fit_path,
                        debug=False,
                        h5_folder=h5_folder,
                    )
                    mock_galaxy.pixedfit_processing(
                        gal_region_use="detection",
                        overwrite=overwrite,
                        dir_images=h5_folder,
                    )
                    # Maybe seg map should be from detection image?
                    # mock_galaxy.pixedfit_binning(overwrite=overwrite)
                    # mock_galaxy.measure_flux_in_bins(overwrite=overwrite)
                    run_ids.append(mock_galaxy.galaxy_id)
                    redshifts.append(mock_galaxy.redshift)
                    del mock_galaxy  # Clear memory

                except AssertionError as e:
                    print(e)
                    print(f"Failed on {redshift_code} galaxy {gal_id}")
                    continue

    print("Finished creating mock galaxies.")
    stop = input("Press enter to continue.")
    continuity_dicts = create_dicts(continuity_dict, len(run_ids))
    delayed_dicts = create_dicts(delayed_dict, len(run_ids))
    dpl_dicts = create_dicts(dpl_dict, len(run_ids))
    lognorm_dicts = create_dicts(lognorm_dict, len(run_ids))
    resolved_dicts = create_dicts(resolved_dict, len(run_ids))

    # Update continuity_dicts_bins
    for pos, dict in enumerate(continuity_dicts):
        continuity_dicts[pos]["fit_instructions"]["continuity"][
            "bin_edges"
        ] = list(
            calculate_bins(
                redshift=redshifts[pos],
                num_bins=6,
                first_bin=10 * u.Myr,
                second_bin=None,
                return_flat=True,
                output_unit="Myr",
                log_time=False,
            )
        )

    # Doesn't preserve order otherwise - can't guarantee they will be run in the input order
    for dict in [
        delayed_dicts,
        continuity_dicts,
        dpl_dicts,
        lognorm_dicts,
        resolved_dicts,
    ]:
        Parallel(n_jobs=n_jobs)(
            delayed(run_bagpipes_wrapper)(
                galaxy_id,
                dic,
                cutout_size=None,  # PLACEHOLDER, not used
                overwrite=False,
                overwrite_internal=True,
                # if dic["meta"]["run_name"] == "CNST_SFH_RESOLVED"
                # else False,
                h5_folder=h5_folder,
            )
            for galaxy_id, dic in zip(run_ids, dict)
        )
