import glob
import os
import astropy.units as u
import h5py
from astropy.table import Table
import ast
import numpy as np

try:
    import dense_basis as db
except ImportError:
    print(
        "Warning: dense_basis not found. Dense_basis functions will not work"
    )


def get_filter_files(
    root_file="/nvme/scratch/work/tharvey/jwst_filters/",
    filter_set_name="nircam_acs_wfc",
    db_dir="/nvme/scratch/work/tharvey/dense_basis/scripts/filters/filter_curves",
    instruments={
        "ACS_WFC": ["f435W", "f606W", "f814W", "f775W", "f850LP"],
        "nircam": [
            "f090W",
            "f115W",
            "f150W",
            "f162M",
            "f182M",
            "f200W",
            "f210M",
            "f250M",
            "f277W",
            "f300M",
            "f335M",
            "f356W",
            "f410M",
            "f444W",
        ],
    },
    wav_units={"ACS_WFC": u.AA, "nircam": u.um},
    output_wav_unit=u.AA,
):
    filter_files = []
    for instrument in instruments:
        for filter in instruments[instrument]:
            files = glob.glob(f"{root_file}/{instrument}/{filter.upper()}*")
            if len(files) == 0:
                print(f"No files found for {instrument} {filter}")
            elif len(files) > 1:
                print(f"Multiple files found for {instrument} {filter}")
            else:
                tab = Table.read(files[0], format="ascii")
                wav = tab.columns[0] * wav_units[instrument]  # first column
                throughput = tab.columns[1]
                output_tab = Table(
                    [wav.to(output_wav_unit), throughput],
                    names=["wav", "throughput"],
                )
                output_filename = f"{instrument.upper()}_{filter.upper()}.dat"
                output_tab.write(
                    f"{db_dir}/{output_filename}",
                    format="ascii.commented_header",
                    overwrite=True,
                )
                filter_files.append(output_filename)
    # Save list of paths to txt
    path = f"{db_dir}/{filter_set_name}.txt"

    with open(path, "w") as f:
        for file in filter_files:
            f.write(file + "\n")

    return f"{filter_set_name}.txt"


def run_db_fit_parallel(
    obs_sed,
    obs_err,
    db_atlas_name,
    atlas_path,
    bands=None,
    use_emcee=False,
    emcee_samples=10_000,
    min_flux_err=0.1,
    fix_redshift=False,
    round_negative_fluxes=True,
    warn_missing_bands=True,
    delta_z=0.01,
    verbose=True,
    evaluate_posterior_percentiles=True,
    atlas=None,
):
    """
    Run a dense_basis fit in parallel

    obs_sed: np.array - observed SED in uJy
    obs_err: np.array - observed SED uncertainties in uJy


    """

    # Set the minimum flux error
    obs_err[(obs_err / obs_sed < min_flux_err) & (obs_sed > 0)] = (
        min_flux_err
        * obs_sed[(obs_err / obs_sed < min_flux_err) & (obs_sed > 0)]
    )

    N_param = int(atlas_path.split("Nparam_")[1].split(".dbatlas")[0])
    N_pregrid = int(atlas_path.split("_Nparam")[0].split("_")[-1])
    grid_name = os.path.basename(atlas_path).split(f"_{N_pregrid}")[0]
    if atlas is None:
        if verbose:
            print(
                f"Loading atlas {db_atlas_name} with N_pregrid={N_pregrid} and N_param={N_param} at {os.path.dirname(atlas_path)}"
            )

        atlas = db.load_atlas(
            grid_name,
            N_pregrid=N_pregrid,
            N_param=N_param,
            path=f"{os.path.dirname(atlas_path)}/",
        )

    with h5py.File(atlas_path, "r") as f:
        atlas_bands = f.attrs["bands"]
        atlas_bands = ast.literal_eval(atlas_bands)

    if bands is None:
        assert (
            len(atlas_bands) == len(obs_sed)
        ), f"{len(atlas_bands)} != {len(obs_sed)} number of bands in grid must match length of input photometry if bands are not specified"
    else:
        # Check all bands are in the atlas
        assert (
            len(bands) == len(obs_sed) == len(obs_err)
        ), f"{len(bands)} != {len(obs_sed)} != {len(obs_err)} number of bands must match length of input photometry"
        if warn_missing_bands:
            if not all([band in atlas_bands for band in bands]) and verbose:
                print(
                    f"Warning: Some bands are not in the atlas: {[band for band in bands if band not in atlas_bands]}"
                )
        else:
            assert all(
                [band in atlas_bands for band in bands]
            ), f"All bands must be in the atlas: {[band for band in bands if band not in atlas_bands]} is not in the atlas"
        # Order the bands in the same order as the atlas
        fit_bands = []
        fit_mask = []  # Temporary

        obs_sed_new = []
        obs_err_new = []

        for pos, band in enumerate(atlas_bands):
            if band in bands:
                fit_bands.append(band)
                obs_sed_new.append(obs_sed[bands.index(band)])
                obs_err_new.append(obs_err[bands.index(band)])
                fit_mask.append(True)
            else:
                obs_sed_new.append(np.nan)
                obs_err_new.append(np.nan)
                fit_mask.append(False)

        obs_sed = np.array(obs_sed_new)
        obs_err = np.array(obs_err_new)

    # print(atlas_bands)
    # print(f"Running fit with bands: {fit_bands}")

    if round_negative_fluxes:
        mask = obs_sed < 0
        obs_sed[mask] = 1e-6 * obs_err[mask]
    # Need to generate obs_sed, obs_err, and fit_mask based on the input filter files

    # run_emceesampler takes zbest and deltaz. Can't fix redshift precisely (i.e. between grid spacing), but can set zbest and small deltaz to only allow
    # templates closest to known redshift to be used. Could precheck grid for nearest redshifts and set deltaz accordinglu.
    if fix_redshift:
        redshift = fix_redshift

    sedfit = db.SedFit(
        obs_sed,
        obs_err,
        atlas,
        fit_mask=fit_mask,
        zbest=redshift,
        deltaz=delta_z,
    )

    if use_emcee:
        sampler = db.run_emceesampler(
            obs_sed,
            obs_err,
            atlas,
            epochs=emcee_samples,
            plot_posteriors=False,
            fit_mask=fit_mask,
            zbest=redshift,
            deltaz=delta_z,
        )
        # Need to somehow propogate the sampler back to the sedfit object

    else:
        # pass the atlas and the observed SED + uncertainties into the fitter,
        sedfit.evaluate_likelihood()
        if evaluate_posterior_percentiles:
            sedfit.evaluate_posterior_percentiles()

    return sedfit


def make_db_grid(
    bands,
    db_dir="/nvme/scratch/work/tharvey/dense_basis/scripts/filters/filter_curves",
    filter_set_name="JOF",
    fname="db_atlas_JOF_",
    N_pregrid=10000,
    pregrid_path="pregrids/",
    N_sfh_priors=3,
    parameters={
        "mass": {"min": 5, "max": 12},
        "Z": {"min": -4, "max": 0.5},
        "Av": {"min": 0, "max": 6},
        "z": {"min": 0, "max": 25},
    },
):
    hst_bands = ["F435W", "F606W", "F814W", "F775W", "F850LP"]

    hst_bands_used = []
    nircam_bands = []

    for band in bands:
        if band in hst_bands:
            hst_bands_used.append(band.replace("F", "f"))
        else:
            nircam_bands.append(band.replace("F", "f"))

    filter_list = get_filter_files(
        db_dir=db_dir,
        instruments={"ACS_WFC": hst_bands_used, "nircam": nircam_bands},
        filter_set_name=fname,
    )

    priors = db.Priors()

    priors.Nparam = N_sfh_priors

    for param in parameters.keys():
        for key in parameters[param]:
            setattr(priors, f"{param}_{key}", parameters[param][key])

    db.generate_atlas(
        N_pregrid=N_pregrid,
        priors=priors,
        fname=fname,
        store=True,
        path=pregrid_path,
        filter_list=filter_list,
        filt_dir=db_dir,
    )

    h5_path = (
        f"{pregrid_path}/{fname}_{N_pregrid}_Nparam_{N_sfh_priors}.dbatlas"
    )

    with h5py.File(h5_path, "a") as f:
        # add bands as metadata
        f.attrs["bands"] = str(bands)

    return h5_path


def get_priors(atlas_path=None):
    priors = db.Priors()

    if atlas_path is not None:
        Nparam = int(atlas_path.split("Nparam_")[1].split(".dbatlas")[0])
        priors.Nparam = Nparam
        # Do this better!
        # with h5py.File(atlas_path, "r") as f:

    return priors


def get_bands_from_atlas(atlas_path):
    with h5py.File(atlas_path, "r") as f:
        bands = f.attrs["bands"]
        bands = ast.literal_eval(bands)
    return bands


[
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "10",
    "11",
    "12",
    "13",
    "14",
    "15",
    "16",
    "17",
    "18",
]
