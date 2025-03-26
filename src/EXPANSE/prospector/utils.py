import numpy as np
import glob
from numpy.lib.recfunctions import structured_to_unstructured
import fnmatch
import astropy.units as u
from astropy.table import Table
from astropy.io import fits
import copy
import sys

try:
    from mpi4py import MPI

    rank = MPI.COMM_WORLD.Get_rank()
    size = MPI.COMM_WORLD.Get_size()
    from mpi4py.futures import MPIPoolExecutor

    if size > 1:
        MPI.COMM_WORLD.Barrier()
        sys.stdout.flush()
        MPI.COMM_WORLD.Barrier()
except ImportError:
    rank = 0
    size = 1


def filter_catalog(catalog, filter_val, filter_field):
    if filter_val != None and filter_field != None:
        if type(filter_val) in [list, np.ndarray]:
            print(f"Selecting only galaxies which are have {filter_val} in column {filter_field}")
            mask = [True if i in filter_val else False for i in catalog[filter_field]]
            catalog = catalog[mask]
        elif type(filter_val) == tuple and len(filter_val) == 2:
            catalog = catalog[
                (catalog[filter_field] > filter_val[0]) & (catalog[filter_field] < filter_val[1])
            ]
            print(f"Selecting galaxies with {filter_val[0]} < {filter_field} < {filter_val[1]}")
        elif type(filter_val) in [float, int]:
            catalog = catalog[catalog[filter_field] == filter_val]
            print(f"Selecting galaxies where column {filter_field} == {filter_val}")
        elif type(filter_val) == str and filter_val[0] == ">":
            catalog = catalog[catalog[filter_field] > float(filter_val[1:])]
            print(f"Selecting galaxies where column {filter_field} > {filter_val[1]}")
        elif type(filter_val) and filter_val[0] == "<":
            catalog = catalog[catalog[filter_field] < float(filter_val[1:])]
            print(f"Selecting galaxies where column {filter_field} < {filter_val[1]}")

    return catalog


def find_bands(
    table, flux_wildcard="FLUX_APER_*_aper_corr"
):  # , error_wildcard='FLUXERR_APER_*_loc_depth'):
    # glob-like matching for column names
    flux_columns = fnmatch.filter(table.colnames, flux_wildcard)
    # get the band names from the column names
    flux_split = flux_wildcard.split("*")
    flux_bands = [col.replace(flux_split[0], "").replace(flux_split[1], "") for col in flux_columns]
    return flux_bands


def provide_phot(
    table,
    bands=None,
    flux_wildcard="FLUX_APER_*_aper_corr_Jy",
    error_wildcard="FLUXERR_APER_*_loc_depth_10pc_Jy",
    min_percentage_error=0.1,
    flux_unit=u.Jy,
    multi_item_columns_slice=None,
):
    if bands is None:
        bands = find_bands(table)

    flux_columns = [flux_wildcard.replace("*", band) for band in bands]
    error_columns = [error_wildcard.replace("*", band) for band in bands]

    assert all(
        [col in table.colnames for col in flux_columns]
    ), f"Flux columns {flux_columns} not found in table"
    assert all(
        [col in table.colnames for col in error_columns]
    ), f"Error columns {error_columns} not found in table"

    if multi_item_columns_slice is not None:
        raise NotImplementedError("Do this I guess.")

    fluxes = structured_to_unstructured(table[flux_columns].as_array()) * flux_unit
    errors = structured_to_unstructured(table[error_columns].as_array()) * flux_unit

    mask = ((errors / fluxes) < min_percentage_error) & (fluxes > 0)
    errors[mask] = fluxes[mask] * min_percentage_error

    return fluxes, errors


def load_spectra(
    spectra_path,
    wav_column="guess",
    flux_column="guess",
    flux_err_column="guess",
    input_flux_units=None,
    output_flux_units=u.Jy,
    input_wav_units=None,
    output_wav_units=u.AA,
):
    """
    Load spectral data from a file and convert to specified units.

    Args:
        ID: Identifier for the spectrum (appears unused in current implementation)
        spec_units: Target units for spectral flux data
        wav_units: Target units for wavelength data

    Returns:
        numpy.ndarray: Array containing wavelength, flux, and flux error data
    """
    # Column name possibilities
    possible_wavelength_columns = [
        "WAVELENGTH",
        "WAVE",
        "LAMBDA",
        "LAMBDA_OBS",
        "LAMBDA_RF",
        "wav",
        "LAMBDA_OBSERVED",
        "LAMBDA_REST",
        "wavelength",
        "wave",
        "lambda",
        "lambda_obs",
        "lambda_rf",
        "lambda_observed",
        "lambda_rest",
        "Wave",
        "Wavelength",
    ]

    possible_flux_columns = [
        "FLUX",
        "FLUX_DENSITY",
        "FLUX_DENSITY_OBS",
        "FLUX_DENSITY_RF",
        "FLUX_DENSITY_OBSERVED",
        "FLUX_DENSITY_REST",
        "flux",
        "flux_density",
        "flux_density_obs",
        "flux_density_rf",
        "flux_density_observed",
        "flux_density_rest",
        "Flux",
    ]

    possible_flux_err_columns = [
        "FLUX_ERR",
        "FLUX_ERROR",
        "FLUX_ERR_DENSITY",
        "FLUX_ERROR_DENSITY",
        "FLUX_ERR_DENSITY_OBS",
        "FLUX_ERROR_DENSITY_OBS",
        "FLUX_ERR_DENSITY_RF",
        "FLUX_ERROR_DENSITY_RF",
        "FLUX_ERR_DENSITY_OBSERVED",
        "err",
        "FLUX_ERROR_DENSITY_OBSERVED",
        "FLUX_ERR_DENSITY_REST",
        "FLUX_ERROR_DENSITY_REST",
        "flux_err",
        "flux_error",
        "flux_err_density",
        "flux_error_density",
        "flux_err_density_obs",
        "flux_error_density_obs",
        "flux_err_density_rf",
        "flux_error_density_rf",
        "flux_err_density_observed",
        "flux_error_density_observed",
        "flux_err_density_rest",
        "flux_error_density_rest",
        "fluxerr",
        "fluxerror",
    ]

    # Unit mappings
    possible_flux_units = {
        "erg/(s * cm2 * AA": u.erg / (u.s * u.cm**2 * u.Angstrom),
        "mJy": u.mJy,
        "Jy": u.Jy,
        "ergscma": u.erg / (u.s * u.cm**2 * u.Angstrom),
        "uJy": u.uJy,
        "nJy": u.nJy,
    }

    possible_wavelength_units = {
        "Angstrom": u.AA,
        "AA": u.AA,
        "um": u.um,
        "nm": u.nm,
        "micron": u.um,
        "angstrom": u.AA,
    }

    all_possible_flux_units = copy.copy(possible_flux_units)
    all_possible_wavelength_units = copy.copy(possible_wavelength_units)

    for key in possible_flux_units:
        variations = [
            f"_{key}",
            f"_{key.upper()}",
            f"_{key.lower()}",
            f"_{key.capitalize()}",
            f"_{key.title()}",
            f"_{key.swapcase()}",
        ]
        all_possible_flux_units.update({v: possible_flux_units[key] for v in variations})

    for key in possible_wavelength_units:
        variations = [
            f"_{key}",
            f"_{key.upper()}",
            f"_{key.lower()}",
            f"_{key.capitalize()}",
            f"_{key.title()}",
            f"_{key.swapcase()}",
        ]
        all_possible_wavelength_units.update(
            {v: possible_wavelength_units[key] for v in variations}
        )

    # Add unit combinations to column names
    all_possible_flux_columns = possible_flux_columns + [
        col + f"_{key}" for key in all_possible_flux_units for col in possible_flux_columns
    ]
    all_possible_flux_err_columns = possible_flux_err_columns + [
        col + f"_{key}" for key in all_possible_flux_units for col in possible_flux_err_columns
    ]
    all_possible_wavelength_columns = possible_wavelength_columns + [
        col + f"_{key}"
        for key in all_possible_wavelength_units
        for col in possible_wavelength_columns
    ]

    err_unit_found = False
    flux_unit_found = False
    wav_unit_found = False

    # Read data file
    done = False
    # Add special case for DJA files
    if spectra_path.endswith(".fits"):
        with fits.open(spectra_path) as hdul:
            # Check if it has multiple extensions
            if len(hdul) > 1:
                # DJA names are SCI, SPEC1D,
                if "SCI" in hdul and "SPEC1D" in hdul:
                    if rank == 0:
                        print(f"Identified DJA spectrum file with SPEC1D extension.")
                    from msaexp.spectrum import SpectrumSampler

                    spec = SpectrumSampler(hdul)
                    table = spec.spec

                    # Read SPEC1D extension as table
                    flux_column = "flux"
                    flux_err_column = "full_err"
                    wav_column = "wave"

                    header = hdul["SPEC1D"].header
                    table[wav_column].unit = u.Unit(header[f"TUNIT1"])
                    flux = table[flux_column].unit = u.Unit(header[f"TUNIT2"])
                    flux_err = table[flux_err_column].unit = u.Unit(header[f"TUNIT3"])

                    # Delete invalid data
                    mask = table["valid"]
                    table[flux_column][~mask] = np.nan
                    table[flux_err_column][~mask] = np.nan

                    done = True
                    err_unit_found = True
                    flux_unit_found = True
                    wav_unit_found = True

    if spectra_path.endswith(".npy"):
        output = np.load(spectra_path)
        # get dimensions of output - take longest dimension as wav, flux, flux_err
        shape = output.shape
        assert len(shape) == 2, "Numpy array must have 2 dimensions."
        assert isinstance(input_flux_units, u.Unit), "Input spec units must be an astropy unit."
        assert isinstance(input_wav_units, u.Unit), "Input wav units must be an astropy unit."

        if shape[0] < shape[1]:
            output = output.T

        wav = output[0, :] * input_wav_units
        flux = output[1, :] * input_flux_units
        flux_err = output[2, :] * input_flux_units
        table = Table([wav, flux, flux_err], names=("WAVELENGTH", "FLUX", "FLUX_ERR"))
        wav_column = "WAVELENGTH"
        flux_column = "FLUX"
        flux_err_column = "FLUX_ERR"
        done = True

    if not done:
        fmt = "ascii.commented_header" if spectra_path.endswith(".dat") else None
        table = Table.read(spectra_path, format=fmt)

        # Find relevant columns

        for col in table.colnames:
            if wav_column == "guess" and any(
                [col in col_guess for col_guess in all_possible_wavelength_columns]
            ):
                print("Wavelength column found:", col)
                wav_column = col
            if flux_column == "guess" and any(
                [col in col_guess for col_guess in all_possible_flux_columns]
            ):
                print("Flux column found:", col)
                flux_column = col
            if flux_err_column == "guess" and any(
                [col in col_guess for col_guess in all_possible_flux_err_columns]
            ):
                print("Flux error column found:", col)
                flux_err_column = col

    if wav_column is None:
        raise ValueError(f"No wavelength column found in data file. Columns: {table.colnames}")
    if flux_column is None:
        raise ValueError(f"No flux column found in data file. Columns: {table.colnames}")
    if flux_err_column is None:
        raise ValueError(f"No flux error column found in data file. Columns: {table.colnames}")

    # Process units
    if flux_unit_found and err_unit_found:
        pass
    elif input_flux_units is not None:
        flux_units = input_flux_units
        table[flux_column].unit = flux_units
        table[flux_err_column].unit = flux_units
        err_unit_found = True
        flux_unit_found = True
        print(f"Using input flux units: {input_flux_units}")

    else:
        if table[flux_column].unit is None:
            for unit in all_possible_flux_units:
                table_flux_units = all_possible_flux_units[unit]

                # Check flux error units
                if not err_unit_found and (
                    unit in flux_err_column
                    or (
                        table[flux_err_column].unit is not None
                        and unit in str(table[flux_err_column].unit)
                    )
                ):
                    print("Flux error units found:", unit)
                    table[flux_err_column].unit = table_flux_units
                    err_unit_found = True

                # Check flux units
                if not flux_unit_found and (
                    unit in flux_column
                    or (
                        table[flux_column].unit is not None and unit in str(table[flux_column].unit)
                    )
                ):
                    print("Flux units found:", unit)
                    table[flux_column].unit = table_flux_units
                    flux_units = table_flux_units
                    flux_unit_found = True

                if flux_unit_found and err_unit_found:
                    break
        else:
            flux_units = table[flux_column].unit
            flux_unit_found = True
            err_unit_found = True

    # Set default units if not found
    if not flux_unit_found:
        print("No flux units found. Assuming erg/(s * cm2 * AA).")
        flux_units = u.erg / (u.s * u.cm**2 * u.Angstrom)
        table[flux_column].unit = flux_units

    if not err_unit_found:
        print("No flux error units found. Assuming same as flux.")
        table[flux_err_column].unit = flux_units

    if wav_unit_found:
        pass
    elif input_wav_units is None:
        if table[wav_column].unit is None:
            # Process wavelength units
            for unit in all_possible_wavelength_units:
                if unit in wav_column or (
                    table[wav_column].unit is not None and unit in str(table[wav_column].unit)
                ):
                    print("Wavelength units found:", unit)
                    table[wav_column].unit = all_possible_wavelength_units[unit]
                    break
    else:
        print(f"Using input wavelength units: {input_wav_units}")
        table[wav_column].unit = input_wav_units

    # Extract and convert data
    wav = table[wav_column].to(output_wav_units)
    flux = table[flux_column].to(output_flux_units, equivalencies=u.spectral_density(wav))
    flux_err = table[flux_err_column].to(output_flux_units, equivalencies=u.spectral_density(wav))

    # Remove NaN values
    nanmask = np.isnan(flux)
    wav = wav[~nanmask]
    flux = flux[~nanmask]
    flux_err = flux_err[~nanmask]

    # assert wav is monotonically increasing
    assert np.all(np.diff(wav) > 0), "Wavelengths must be monotonically increasing."

    return wav, flux, flux_err
