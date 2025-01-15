import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from astropy.table import Table
from synthesizer.imaging import Image, ImageCollection

filt_names = [
    "F070W",
    "F090W",
    "F115W",
    "F150W",
    "F200W",
    "F277W",
    "F356W",
    "F444W",
    "F140M",
    "F162M",
    "F182M",
    "F210M",
    "F250M",
    "F300M",
    "F335M",
    "F360M",
    "F410M",
    "F430M",
    "F460M",
    "F480M",
]
filter_labels = [
    r"${\rm F070W}$",
    r"${\rm F090W}$",
    r"${\rm F115W}$",
    r"${\rm F150W}$",
    r"${\rm F200W}$",
    r"${\rm F277W}$",
    r"${\rm F356W}$",
    r"${\rm F444W}$",
    r"${\rm F140M}$",
    r"${\rm F162M}$",
    r"${\rm F182M}$",
    r"${\rm F210M}$",
    r"${\rm F250M}$",
    r"${\rm F300M}$",
    r"${\rm F335M}$",
    r"${\rm F360M}$",
    r"${\rm F410M}$",
    r"${\rm F430M}$",
    r"${\rm F460M}$",
    r"${\rm F480M}$",
]

filt_dict = {
    filt_names[i]: {
        "idx": i,
        "label": filter_labels[i],
    }
    for i in range(len(filt_names))
}

ndir = 10  # number of lines of sight used


def generate_full_images(
    image_dir,
    halo_id,
    redshift,
    direction=0,
    bands=filt_names,
    sphinx_data_dir="/raid/scratch/work/tharvey/SPHINX/SPHINX-20-data/",
):
    full_catalogue = Table.read(f"{sphinx_data_dir}/data/all_basic_data.csv")

    # Get the galaxy

    galaxy = full_catalogue[
        (full_catalogue["halo_id"] == halo_id)
        & (full_catalogue["redshift"] == redshift)
    ]
    assert len(galaxy) == 1

    # line images
    line_image = np.load(f"{image_dir}/halo_{halo_id}_lines_dir_0.npy")
    # Star images
    star_image = np.load(
        f"{image_dir}/halo_{halo_id}_continuum_mags_dir_{direction}.npy"
    )
    # Nebular continuum images
    nebc_image = np.load(
        f"{image_dir}/halo_{halo_id}_mags_dir_{direction}.npy"
    )

    npix = int(np.sqrt(nebc_image.shape[0]))
    assert npix**2 == nebc_image.shape[0], "Image shape is not square"

    # img = np.zeros((npix,npix,len(bands)))

    # Sum the flux in the channels

    images = {}
    for pos, fi in enumerate(bands):
        img = np.zeros((npix, npix))
        # img[:,:,pos] += 3631*(10.**(star_image[:,filt_dict[fi]["idx"]].reshape(npix,npix)/(-2.5)))
        # img[:,:,pos] += 3631*(10.**(nebc_image[:,filt_dict[fi]["idx"]].reshape(npix,npix)/(-2.5)))
        # img[:,:,pos] += 3631*(10.**(line_image[filt_dict[fi]["idx"],direction,:].reshape(npix,npix)/(-2.5)))
        img += 3631 * (
            10.0
            ** (
                star_image[:, filt_dict[fi]["idx"]].reshape(npix, npix)
                / (-2.5)
            )
        )
        img += 3631 * (
            10.0
            ** (
                nebc_image[:, filt_dict[fi]["idx"]].reshape(npix, npix)
                / (-2.5)
            )
        )
        img += 3631 * (
            10.0
            ** (
                line_image[filt_dict[fi]["idx"], direction, :].reshape(
                    npix, npix
                )
                / (-2.5)
            )
        )
        # img array is now in Jy
        img *= 1e6  # Image is now in microJy

        image = Image(resolution, fov, img=img)
        images[bands[pos]] = image

        return images


def get_spectra(
    halo_id,
    redshift,
    direction=0,
    sphinx_data_dir="/raid/scratch/work/tharvey/SPHINX/SPHINX-20-data/",
    spec_type="total",
):
    """
    Possible types
    'total', 'stellar_continuum', 'nebular_continuum', 'emission_line'

    """

    full_catalogue = Table.read(f"{sphinx_data_dir}/data/all_basic_data.csv")

    # Get the galaxy

    galaxy = full_catalogue[
        (full_catalogue["halo_id"] == halo_id)
        & (full_catalogue["redshift"] == redshift)
    ]
    assert len(galaxy) == 1

    # Get spectra

    spectra_path = f"/raid/scratch/work/tharvey/SPHINX/SPHINX-20-data/data/spectra/all_spec_z{redshift}.json"

    with open(spectra_path, "r") as f:
        spectra = json.load(f)

    spectra_gal = spectra[str(halo_id)]

    wavelengths = np.array(spectra_gal["wavelengths"])  # in um
    wavelengths *= u.um

    flux = (
        10.0 ** np.array(spectra_gal[f"dir_{direction}"][spec_type])
        * u.erg
        / u.s
        / u.cm**2
        / u.Hz
    )

    flux = flux.to(u.uJy, equivalencies=u.spectral_density(wavelengths))

    seds = {}

    seds["wav"] = wavelengths.to(u.Angstrom).value

    seds[f"{spec_type}_total_fnu"] = {}
    seds[f"{spec_type}_total_fnu"]["total"] = flux.to(u.uJy).value

    return seds


def get_sfh(
    halo_id,
    redshift,
    direction=0,
    sphinx_data_dir="/raid/scratch/work/tharvey/SPHINX/SPHINX-20-data/",
):
    sfh_path = f"{sphinx_data_dir}/SFHs/sfhs_z{int(redshift)}.json"
    with open(sfh_path, "r") as f:
        sfhs = json.load(f)

    age_bins = np.array(sfhs["age_bins"])
    sfh = sfhs["sfhs"][str(gal_id)]

    age_bins_mid = 0.5 * (age_bins[1:] + age_bins[:-1])

    sfh_dict = {}

    sfh_dict["total_bin"]

    time = age_bins.to(u.Gyr).value
    savedata = np.vstack([time, sfh]).T

    sfh_dict["total_bin"] = savedata

    return sfh_dict


def get_meta(
    halo_id,
    redshift,
    direction=0,
    sphinx_data_dir="/raid/scratch/work/tharvey/SPHINX/SPHINX-20-data/",
    columns_to_add="all_direction",
):
    full_catalogue = Table.read(f"{sphinx_data_dir}/data/all_basic_data.csv")

    # Get the galaxy

    galaxy = full_catalogue[
        (full_catalogue["halo_id"] == halo_id)
        & (full_catalogue["redshift"] == redshift)
    ]
    assert len(galaxy) == 1

    meta_properties = {
        "redshift": redshift,
        "halo_id": halo_id,
        "direction": direction,
    }

    if columns_to_add == "all":
        columns_to_add = galaxy.colnames
    elif columns_to_add == "all_direction":
        columns_to_add = []
        for col in galaxy.colnames:
            if "_dir" in col:
                if f"_dir_{int(direction)}" in col:
                    columns_to_add.append(col)
            else:
                columns_to_add.append(col)

    for col in columns_to_add:
        meta_properties[col] = galaxy[col][0]

    return meta_properties


if __name__ == "__main__":
    generate_full_images(
        "/nvme/scratch/work/tharvey/SPHINX/z7_halo_62333", 62333, 7.0
    )
