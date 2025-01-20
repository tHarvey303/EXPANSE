import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from astropy.table import Table
from synthesizer.imaging import Image, ImageCollection
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
from astropy.wcs import WCS
from astropy.io import fits
from reproject import reproject_adaptive


cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

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
    n_image_pixels=1000,  # number of pixels on the side of the full image
    out_pixel_scale=0.03 * u.arcsec,
    sphinx_data_dir="/raid/scratch/work/tharvey/SPHINX/SPHINX-20-data/",
):
    full_catalogue = Table.read(f"{sphinx_data_dir}/data/all_basic_data.csv")

    # Get the galaxy

    galaxy = full_catalogue[
        (full_catalogue["halo_id"].astype(str) == str(halo_id))
        & (full_catalogue["redshift"].astype(str) == str(float(redshift)))
    ]
    assert (
        len(galaxy) == 1
    ), f"Galaxy not found: {halo_id}, {redshift} or too many: {len(galaxy)}"

    # Get the image size
    l_box_kpc = (
        20 * 1000 / (1 + galaxy["redshift"][0])
    )  # the SPHINX box is 20 comoving Mpc wide so we correct for redshift
    img_side = (
        2.0 * galaxy["rvir"][0] * l_box_kpc
    )  # image side in physical kpc. Rvir is normalised by the boxsize at that redshift
    pixel_size_kpc = img_side / n_image_pixels
    pixel_size_kpc = pixel_size_kpc
    in_pixel_scale = (
        pixel_size_kpc
        / cosmo.angular_diameter_distance(galaxy["redshift"][0])
        .to(u.kpc)
        .value
    ) * u.rad.to(u.arcsec)
    in_pixel_scale = in_pixel_scale * u.arcsec

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

        assert np.all(
            np.isnan(img) == False
        ), f"Image is all NaN: {fi}, {halo_id}, {redshift}, {direction}"

        # print(f"Resizing image from native {in_pixel_scale:.5f} to output {out_pixel_scale:.5f}")

        if out_pixel_scale != in_pixel_scale:
            img = resize(
                img, out_pix_scale=out_pixel_scale, in_pix_scale=in_pixel_scale
            )

        assert np.all(
            np.isnan(img) == False
        ), f"Image is all NaN (after resizing): {fi}, {halo_id}, {redshift}, {direction}"
        # fov in kpc as physical size in unyt quantity
        # At redshift
        from unyt import kpc, uJy

        img *= uJy
        resolution = pixel_size_kpc * kpc
        fov = (img.shape[0] - 0.00001) * resolution

        image = Image(resolution, fov, img=img)
        images[bands[pos]] = image

    image_collection = ImageCollection(
        imgs=images, fov=fov, resolution=resolution, npix=img.shape[0]
    )

    return image_collection


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
        (full_catalogue["halo_id"].astype(str) == str(halo_id))
        & (full_catalogue["redshift"].astype(str) == str(float(redshift)))
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
    sfh_path = f"{sphinx_data_dir}/data/SFHs/sfhs_z{int(redshift)}.json"
    with open(sfh_path, "r") as f:
        sfhs = json.load(f)

    age_bins = np.array(sfhs["age_bins"])
    sfh = sfhs["sfhs"][str(halo_id)]

    age_bins_mid = 0.5 * (age_bins[1:] + age_bins[:-1]) * u.Myr

    sfh_dict = {}

    time = age_bins_mid.to(u.Gyr).value
    savedata = np.vstack([time, sfh]).T

    sfh_dict["total_bin"] = savedata

    return sfh_dict


def get_meta(
    halo_id,
    redshift,
    image_dir="",
    direction=0,
    sphinx_data_dir="/raid/scratch/work/tharvey/SPHINX/SPHINX-20-data/",
    columns_to_add="all_direction",
):
    full_catalogue = Table.read(f"{sphinx_data_dir}/data/all_basic_data.csv")

    # Get the galaxy

    galaxy = full_catalogue[
        (full_catalogue["halo_id"].astype(str) == str(halo_id))
        & (full_catalogue["redshift"].astype(str) == str(float(redshift)))
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

    # Add image paths

    if image_dir != "":
        for band in filt_names:
            meta_properties[f"{band}_cont_image"] = (
                f"{image_dir}/halo_{halo_id}_continuum_mags_dir_{direction}.npy"
            )
            meta_properties[f"{band}_neb_image"] = (
                f"{image_dir}/halo_{halo_id}_mags_dir_{direction}.npy"
            )
            meta_properties[f"{band}_line_image"] = (
                f"{image_dir}/halo_{halo_id}_lines_dir_0.npy"
            )

    for col in columns_to_add:
        meta_properties[col] = galaxy[col][0]

    return meta_properties


def get_mass_ratio(halo_id, redshift):
    halo_id = str(int(halo_id))

    if redshift not in sfh_cache:
        sfh_path = f"/raid/scratch/work/tharvey/SPHINX/SPHINX-20-data/data/SFHs/sfhs_z{int(redshift)}.json"

        with open(sfh_path, "r") as f:
            sfhs = json.load(f)
        sfh_cache[redshift] = sfhs

    else:
        sfhs = sfh_cache[redshift]

    age_bins = np.array(sfhs["age_bins"])
    sfh = sfhs["sfhs"][str(halo_id)]

    age_bins_mid = 0.5 * (age_bins[1:] + age_bins[:-1]) * u.Myr
    csfh = np.cumsum(sfh[::-1])[::-1]

    # Convert to correct units given age is in Myr - each step should be * 10**6

    csfh *= 10**6

    age_idx = np.where(age_bins_mid > 100 * u.Myr)[0]
    mass_ratio = csfh[age_idx[0]] / csfh[0]

    return mass_ratio


def plot_sfh(halo_id, redshift, ax=None, fig=None):
    if fig is None:
        fig, ax = plt.subplots(dpi=200)

    if redshift not in sfh_cache:
        sfh_path = f"/raid/scratch/work/tharvey/SPHINX/SPHINX-20-data/data/SFHs/sfhs_z{int(redshift)}.json"
        with open(sfh_path, "r") as f:
            sfhs = json.load(f)
    else:
        sfhs = sfh_cache[redshift]

    age_bins = np.array(sfhs["age_bins"])
    sfh = sfhs["sfhs"][str(halo_id)]

    age_bins_mid = 0.5 * (age_bins[1:] + age_bins[:-1])

    age_bins_mid *= u.Myr

    ax.plot(age_bins_mid.to(u.Myr), sfh, label=f"ID: {halo_id}, z={redshift}")
    # Reverse cumsum

    # From agebins and SFH, calculate cumaltive stellar mass formed.

    csfh = np.cumsum(sfh[::-1])[::-1]

    # Convert to correct units given age is in Myr - each step should be * 10**6

    csfh *= 10**6

    age_idx = np.where(age_bins_mid > 100 * u.Myr)[0]
    mass_ratio = csfh[age_idx[0]] / csfh[0]

    ax.text(
        0.98,
        0.98,
        f"$F_{{>100 \ \\rm Myr}} = {mass_ratio:.2f}$",
        transform=ax.transAxes,
        fontsize=8,
        ha="right",
        va="top",
    )
    # Plot on same axis, on right hand side
    ax2 = ax.twinx()

    ax2.plot(age_bins_mid.to(u.Myr), csfh, color="r", linestyle="--")

    print(
        f"ID: {halo_id}, z={redshift}, Stellar mass formed: {np.log10(csfh[0])} M_sun"
    )

    ax2.set_ylabel(r"${\rm Stellar \ Mass \ Formed \ (M_{\odot})}$", color="r")

    # ax.vlines(100, 0, 1, color='k', linestyle='--')
    ax.set_xlabel(r"${\rm Lookback \ Time \ (Myr)}$")
    ax.set_ylabel(r"${\rm SFR/M_{\odot}yr^{-1}}$")

    plt.xscale("log")
    # plt.legend(fontsize=8)
    return fig, ax


def plot_spectra(direction, halo_id, redshift, fig=None, ax=None):
    if redshift not in spectra_cache:
        spectra_path = f"/raid/scratch/work/tharvey/SPHINX/SPHINX-20-data/data/spectra/all_spec_z{redshift:.0f}.json"

        with open(spectra_path, "r") as f:
            spectra = json.load(f)

        spectra_cache[redshift] = spectra

    else:
        spectra = spectra_cache[redshift]

    spectra_gal = spectra[str(halo_id)]
    wavelengths = np.array(spectra_gal["wavelengths"])

    if fig is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6), dpi=200)
    max = 0
    for pos, ftype in enumerate(
        ["total", "stellar_continuum", "nebular_continuum", "emission_line"]
    ):
        if not direction.startswith("dir_"):
            direction = f"dir_{direction}"
        flux = (
            10.0 ** np.array(spectra_gal[direction][ftype])
        )  # ['total', 'stellar_continuum', 'nebular_continuum', 'emission_line']
        ax.plot(
            wavelengths,
            flux,
            label=ftype,
            zorder=6 - pos,
            alpha=1 if pos == 0 else 0.5,
        )
        max = np.max([max, np.max(flux)])
        if pos == 0:
            average_flux = np.median(flux)

    ax.set_xlabel(r"${\rm Wavelength \ (\mu m)}$")
    ax.set_ylabel(r"${\rm Flux \ (erg/s/cm^2/Hz)}$")
    ax.set_yscale("log")
    ax.set_ylim(0.2 * average_flux, 1.2 * max)
    ax.legend()

    return fig, ax


def get_size(halo_id, redshift, size_band, direction="max"):
    halo_id = str(int(halo_id))
    if redshift not in size_cache:
        size_path = f"/raid/scratch/work/tharvey/SPHINX/SPHINX-20-data/data/galaxy_sizes/morph_z{int(redshift)}.json"
        with open(size_path, "r") as f:
            sizes = json.load(f)
        size_cache[redshift] = sizes
    else:
        sizes = size_cache[redshift]

    if direction == "max":
        keys = sizes[str(halo_id)][size_band].keys()
        max_size = 0
        max_direction = None
        for key in keys:
            size = sizes[str(halo_id)][size_band][key]["half"]
            if size > max_size:
                max_size = size
                max_direction = key
        size = max_size
        direction = max_direction
    else:
        direction = f"dir_{direction}"

    size = sizes[str(halo_id)][size_band][f"{direction}"]["half"]

    half_light_radius = size * u.kpc
    d_A = cosmo.angular_diameter_distance(redshift)
    half_light_radius_arcsec = (half_light_radius / d_A).to(
        u.arcsec, u.dimensionless_angles()
    )
    size = half_light_radius_arcsec.value

    return size, directio


def resize(img, out_pix_scale=0.03 * u.arcsec, in_pix_scale=0.03 * u.arcsec):
    """
    Resize an image to a new pixel scale using WCS reprojection.

    Parameters
    ----------
    img : numpy.ndarray
        Input image array
    out_pix_scale : astropy.units.Quantity
        Output pixel scale (default: 0.03 arcsec)
    in_pix_scale : astropy.units.Quantity
        Input pixel scale (default: 0.03 arcsec)

    Returns
    -------
    numpy.ndarray
        Resized image array
    """
    # Ensure input image is float type
    img = img.astype(float)

    # Create input WCS
    in_wcs = WCS(naxis=2)
    in_wcs.wcs.crpix = [
        img.shape[1] / 2.0,
        img.shape[0] / 2.0,
    ]  # Use floating point
    in_wcs.wcs.cdelt = [
        -in_pix_scale.to(u.deg).value,
        in_pix_scale.to(u.deg).value,
    ]
    in_wcs.wcs.crval = [0, 0]
    in_wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]  # Add projection type

    # Calculate output shape
    scale_factor = (in_pix_scale / out_pix_scale).value
    shape_out = np.ceil(np.array(img.shape) * scale_factor).astype(int)
    shape_out = (shape_out[0], shape_out[1])  # Ensure correct order

    # Create output WCS
    out_wcs = WCS(naxis=2)
    out_wcs.wcs.crpix = [
        shape_out[1] / 2.0,
        shape_out[0] / 2.0,
    ]  # Match new dimensions
    out_wcs.wcs.cdelt = [
        -out_pix_scale.to(u.deg).value,
        out_pix_scale.to(u.deg).value,
    ]
    out_wcs.wcs.crval = [0, 0]
    out_wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]  # Add projection type

    # Perform reprojection
    img_out, footprint = reproject_adaptive(
        (img, in_wcs),
        out_wcs,
        shape_out=shape_out,
        kernel="gaussian",
        conserve_flux=True,
        boundary_mode="constant",
        boundary_fill_value=0.0,
    )

    # Apply footprint mask to remove edge artifacts
    # img_out[footprint < 0.8] = np.nan

    return img_out


if __name__ == "__main__":
    generate_full_images(
        "/nvme/scratch/work/tharvey/SPHINX/z7_halo_62333", 62333, 7.0
    )
