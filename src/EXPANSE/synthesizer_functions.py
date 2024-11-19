import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM
from unyt import Msun, Myr, unyt_array, yr
import copy

cosmo = FlatLambdaCDM(H0=70, Om0=0.3)


def convert_coordinates(coordinates, redshift, pixel_scale=0.03 * u.arcsecond):
    """

    Convert physical coordinates to pixel coordinates

    """
    d_A = cosmo.angular_diameter_distance(redshift)
    coords_arcsec = (coordinates / d_A).to(u.arcsec, u.dimensionless_angles())
    coords_pixels = (coords_arcsec / pixel_scale).value
    return coords_pixels


def apply_pixel_coordinate_mask(
    gal, pixel_mask, pixel_scale=0.03 * u.arcsecond
):
    coords = gal.stars.centered_coordinates.to_astropy().to(u.kpc)

    coords = convert_coordinates(coords, gal.redshift)

    coords_2d_pixels = coords[:, [1, 0]]  # x, y

    coords_2d_pixels[:, 0] = (
        coords_2d_pixels[:, 0] + np.shape(pixel_mask)[1] / 2
    )
    coords_2d_pixels[:, 1] = (
        coords_2d_pixels[:, 1] + np.shape(pixel_mask)[0] / 2
    )

    # print(coords_2d_pixels)

    # Bin coordinates into grid of shape pixel_mask.shape with the same pixel scale
    # Calculate which pixel each coordinate belongs to
    x_bins = (
        np.digitize(
            coords_2d_pixels[:, 0],
            np.linspace(0, pixel_mask.shape[0], pixel_mask.shape[0] + 1),
        )
        - 1
    )
    y_bins = (
        np.digitize(
            coords_2d_pixels[:, 1],
            np.linspace(0, pixel_mask.shape[1], pixel_mask.shape[1] + 1),
        )
        - 1
    )

    masks = []
    for i, j in zip(x_bins, y_bins):
        if (
            i < 0
            or i >= pixel_mask.shape[0]
            or j < 0
            or j >= pixel_mask.shape[1]
        ):
            masks.append(False)
        else:
            masks.append(pixel_mask[j, i])
            # print(i, j, pixel_mask[j, i])

    return np.array(masks, dtype=bool)


def get_spectra_in_mask(
    gal,
    spectra_type="total",
    aperture_mask_radii=None,
    pixel_mask=None,
    pixel_scale=0.03 * u.arcsecond,
    pixel_mask_value=None,
):
    pixel_mask = copy.deepcopy(pixel_mask)

    if aperture_mask_radii is not None and pixel_mask is not None:
        raise ValueError(
            "Must provide only one of aperture_mask or pixel_mask"
        )
    elif aperture_mask_radii is None and pixel_mask is None:
        raise ValueError("Must provide either aperture_mask or pixel_mask")

    assert (
        type(aperture_mask_radii) is type(unyt_array)
        or type(aperture_mask_radii) is u.Quantity
    ) or type(pixel_mask) is np.ndarray

    if aperture_mask_radii is not None:
        coords = gal.stars.centered_coordinates.to_astropy().to(u.kpc)

        if type(aperture_mask_radii) is u.Quantity:
            if aperture_mask_radii.unit == u.arcsec:
                aperture_mask_radii /= pixel_scale
            # Convert to pixels if not providing a unyt quantity
            coords = convert_coordinates(
                coords, gal.redshift, pixel_scale=pixel_scale
            )

        mask = coords[:, 0] ** 2 + coords[:, 1] ** 2 < aperture_mask_radii**2

    if pixel_mask is not None:
        if pixel_mask_value is not None:
            print("Applying pixel mask with value", pixel_mask_value)
            # if only 0 and 1s, just use 1
            if np.all(np.unique(pixel_mask) == [0, 1]):
                print("Boolean mask... using 1 as True.")
                pixel_mask[pixel_mask == 1] = True
                pixel_mask[pixel_mask == 0] = False

            elif pixel_mask_value == "center":
                # This option is for e.g. a segmentation map with multiple regions.
                # Get value at the center of the mask
                center = np.array(pixel_mask.shape) / 2
                center_val = pixel_mask[int(center[0]), int(center[1])]
                # Create boolean mask first, then assign values
                mask = pixel_mask == center_val
                pixel_mask = mask.astype(bool)
                print(
                    "Center value:",
                    center_val,
                    "with",
                    np.sum(pixel_mask),
                    "unmasked pixels",
                )

            elif type(pixel_mask_value) in [int, float]:
                pixel_mask[pixel_mask == pixel_mask_value] = True
                pixel_mask[pixel_mask != pixel_mask_value] = False
            else:
                raise ValueError(
                    "pixel_mask_value must be 'center' or a number"
                )

        mask = apply_pixel_coordinate_mask(gal, pixel_mask)

        if np.sum(mask) == 0:
            print("WARNING! Mask is all False.")

    spectra_mask = gal.stars.particle_spectra["total"].fnu[mask]
    spectra_mask_total = np.sum(spectra_mask, axis=0)

    return spectra_mask_total


def calculate_sfh(galaxy, binw=10 * Myr, pixel_mask=None, plot=False):
    if pixel_mask is not None:
        mask = apply_pixel_coordinate_mask(galaxy, pixel_mask)
    else:
        mask = np.ones(len(galaxy.stars.ages), dtype=bool)

    max_age = galaxy.stars.ages[mask].max()
    bins = np.arange(0.0 * Myr, max_age, binw)
    binc = 0.5 * (bins[:-1] + bins[1:])

    ages = galaxy.stars.ages.to(Myr)[mask]
    masses = galaxy.stars.initial_masses[mask]
    sorted_indexes = np.argsort(ages)

    ages = ages[sorted_indexes]
    masses = masses[sorted_indexes]

    # Calculate the SFR on the bins grid

    sfr = np.zeros(len(bins) - 1) * Msun / yr

    # Count backwards, so that the oldest stars are added first, and only once
    for i in range(len(ages) - 1, 0, -1):
        age = ages[i]
        mass = masses[i]
        for j in range(len(bins) - 1):
            if bins[j] < age < bins[j + 1]:
                sfr[j] += mass / binw.to(yr)

    if plot:
        fig, ax = plt.subplots()
        ax.plot(binc, sfr.to(Msun / yr))
        ax.set_xlabel("Time (Myr)")
        ax.set_ylabel(r"SFR ($M_{\odot}/yr)$")

        plt.show()

    binc *= Myr

    return binc, sfr


def plot_particle_sed(
    gal,
    spec_type="total",
    filters=[],
    fig=None,
    ylim=(30, 26),
    xlim=(0, 52000),
):
    from unyt import nJy

    if fig is None:
        fig, ax = plt.subplots(
            nrows=2, ncols=1, height_ratios=[3, 1], dpi=200, facecolor="white"
        )
        ax, ax_filt = ax

    else:
        gs = fig.add_gridspec(2, 1, height_ratios=[3, 1])
        ax = fig.add_subplot(gs[0])
        ax_filt = fig.add_subplot(gs[1])

    if spec_type == "all":
        spec_type = gal.stars.spectra.keys()
    elif type(spec_type) == str:
        spec_type = [spec_type]
    for s in spec_type:
        sed = gal.stars.spectra[s]
        phot = gal.stars.photo_fnu[s]
        ax.plot(sed.obslam, -2.5 * np.log10(sed.fnu) + 31.40, label=s)
        ax.scatter(
            filters.pivot_lams.ndarray_view(),
            -2.5 * np.log10(phot.photo_fnu.to(nJy)) + 31.40,
            marker="o",
            color="black",
            facecolors="none",
            zorder=10,
        )

    ax.set_xlim(xlim[0], xlim[1])
    ax_filt.set_xlim(xlim[0], xlim[1])

    ax.set_xticks([])
    ax.set_ylim(ylim[0], ylim[1])

    filters.plot_transmission_curves(ax=ax_filt)
    ax_filt.set_xlabel("Wavelength (Angstrom)")
    ax.set_ylabel("Flux (AB mag)")
    # Remove upper and right axes
    ax_filt.spines["top"].set_visible(False)
    ax.spines["bottom"].set_alpha(0.2)

    fig.subplots_adjust(hspace=0.01)
    # No legends
    ax_filt.legend().remove()
    ax_filt.set_ylim(0.01, None)
    ax_filt.set_yticklabels("")
    ax_filt.set_yticks([])

    text = "\n".join(
        (
            f"$z = {gal.redshift:.2f}$",
            f"$M_* = $ {gal.stellar_mass.value:.2e} ${gal.stellar_mass.units.latex_repr}$",
            f"Age = {gal.stellar_mass_weighted_age:.2e}",
            rf"$\overline{{\tau_V}} = ${gal.tau_v.mean():.2e}",
        )
    )
    props = dict(boxstyle="square", facecolor="lightgrey", alpha=0.5)
    ax.text(
        0.01,
        0.96,
        text,
        transform=ax.transAxes,
        fontsize=7,
        verticalalignment="top",
        bbox=props,
    )

    if len(spec_type) > 1:
        ax.legend(frameon=False)
    return fig
