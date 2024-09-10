from astropy import units as u
from astropy.cosmology import FlatLambdaCDM
import numpy as np
from unyt import unyt_array, Msun, yr, Myr, Gyr
import matplotlib.pyplot as plt


cosmo = FlatLambdaCDM(H0 = 70, Om0 = 0.3)

def convert_coordinates(coordinates, redshift, pixel_scale = 0.03 * u.arcsecond):
    '''

    Convert physical coordinates to pixel coordinates

    '''
    d_A = cosmo.angular_diameter_distance(redshift)
    coords_arcsec = (coordinates / d_A).to(u.arcsec, u.dimensionless_angles())
    coords_pixels = (coords_arcsec / pixel_scale).value
    return coords_pixels

def apply_pixel_coordinate_mask(gal, pixel_mask, pixel_scale = 0.03 * u.arcsecond):
    coords = gal.stars.centered_coordinates.to_astropy().to(u.kpc)

    coords = convert_coordinates(coords, gal.redshift)

    coords_2d_pixels = coords[:, [1, 0]] # x, y 

    coords_2d_pixels[:, 0] = coords_2d_pixels[:, 0] + np.shape(pixel_mask)[1] / 2
    coords_2d_pixels[:, 1] = coords_2d_pixels[:, 1] + np.shape(pixel_mask)[0] / 2

    #print(coords_2d_pixels)


    # Bin coordinates into grid of shape pixel_mask.shape with the same pixel scale
    # Calculate which pixel each coordinate belongs to
    x_bins = np.digitize(coords_2d_pixels[:, 0], np.linspace(0, pixel_mask.shape[0], pixel_mask.shape[0] + 1)) - 1
    y_bins = np.digitize(coords_2d_pixels[:, 1], np.linspace(0, pixel_mask.shape[1], pixel_mask.shape[1] + 1)) - 1

    masks = []
    for i, j in zip(x_bins, y_bins):
        if i < 0 or i >= pixel_mask.shape[0] or j < 0 or j >= pixel_mask.shape[1]:
            masks.append(False)
        else:
            masks.append(pixel_mask[j, i])
            #print(i, j, pixel_mask[j, i])

    return np.array(masks, dtype = bool)

def get_spectra_in_mask(gal, spectra_type = 'total', aperture_mask_radii = None, pixel_mask = None, pixel_scale = 0.03 * u.arcsecond):
    if aperture_mask_radii is not None and pixel_mask is not None:
        raise ValueError('Must provide only one of aperture_mask or pixel_mask')
    elif aperture_mask_radii is None and pixel_mask is None:
        raise ValueError('Must provide either aperture_mask or pixel_mask')

    assert (type(aperture_mask_radii) == type(unyt_array) or type(aperture_mask_radii) == u.Quantity) or type(pixel_mask) == np.ndarray

    if aperture_mask_radii is not None:
        coords = gal.stars.centered_coordinates.to_astropy().to(u.kpc)

        if type(aperture_mask_radii) == u.Quantity:
            if aperture_mask_radii.unit == u.arcsec:
                aperture_mask_radii /= pixel_scale
            # Convert to pixels if not providing a unyt quantity
            coords = convert_coordinates(coords, gal.redshift, pixel_scale = pixel_scale)

        mask = coords[:, 0]**2 + coords[:, 1]**2 < aperture_mask_radii**2

    if pixel_mask is not None:
        mask = apply_pixel_coordinate_mask(gal, pixel_mask)
    
    spectra_mask = gal.stars.particle_spectra['total'].fnu[mask]
    spectra_mask_total = np.sum(spectra_mask, axis = 0)

    return spectra_mask_total


def calculate_sfh(galaxy, binw = 5 * Myr, pixel_mask = None, plot = False):
    
    if pixel_mask is not None:
        mask = apply_pixel_coordinate_mask(galaxy, pixel_mask)
    else:
        mask = np.ones(len(galaxy.stars.ages), dtype=bool)

    max_age = galaxy.stars.ages[mask].max()
    bins = np.arange(0.0 * Myr, max_age, binw) 
    binc = 0.5*(bins[:-1]+bins[1:])

    ages = galaxy.stars.ages.to(Myr)[mask]
    masses = galaxy.stars.initial_masses[mask]
    sorted_indexes = np.argsort(ages)

    ages = ages[sorted_indexes]
    masses = masses[sorted_indexes]

    # Calculate the SFR on the bins grid

    sfr = np.zeros(len(bins)-1) * Msun/yr

    # Count backwards, so that the oldest stars are added first, and only once
    for i in range(len(ages)-1, 0, -1):
        age = ages[i]
        mass = masses[i]
        for j in range(len(bins)-1):
            if bins[j] < age < bins[j+1]:
                sfr[j] += mass / binw.to(yr)



    if plot:
        fig, ax = plt.subplots()
        ax.plot(binc, sfr.to(Msun/yr))
        ax.set_xlabel('Time (Myr)')
        ax.set_ylabel('SFR ($M_{\odot}/yr)$')

        plt.show()

    binc *= Myr

    return binc, sfr
