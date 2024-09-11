from astropy.io import fits
from astropy.convolution import convolve_fft
import matplotlib.pyplot as plt
from photutils.aperture import CircularAperture
from photutils.centroids import centroid_2dg
import numpy as np

filters = [
    "F090W",
    "F115W",
    "F150W",
    "F162M",
    "F182M",
    "F200W",
    "F210M",
    "F250M",
    "F277W",
    "F300M",
    "F335M",
    "F356W",
    "F410M",
    "F430M",
]
cmap = plt.get_cmap("rainbow")
fig, ax = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

ax[0].set_title("Before convolution")
ax[1].set_title("After convolution")
fig.suptitle("Encircled Energy")
fig.subplots_adjust(wspace=0)
radii = np.linspace(1, 1.5 / 0.03, 50)
radii_arcsec = radii * 0.03

psf_f444w = fits.open("webbpsf/F444W_PSF_003.fits")[0].data
center = centroid_2dg(psf_f444w)
flux_f444w = np.array(
    [
        aperture.do_photometry(psf_f444w)[0]
        for aperture in [CircularAperture(center, r=radius) for radius in radii]
    ]
)

# print('F444W', np.sum(psf_f444w))
for i, filter in enumerate(filters):
    psf = fits.open(f"webbpsf/{filter}_PSF_003.fits")[0].data
    kernel = fits.open(f"kernel_003_{filter}toF444W.fits")[0].data
    size = psf.shape[0]
    print("Filter:", filter, "Sum:", np.sum(psf))
    center_old = (size - 1) / 2
    print(center_old, center)
    center = centroid_2dg(psf)

    encircled_energy_before = []
    print(kernel.shape, psf.shape)
    psf_convolved = convolve_fft(psf, kernel, normalize_kernel=True)
    encircled_energy_after = []
    for radius in radii:
        aperture = CircularAperture(center, r=radius)
        encircled_energy_before.append(aperture.do_photometry(psf)[0])
        encircled_energy_after.append(aperture.do_photometry(psf_convolved)[0])
    encircled_energy_before = np.array(encircled_energy_before) / np.sum(psf)
    encircled_energy_after = np.array(encircled_energy_after) / np.sum(psf_convolved)

    ax[0].plot(
        radii_arcsec,
        encircled_energy_before,
        label=filter,
        color=cmap(i / len(filters)),
    )
    ax[1].plot(
        radii_arcsec, encircled_energy_after, label=filter, color=cmap(i / len(filters))
    )

# F444W
ax[0].plot(radii_arcsec, flux_f444w / np.sum(psf_f444w), label="F444W", color="black")
ax[1].plot(radii_arcsec, flux_f444w / np.sum(psf_f444w), label="F444W", color="black")

ax[0].set_xlabel("Radius (arcsec)")
ax[0].set_ylabel("Encircled Energy")
ax[1].set_xlabel("Radius (arcsec)")
ax[0].legend(ncol=3, loc="lower right", columnspacing=0.75, frameon=False)
fig.savefig("encircled_energy.png")
