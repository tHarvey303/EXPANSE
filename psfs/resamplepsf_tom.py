import scipy.ndimage
from astropy.io import fits

flist = [
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
    "F360M",
    "F410M",
    "F430M",
    "F444W",
    "F460M",
    "F480M",
]


for f in flist:
    hdul2 = fits.open("originalPSF/PSF_%scen_G5V_fov299px_ISIM41.fits" % f)
    imgdata = hdul2[0].data
    origres = 0.01575
    newres = 0.03
    factor = origres / newres
    print(factor)
    newpsf = scipy.ndimage.zoom(imgdata, factor, order=2)
    print(imgdata.shape)
    print(newpsf.shape)
    header = fits.Header()
    header["PIXSCALE"] = newres

    hdu = fits.PrimaryHDU(newpsf, header=header)
    hdul = fits.HDUList([hdu])

    hdul.writeto("TomPSFs/PSF_Resample_03_%s.fits" % f, overwrite=True)
