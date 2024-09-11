import webbpsf

nc = webbpsf.NIRCam()

nc.options["output_mode"] = "detector sampled"
nc.pixelscale = 0.03


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
    "F360M",
    "F410M",
    "F430M",
    "F444W",
]
filters = ["F360M"]
for i, filter in enumerate(filters):
    nc.filter = filter
    psf = nc.calc_psf(f"webbpsf/{filter}_PSF_003.fits", fov_pixels=623)
