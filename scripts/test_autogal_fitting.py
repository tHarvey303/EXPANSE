from EXPANSE import ResolvedGalaxy

galaxy = ResolvedGalaxy.init_from_h5("JOF_psfmatched_15021.h5")

print(galaxy.im_pixel_scales)
galaxy.run_autogalaxy(
    model_type="sersic", band="F444W", mask_type="circular", mask_radius=0.95
)
