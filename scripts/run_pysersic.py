from EXPANSE import ResolvedGalaxy, ResolvedGalaxies

galaxies = ResolvedGalaxies(
    ResolvedGalaxy.init_all_field_from_h5("JOF_psfmatched", n_jobs=6)
)


overwrite_ids = ["10130", "10161", "10230", "10376"]
for galaxy in galaxies:
    galaxy.add_psf_models(
        "/nvme/scratch/work/tharvey/EXPANSE/psfs/psf_models/star_stack/JOF_psfmatched/",
        file_ending="_psf.fits",
        overwrite=False,
    )
    galaxy.run_pysersic(
        show_plots=False,
        mask_type="seg_F444W",
        make_diagnostic_plots=True,
        save_plots=True,
        fit_type="sample",
        posterior_sample_method="median",
        overwrite=True if galaxy.galaxy_id in overwrite_ids else False,
        force=True,
    )

print("Done!")
