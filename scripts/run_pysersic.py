from EXPANSE import ResolvedGalaxy, ResolvedGalaxies
from tqdm import tqdm

galaxies = ResolvedGalaxies(ResolvedGalaxy.init_all_field_from_h5("JOF_psfmatched", n_jobs=6))

overwrite_ids = []

for galaxy in tqdm(galaxies):
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
        prior_dict={
            "xc": {
                "type": "uniform",
                "low": -3,
                "high": 3,
                "relative_to_centre": True,
            },
            "yc": {
                "type": "uniform",
                "low": -3,
                "high": 3,
                "relative_to_centre": True,
            },
            "r_eff": {
                "type": "truncated_gaussian",
                "loc": 6.16,
                "scale": 4.96,
                "low": 0.5,
                "high": 10.0,
            },
        },
    )

print("Done!")
