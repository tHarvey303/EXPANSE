from EXPANSE import ResolvedGalaxy, ResolvedGalaxies
from tqdm import tqdm

galaxies = ResolvedGalaxies(
    ResolvedGalaxy.init_all_field_from_h5("JOF_psfmatched", n_jobs=6)
)


overwrite_ids = [
    "295",
    "1542",
    "1623",
    "1685",
    "1991",
    "2398",
    "2491",
    "2521",
    "3434",
    "3823",
    "4990",
    "5001",
    "5114",
    "5532",
    "5990",
    "6788",
    "6882",
    "7687",
    "8514",
    "9071",
    "9078",
    "9091",
    "10161",
    "10484",
    "10816",
    "11072",
    "11358",
    "11825",
    "13097",
    "13322",
    "13330",
    "13344",
    "13434",
    "14194",
    "14708",
    "14940",
    "15099",
    "15149",
]

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
