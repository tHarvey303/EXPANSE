from EXPANSE import MockResolvedGalaxy
import numpy as np
from astropy.table import Table
from tqdm import tqdm

mock_survey = "JOF_psfmatched"
data_path = "/raid/scratch/work/tharvey/SPHINX/SPHINX-20-data/"
h5_folder = "/nvme/scratch/work/tharvey/EXPANSE/galaxies/mock_sphinx/"
overwrite = True

cat = Table.read(f"{data_path}/data/selected_galaxies.csv")


for row in tqdm(cat):
    direction = row["direction_max_jwst_f444w"]
    redshift = row["redshift"]
    halo_id = row["halo_id"]

    print(
        f"Generating galaxy {halo_id} at redshift {redshift} in direction {direction}"
    )

    images_dir = f"/nvme/scratch/work/tharvey/SPHINX/SPHINX_IMAGES/{halo_id}"

    # or just this to load the galaxy if it already exists
    galaxy = MockResolvedGalaxy.init(
        gal_id=halo_id,
        redshift_code=redshift,
        direction=direction,
        h5_folder=h5_folder,
        images_dir=images_dir,
        mock_survey=mock_survey,
    )

    galaxy.sep_process(
        debug=False, overwrite=overwrite, combine_all_regions=True
    )

    galaxy.pixedfit_processing(
        gal_region_use="detection",
        overwrite=overwrite,
        dir_images=h5_folder,
    )
    # Maybe seg map should be from detection image?
    galaxy.pixedfit_binning(overwrite=overwrite)
    galaxy.measure_flux_in_bins(overwrite=overwrite)

    galaxy.eazy_fit_measured_photometry(
        "MAG_APER_0.32 arcsec",
        update_meta_properties=True,
        overwrite=overwrite,
        exclude_bands=["F435W", "F606W", "F775W", "F814W", "F850LP"],
    )
    galaxy.plot_overview(
        bands_to_show=["F090W", "F277W", "F300M", "F444W"], save=True
    )

# to generate the galaxy
"""
galaxy = MockResolvedGalaxy.init_mock_from_sphinx(halo_id=halo_id, redshift=redshift, 
                                                direction=direction, images_dir=images_dir,
                                                h5_folder = h5_folder, mock_survey=mock_survey)
"""
