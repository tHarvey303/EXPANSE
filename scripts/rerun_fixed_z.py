from EXPANSE import ResolvedGalaxy, ResolvedGalaxies
from EXPANSE.bagpipes import create_dicts, continuity_dict, resolved_dict_cnst
import sys
import os

try:
    n_jobs = sys.argv[1]
except IndexError:
    n_jobs = 6

# Get path of this file
file_path = os.path.abspath(__file__)
db_dir = ""

if os.path.exists("/.singularity.d/Singularity"):
    computer = "singularity"
elif "nvme" in file_path:
    computer = "morgan"
elif "Users" in file_path:
    computer = "mac"
else:
    computer = "unknown"


if computer == "morgan":
    galaxies_dir = "/nvme/scratch/work/tharvey/EXPANSE/galaxies/"
    run_dir = "/nvme/scratch/work/tharvey/EXPANSE/pipes/"
elif computer == "singularity":
    galaxies_dir = "/mnt/galaxies/"
    run_dir = "/mnt/pipes/"

binmap_type = "voronoi"
load_only = False
fit_photometry = "bin"

galaxies = ResolvedGalaxy.init_all_field_from_h5("JOF_psfmatched", galaxies_dir, n_jobs=6)

galaxies = ResolvedGalaxies(galaxies)

# Select the galaxies that are at deltz = 0.2

filtered_galaxies = galaxies.filter_single_bins(binmap_type)


override_meta = {
    "redshift": "MAG_APER_0_32 arcsec_fsps_larson_0_05__Asada24_cgm",
    "redshift_key": "zbest",
    "use_bpass": True,
    "name_append": "_zfix_eazy_cgm",
    "remove": ["redshift_sigma", "min_redshift_sigma"],
}

bagpipes_configs = create_dicts(
    continuity_dict, override_meta=override_meta, num=len(filtered_galaxies)
)

filtered_galaxies.run_bagpipes_parallel(
    bagpipes_configs,
    n_jobs=n_jobs,
    fit_photometry=fit_photometry,
    run_dir=run_dir,
    load_only=load_only,
)

override_meta_resolved = {
    "use_bpass": True,
    "run_name": "CNST_SFH_RESOLVED_VORONOI_Asada24_cgm",
    "redshift": "MAG_APER_0_32 arcsec_fsps_larson_0_05__Asada24_cgm",
    "redshift_key": "zbest",
}

# Create a dictionary for the constant redshift model
resolved_dicts_cnst = resolved_dict_cnst(
    override_meta=override_meta_resolved, num=len(filtered_galaxies)
)

for dicts in [
    resolved_dicts_cnst,
]:
    galaxies.run_bagpipes_parallel(
        dicts,
        n_jobs=n_jobs,
        fit_photometry=fit_photometry,
        run_dir=run_dir,
        load_only=load_only,
        override_binmap_type=binmap_type,
    )
