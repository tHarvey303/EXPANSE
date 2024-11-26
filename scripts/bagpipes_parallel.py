from EXPANSE import ResolvedGalaxy, ResolvedGalaxies
from EXPANSE.bagpipes.pipes_models import (
    continuity_dict,
    continuity_bursty_dict,
    create_dicts,
    delayed_dict,
    dpl_dict,
    cnst_dict,
    lognorm_dict,
    resolved_dict_cnst,
    resolved_dict_bursty,
)
import sys
import os
# try and get n_jobs from args

file_path = os.path.abspath(__file__)

if os.path.exists("/.singularity.d/Singularity"):
    computer = "singularity"
elif "nvme" in file_path:
    computer = "morgan"
elif "Users" in file_path:
    computer = "mac"
else:
    computer = "unknown"

try:
    n_jobs = sys.argv[1]
except:
    n_jobs = 4

field = "JOF_psfmatched"
load_only = False

if computer == "morgan":
    galaxies_dir = "/nvme/scratch/work/tharvey/EXPANSE/galaxies/"
    run_dir = "/nvme/scratch/work/tharvey/EXPANSE/pipes/"
elif computer == "singularity":
    galaxies_dir = "/mnt/galaxies/"
    run_dir = "/mnt/pipes/"

fit_photometry = "TOTAL_BIN"
model = cnst_dict  # This is the model we are using
meta = {"use_bpass": True}

dicts = create_dicts(model, len(multiple_galaxies), override_meta=meta)

second_model = continuity_bursty_dict

continuity_bursty_dicts = create_dicts(
    second_model,
    len(multiple_galaxies),
    override_meta={"use_bpass": True, "update_cont_bins": True},
)

galaxies = ResolvedGalaxy.init_all_field_from_h5(
    field, galaxies_dir, save_out=False
)

multiple_galaxies = ResolvedGalaxies(galaxies)


multiple_galaxies.run_bagpipes_parallel(
    dicts,
    n_jobs=n_jobs,
    fit_photometry=fit_photometry,
    run_dir=run_dir,
    load_only=load_only,
)

multiple_galaxies.run_bagpipes_parallel(
    continuity_bursty_dicts,
    n_jobs=n_jobs,
    fit_photometry=fit_photometry,
    run_dir=run_dir,
    load_only=load_only,
)
