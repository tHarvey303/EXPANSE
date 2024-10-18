from EXPANSE import ResolvedGalaxy, MultipleResolvedGalaxy
from EXPANSE.bagpipes.pipes_models import (
    continuity_dict,
    create_dicts,
    delayed_dict,
    dpl_dict,
    lognorm_dict,
    resolved_dict,
)


n_jobs = 4
field = "JOF_psfmatched"
galaxies_dir = "/nvme/scratch/work/tharvey/EXPANSE/galaxies/"
run_dir = "/nvme/scratch/work/tharvey/EXPANSE/pipes/"
fit_photometry = "TOTAL_BIN"
model = resolved_dict  # This is the model we are using
meta = {"use_bpass": True}
model["meta"]["run_name"] = "CNST_SFH_RESOLVED_P"

galaxies = ResolvedGalaxy.init_all_field_from_h5(
    field, galaxies_dir, save_out=False
)

multiple_galaxies = MultipleResolvedGalaxy(galaxies)

dicts = create_dicts(model, len(multiple_galaxies), override_meta=meta)

multiple_galaxies.run_bagpipes_parallel(
    dicts, n_jobs=n_jobs, fit_photometry=fit_photometry, run_dir=run_dir
)
