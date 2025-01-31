from EXPANSE import ResolvedGalaxy, ResolvedGalaxies
from EXPANSE.dense_basis import get_priors, make_db_grid
from tqdm import tqdm

binmap_type = "voronoi"
regenerate_db_grid = True

galaxies = ResolvedGalaxies(
    ResolvedGalaxy.init_all_field_from_h5("JOF_psfmatched", n_jobs=6)
)

bands = galaxies[0].bands
if regenerate_db_grid:
    db_atlas_path = make_db_grid(bands, N_pregrid=500000, N_sfh_priors=3)
else:
    db_atlas_path = f"/nvme/scratch/work/tharvey/EXPANSE/scripts/pregrids/db_atlas_JOF_10000_Nparam_3.dbatlas"

priors = get_priors(db_atlas_path)

ok_galaxies = galaxies.filter_single_bins(binmap=binmap_type)

for galaxy in tqdm(ok_galaxies):
    fit_results = galaxy.run_dense_basis(
        db_atlas_path,
        plot=True,
        fit_photometry="TOTAL_BIN+bin",
        fix_redshift="photoz_delayed",
        binmap_type=binmap_type,
        use_emcee=False,
        priors=priors,
        n_jobs=6,
        save_outputs=True,
        overwrite=True,
        save_full_posteriors=False,
    )
