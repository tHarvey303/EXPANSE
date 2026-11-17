"""Example: fit a single galaxy with a Pop III model using an explicit Bagpipes config.

Loads one ResolvedGalaxy from its .h5 file and fits its integrated photometry
with the Yggdrasil Pop III grid (Z = 0, Salpeter IMF 1-500 Msun, Cloudy nebular
emission), selected with meta["config"]. The config that Bagpipes actually used
is written to the posterior .h5 and to the meta of the EXPANSE results table.

Usage:
    python run_popiii_yggdrasil.py [galaxy.h5] [run_dir]

With no arguments the test galaxy shipped with EXPANSE is fitted (copied to a
scratch folder first, so the repository copy is left untouched). That galaxy
is at z ~ 4.5 and is not a Pop III candidate: it only demonstrates the
mechanics. Point the script at a real candidate for a meaningful fit.
"""

import os
import shutil
import sys

from EXPANSE import ResolvedGalaxy

PIPES_CONFIG = "yggdrasil_PopIII_salpeter_cloudy"

script_dir = os.path.dirname(os.path.abspath(__file__))

if len(sys.argv) > 1:
    galaxy_path = os.path.abspath(sys.argv[1])
else:
    # Work on a copy of the test galaxy: run_bagpipes saves results into the .h5.
    work_dir = os.path.join(os.getcwd(), "popiii_example")
    os.makedirs(work_dir, exist_ok=True)
    galaxy_path = os.path.join(work_dir, "GalaxyTest.h5")
    if not os.path.exists(galaxy_path):
        shutil.copy(os.path.join(script_dir, "../tests/galaxies/GalaxyTest.h5"), galaxy_path)

run_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(galaxy_path), "pipes/")

# Bagpipes writes its "pipes/" tree relative to the working directory.
os.makedirs(run_dir, exist_ok=True)
os.chdir(os.path.dirname(os.path.abspath(run_dir.rstrip("/"))))
run_dir = os.path.basename(run_dir.rstrip("/")) + "/"

galaxy = ResolvedGalaxy.init_from_h5(galaxy_path)
print(f"Loaded {galaxy.galaxy_id} at z = {galaxy.redshift:.2f}")

# A simple Pop III picture: a single, very young, dust-free burst.
#
# - The Pop III phase is short-lived, and the nebular grid only covers ages
#   below 30 Myr, so star formation is constant over the last 1-30 Myr
#   (1 Myr is the youngest age the Bagpipes model grid resolves).
# - t_bc is raised to 30 Myr so all of those stars power nebular emission,
#   which dominates the SED of such a hot, metal-free population.
# - The grid has a single stellar metallicity (Z = 0); the gas metallicity
#   (1e-3 Zsun) and density (1e3 cm^-3) are fixed by the grid.
# - No dust component: pristine gas is taken to be dust-free.
# - Masses are formed masses (the grid carries no stellar remnant fraction).
fit_instructions = {
    "t_bc": 0.03,
    "constant": {
        "age_max": (0.001, 0.03),
        "age_max_prior": "log_10",
        "age_min": 0.0,
        "massformed": (4.0, 10.0),
        "metallicity": 0.0,
    },
    "nebular": {"logU": (-3.0, -1.0)},
}

meta = {
    "run_name": "popIII_yggdrasil_cnst",
    # Explicit Bagpipes config, passed to bagpipes.set_config().
    "config": PIPES_CONFIG,
    # Fit the integrated photometry at the galaxy's stored redshift.
    "fit_photometry": "TOTAL_BIN",
    "redshift": "self",
    "sampler": "nautilus",
}

galaxy.run_bagpipes(
    {"meta": meta, "fit_instructions": fit_instructions},
    run_dir=run_dir,
    skip_single_bin=False,
)

table = galaxy.sed_fitting_table["bagpipes"][meta["run_name"]]
print("\nBagpipes config recorded with the results:")
for key in ["bagpipes_config", "bagpipes_stellar_file", "bagpipes_neb_line_file"]:
    print(f"  {key}: {table.meta.get(key)}")

print("\nPosterior medians:")
for column in ["constant:massformed_50", "constant:age_max_50", "nebular:logU_50", "chisq_phot"]:
    if column in table.colnames:
        print(f"  {column}: {table[column][0]:.3g}")
