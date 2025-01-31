import astropy.units as u
import numpy as np
import os
import sys

os.environ["PIXEDFIT_HOME"] = "/nvme/scratch/work/tharvey/piXedfit/"
from EXPANSE import ResolvedGalaxy, ResolvedGalaxies
from EXPANSE.bagpipes.pipes_models import (
    resolved_dict_cnst,
    resolved_dict_bursty,
    create_dicts,
)
from matplotlib import pyplot as plt
import glob
from scipy import signal
from scipy.interpolate import interp1d
from scipy.stats import binned_statistic
# Change dpi to make plots larger

plt.rcParams["figure.dpi"] = 100

# Disable tex in matplotlib

plt.rcParams["text.usetex"] = False

file_path = os.getcwd()

if os.path.exists("/.singularity.d/Singularity"):
    computer = "singularity"
elif "nvme" in file_path:
    computer = "morgan"
elif "Users" in file_path:
    computer = "mac"
else:
    computer = "unknown"

if computer == "morgan":
    galaxies_dir = "/nvme/scratch/work/tharvey/EXPANSE/galaxies/mock/"
    bagpipes_run_dir = "/nvme/scratch/work/tharvey/EXPANSE/pipes/"
elif computer == "singularity":
    galaxies_dir = "/mnt/galaxies/mock/"
    bagpipes_run_dir = "/mnt/pipes/"


# %matplotlib inline

galaxies = ResolvedGalaxies(
    ResolvedGalaxy.init_all_field_from_h5(
        "JOF_psfmatched",
        n_jobs=6,
    )
)
bagpipes_runs = [
    "photoz_lognorm",
    "photoz_delayed",
    "photoz_dpl",
    "photoz_continuity",
]
binmap_type = "pixel_by_pixel"  # "pixel_by_pixel"
fit_photometry = "bin"

bagpipes_only = True  # This is for running Bagpipes only if the galaxies have already been created
load_only = False  # This is for running Bagpipes - whether to skip running fitting and load existing results

try:
    n_jobs = int(sys.argv[1])
except:
    n_jobs = 6

ids = [
    "10092",
    "10130",
    "10161",
    "10376",
    "10816",
    "10896",
    "11482",
    "11600",
    "12050",
    "12488",
    "12779",
    "12816",
    "13892",
    "1623",
    "1951",
    "2015",
    "2439",
    "2643",
    "2994",
    "3398",
    "3434",
    "3845",
    "3908",
    "4234",
    "4266",
    "531",
    "6074",
    "615",
    "6744",
    "7155",
    "7526",
    "7687",
    "845",
    "8854",
    "906",
    "9078",
]

"""
remove = []
for galaxy in galaxies:
    galaxy.pixel_by_pixel_galaxy_region(snr_req=2, band_req='all_wide_nobreak', mask='detection')
    galaxy.pixel_by_pixel_binmap(galaxy.gal_region['SNR_2_all_wide_nobreak'])
    if galaxy.get_number_of_bins(binmap_type) == 0:
        print(f"Galaxy {galaxy.galaxy_id} has no bins")
        remove.append(galaxy.galaxy_id)
    else:
        galaxy.measure_flux_in_bins(binmap_type=binmap_type, overwrite=False)


galaxies_lowmass = galaxies_lowmass.filter_IDs(remove, invert=True)
"""


"""
table = galaxies.save_to_fits(save=False)


delta_masses = {}
masses = {}

for bagpipes_run in bagpipes_runs:
    if bagpipes_run == "photoz_dpl":
        # Something weird going on with this run, skip it for now
        continue
    delta_mass = (
        table["CNST_SFH_RESOLVED_resolved_mass"][:, 1]
        - table[f"{bagpipes_run}_stellar_mass_50"]
    )
    delta_masses[bagpipes_run] = delta_mass
    masses[bagpipes_run] = table[f"{bagpipes_run}_stellar_mass_50"]

# Select outshined galaxies

# select photoz_lognorm which have delta_masses > 0.2 dex

select_pos = np.zeros(len(galaxies), dtype=bool)
select_pos2 = np.zeros(len(galaxies), dtype=bool)

for key in delta_masses.keys():
    select_pos = select_pos | (delta_masses[key] > 0.2)
    select_pos2 = select_pos2 | ((masses[key] < 8) & (delta_masses[key] < 0.2))

# Remove overlap with select_pos2
# Run either.
select_pos2 = select_pos2 | select_pos

galaxy_ids = table["galaxy_id"][select_pos2]

galaxies_outshined = ResolvedGalaxies(
    [galaxy for galaxy in galaxies if galaxy.galaxy_id in galaxy_ids]
)

resolved_run = "CNST_SFH_RESOLVED"
# filter out single bins
galaxies = galaxies.filter_single_bins("pixedfit")

"""
resolved_run = "CNST_SFH_RESOLVED"

masses = []
for galaxy in galaxies:
    try:
        mass = galaxy.get_total_resolved_property(resolved_run)[1]
        masses.append(mass)
    except:
        masses.append(0)

galaxies_lowmass = galaxies[np.array(masses) < 9]


"""
for galaxy in galaxies_outshined:
    print(galaxy.galaxy_id)
    galaxy.pixedfit_processing(gal_region_use="detection", overwrite=False)
    galaxy.pixedfit_binning(
        save_out=True,
        SNR_reqs=4,
        Dmin_bin=1,
        redc_chi2_limit=100.0,
        del_r=1.0,
        overwrite=False,
        name_out=binmap_type,
    )
    galaxy.measure_flux_in_bins(binmap_type=binmap_type, overwrite=False)
"""
# Now we have the pixedfit results for the outshined galaxies, we can rerun the bagpipes fits

override_meta_resolved = {
    "use_bpass": True,
    "run_name": "CNST_SFH_RESOLVED_PBP",
}
resolved_dicts_cnst = create_dicts(
    resolved_dict_cnst,
    num=len(galaxies_lowmass),
    override_meta=override_meta_resolved,
)

override_meta_resolved = {
    "use_bpass": True,
    "run_name": "BURSTY_SFH_RESOLVED_PBP",
}

resolved_dicts_bursty = create_dicts(
    resolved_dict_bursty,
    num=len(galaxies_lowmass),
    override_meta=override_meta_resolved,
)

for dicts in [
    resolved_dicts_cnst,
    # resolved_dicts_bursty,
]:
    galaxies_lowmass.run_bagpipes_parallel(
        dicts,
        n_jobs=n_jobs,
        fit_photometry=fit_photometry,
        run_dir=bagpipes_run_dir,
        load_only=load_only,
        override_binmap_type=binmap_type,
    )
