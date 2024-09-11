import copy
import astropy.units as u
from .plotpipes import calculate_bins


# First fit
sfh = {}
sfh_type = "delayed"
sfh["tau"] = (0.01, 15)  # `Gyr`
sfh["massformed"] = (5.0, 12.0)  # Log_10 total stellar mass formed: M_Solar

sfh["age"] = (0.001, 15)  # Gyr
sfh["age_prior"] = "uniform"
sfh["metallicity_prior"] = "uniform"
sfh["metallicity"] = (1e-3, 2.5)

dust = {}
dust["type"] = "Calzetti"
dust["Av"] = (0, 5.0)

nebular = {}
nebular["logU"] = (-3.0, -1.0)

fit_instructions_delayed = {
    "t_bc": 0.01,
    sfh_type: sfh,
    "nebular": nebular,
    "dust": dust,
}

meta_delayed = {
    "run_name": "photoz_delayed",
    "redshift": "self",
    "redshift_sigma": "min",
    "min_redshift_sigma": 0.5,
    "fit_photometry": "TOTAL_BIN",
    "sampler": "multinest",
}

overall_dict = {"meta": meta_delayed, "fit_instructions": fit_instructions_delayed}

delayed_dict = copy.deepcopy(overall_dict)

# -------------------------------------------------------------------------------------
# Second fit
sfh = "continuity"
continuity = {}
continuity["massformed"] = (5.0, 12.0)  # Log_10 total stellar mass formed: M_Solar
continuity["metallicity"] = (1e-3, 2.5)
cont_nbins = 6
first_bin = 10 * u.Myr
second_bin = None
continuity["bin_edges"] = list(
    calculate_bins(
        redshift=8,
        num_bins=cont_nbins,
        first_bin=first_bin,
        second_bin=second_bin,
        return_flat=True,
        output_unit="Myr",
        log_time=False,
    )
)
scale = 0
if sfh == "continuity":
    scale = 0.3
if sfh == "continuity_bursty":
    scale = 1.0

for i in range(1, len(continuity["bin_edges"]) - 1):
    continuity["dsfr" + str(i)] = (-10.0, 10.0)
    continuity["dsfr" + str(i) + "_prior"] = "student_t"
    continuity["dsfr" + str(i) + "_prior_scale"] = (
        scale  # Defaults to this value as in Leja19, but can be set
    )
    continuity["dsfr" + str(i) + "_prior_df"] = (
        2  # Defaults to this value as in Leja19, but can be set
    )

fit_instructions_continuity = {
    "t_bc": 0.01,
    "continuity": continuity,
    "nebular": nebular,
    "dust": dust,
}

meta_continuity = {
    "run_name": "photoz_continuity",
    "redshift": "self",
    "redshift_sigma": "min",
    "min_redshift_sigma": 0.5,
    "fit_photometry": "TOTAL_BIN",
    "sampler": "multinest",
}

overall_dict = {
    "meta": meta_continuity,
    "fit_instructions": fit_instructions_continuity,
}

continuity_dict = copy.deepcopy(overall_dict)

# -------------------------------------------------------------------------------------

# third fit - Double power law

sfh = {}

sfh_type = "dblplaw"

sfh["tau"] = (0.001, 3.0)  # Vary the time of peak star-formation between
# the Big Bang at 0 Gyr and 15 Gyr later. In
# practice the code automatically stops this
# exceeding the age of the universe at the
# observed redshift.

sfh["tau_prior"] = "uniform"  # Impose a prior which is uniform in log_10 of the
if sfh["tau"][0] == 0 and sfh["tau_prior"] == "log_10":
    sfh["tau"][0] = 0.01

sfh["alpha"] = (0, 10.0)  # Vary the falling power law slope from 0.01 to 1000.
sfh["beta"] = (0, 10.0)  # Vary the rising power law slope from 0.01 to 1000.
sfh["alpha_prior"] = "uniform"  # Impose a prior which is uniform in log_10 of the
sfh["beta_prior"] = "uniform"  # parameter between the limits which have been set
# above as in Carnall et al. (2017).
sfh["massformed"] = (5.0, 12.0)
sfh["metallicity"] = (1e-3, 2.5)

fit_instructions_dpl = {"t_bc": 0.01, sfh_type: sfh, "nebular": nebular, "dust": dust}

meta_dpl = {
    "run_name": "photoz_dpl",
    "redshift": "self",
    "redshift_sigma": "min",
    "min_redshift_sigma": 0.5,
    "fit_photometry": "TOTAL_BIN",
    "sampler": "multinest",
}

overall_dict = {"meta": meta_dpl, "fit_instructions": fit_instructions_dpl}

dpl_dict = copy.deepcopy(overall_dict)

# -------------------------------------------------------------------------------------
# Fourth fit
# Log normal SFH

sfh = {}
sfh_type = "lognormal"

sfh["tmax"] = (0.01, 2.5)  # Gyr,  these will default to a flat prior probably
sfh["fwhm"] = (0.01, 1)  # Gyr
sfh["massformed"] = (5.0, 12.0)  # Log_10 total stellar mass formed: M_Solar
sfh["metallicity"] = (1e-3, 2.5)

fit_instructions_lognorm = {
    "t_bc": 0.01,
    sfh_type: sfh,
    "nebular": nebular,
    "dust": dust,
}

meta_lognorm = {
    "run_name": "photoz_lognorm",
    "redshift": "self",
    "redshift_sigma": "min",
    "min_redshift_sigma": 0.5,
    "fit_photometry": "TOTAL_BIN",
    "sampler": "multinest",
}

overall_dict = {"meta": meta_lognorm, "fit_instructions": fit_instructions_lognorm}

lognorm_dict = copy.deepcopy(overall_dict)


# -------------------------------------------------------------------------------------

# Fifth fit

resolved_sfh = {
    "age_max": (0.01, 2.5),  # Gyr
    "age_min": (0, 2.5),  # Gyr
    "metallicity": (1e-3, 2.5),  # solar
    "massformed": (4, 12),  # log mstar/msun
}

fit_instructions_resolved = {
    "t_bc": 0.01,
    "constant": resolved_sfh,
    "nebular": nebular,
    "dust": dust,
}
# This means that we are fixing the photo-z to the results from the 'photoz_DPL' run,
# specifically the 'MAG_APER_TOTAL' photometry
# We are fitting only the resolved photometry in the 'TOTAL_BIN' bins
meta_resolved = {
    "run_name": "CNST_SFH_RESOLVED",
    "redshift": "photoz_delayed",
    "redshift_id": "TOTAL_BIN",
    "fit_photometry": "bin",
}

overall_dict = {"meta": meta_resolved, "fit_instructions": fit_instructions_resolved}
resolved_dict = copy.deepcopy(overall_dict)


def create_dicts(dict, num, override_meta=None):
    dict = copy.deepcopy(dict)
    if override_meta:
        dict["meta"].update(override_meta)
    results = [copy.deepcopy(dict) for i in range(num)]
    return results
