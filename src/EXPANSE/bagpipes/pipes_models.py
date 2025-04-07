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

overall_dict = {
    "meta": meta_delayed,
    "fit_instructions": fit_instructions_delayed,
}

delayed_dict = copy.deepcopy(overall_dict)

# -------------------------------------------------------------------------------------
# Second fit
sfh = "continuity"
continuity = {}
continuity_bursty = {}
continuity["massformed"] = (
    5.0,
    12.0,
)  # Log_10 total stellar mass formed: M_Solar
continuity_bursty["massformed"] = (
    5.0,
    12.0,
)  # Log_10 total stellar mass formed: M_Solar

continuity["metallicity"] = (1e-3, 2.5)
continuity_bursty["metallicity"] = (1e-3, 2.5)

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
continuity_bursty["bin_edges"] = copy.deepcopy(continuity["bin_edges"])

for i in range(1, len(continuity["bin_edges"]) - 1):
    continuity["dsfr" + str(i)] = (-10.0, 10.0)
    continuity["dsfr" + str(i) + "_prior"] = "student_t"
    continuity["dsfr" + str(i) + "_prior_scale"] = (
        0.3  # Defaults to this value as in Leja19, but can be set
    )
    continuity["dsfr" + str(i) + "_prior_df"] = (
        2  # Defaults to this value as in Leja19, but can be set
    )

    continuity_bursty["dsfr" + str(i)] = (-10.0, 10.0)
    continuity_bursty["dsfr" + str(i) + "_prior"] = "student_t"
    continuity_bursty["dsfr" + str(i) + "_prior_scale"] = (
        1.0  # Defaults to this value as in Leja19, but can be set
    )
    continuity_bursty["dsfr" + str(i) + "_prior_df"] = (
        2  # Defaults to this value as in Leja19, but can be set
    )


fit_instructions_continuity = {
    "t_bc": 0.01,
    "continuity": continuity,
    "nebular": nebular,
    "dust": dust,
}

fit_instructions_continuity_bursty = {
    "t_bc": 0.01,
    "continuity": continuity_bursty,
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
    "update_cont_bins": True,
}

meta_continuity_bursty = copy.deepcopy(meta_continuity)
meta_continuity_bursty["run_name"] = "photoz_continuity_bursty"

overall_dict = {
    "meta": meta_continuity,
    "fit_instructions": fit_instructions_continuity,
}

continuity_dict = copy.deepcopy(overall_dict)

overall_dict = {
    "meta": meta_continuity_bursty,
    "fit_instructions": fit_instructions_continuity_bursty,
}

continuity_bursty_dict = copy.deepcopy(overall_dict)

# -------------------------------------------------------------------------------------

# third fit - Double power law

sfh = {}

sfh_type = "dblplaw"

sfh["tau"] = (0.001, 5.0)  # Vary the time of peak star-formation between
# the Big Bang at 0 Gyr and 15 Gyr later. In
# practice the code automatically stops this
# exceeding the age of the universe at the
# observed redshift.

sfh["tau_prior"] = "uniform"
if sfh["tau"][0] == 0 and sfh["tau_prior"] == "log_10":
    sfh["tau"][0] = 0.01

sfh["alpha"] = (
    0.01,
    100.0,
)  # Vary the falling power law slope from 0.01 to 1000.
sfh["beta"] = (
    0.01,
    100.0,
)  # Vary the rising power law slope from 0.01 to 1000.
sfh["alpha_prior"] = "log_10"
sfh["beta_prior"] = "log_10"
# above as in Carnall et al. (2017).
sfh["massformed"] = (5.0, 12.0)
sfh["metallicity"] = (1e-3, 2.5)

fit_instructions_dpl = {
    "t_bc": 0.01,
    sfh_type: sfh,
    "nebular": nebular,
    "dust": dust,
}

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
# Constant SFH fit to match resolved

sfh = {}
sfh_type = "constant"


sfh["age_max"] = (0.01, 2.5)  # Gyr
sfh["age_min"] = (0, 2.5)  # Gyr
sfh["massformed"] = (5.0, 12.0)  # Log_10 total stellar mass formed: M_Solar
sfh["metallicity"] = (1e-3, 2.5)
sfh["age_prior"] = "uniform"
sfh["metallicity_prior"] = "uniform"

fit_instructions_cnst = {
    "t_bc": 0.01,
    sfh_type: sfh,
    "nebular": nebular,
    "dust": dust,
}

meta_cnst = {
    "run_name": "photoz_cnst",
    "redshift": "self",
    "redshift_sigma": "min",
    "min_redshift_sigma": 0.5,
    "fit_photometry": "TOTAL_BIN",
    "sampler": "multinest",
}

overall_dict = {
    "meta": meta_cnst,
    "fit_instructions": fit_instructions_cnst,
}

cnst_dict = copy.deepcopy(overall_dict)


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

overall_dict = {
    "meta": meta_lognorm,
    "fit_instructions": fit_instructions_lognorm,
}

lognorm_dict = copy.deepcopy(overall_dict)


# -------------------------------------------------------------------------------------

# Fifth fit

resolved_sfh_cnst = {
    "age_max": (0.01, 2.5),  # Gyr
    "age_min": (0, 2.5),  # Gyr
    "metallicity": (1e-3, 2.5),  # solar
    "massformed": (4, 12),  # log mstar/msun
}

fit_instructions_resolved_cnst = {
    "t_bc": 0.01,
    "constant": resolved_sfh_cnst,
    "nebular": nebular,
    "dust": dust,
}
# This means that we are fixing the photo-z to the results from the 'photoz_DPL' run,
# specifically the 'MAG_APER_TOTAL' photometry
# We are fitting only the resolved photometry in the 'TOTAL_BIN' bins
meta_resolved_cnst = {
    "run_name": "CNST_SFH_RESOLVED",
    "redshift": "photoz_delayed",
    "redshift_id": "TOTAL_BIN",
    "fit_photometry": "bin",
}

overall_dict = {
    "meta": meta_resolved_cnst,
    "fit_instructions": fit_instructions_resolved_cnst,
}
resolved_dict_cnst = copy.deepcopy(overall_dict)

# -------------------------------------------------------------------------------------
# 6th fit - DB SFH

sfh = {}
sfh_type = "iyer2019"

sfh["massformed"] = (5.0, 12.0)  # Log_10 total stellar mass formed: M_Solar
sfh["metallicity"] = (1e-3, 2.5)

sfh["sfr"] = (
    1e-3,
    1e3,
)  # M_Solar/yr (or not - don't understand this parameter. Seems too large.)
sfh["bins"] = 4
sfh["bins_prior"] = "dirichlet"
sfh["alpha"] = 3.0  # Dirichlet alpha parameter  - this matches DB tutorial

# The Dirichlet prior has a single tunable parameter α that
# specifies how correlated the values are.
# In our case, values of this parameter α<1 result in values that can be arbitrarily close,
# leading to extremely spiky SFHs because galaxies have to assemble a
# significant fraction of their mass in a very short period of time,
# while α>1 leads to smoother SFHs with more evenly spaced values
# that never- theless have considerable diversity.
# In practice, we use a value of α=5, which leads to a
# distribution of parameters that is similar to what we find in SAM and MUFASA.

fit_instructions_db = {
    "t_bc": 0.01,
    sfh_type: sfh,
    "nebular": nebular,
    "dust": dust,
}

meta_db = {
    "run_name": "photoz_db",
    "redshift": "self",
    "redshift_sigma": "min",
    "min_redshift_sigma": 0.5,
    "fit_photometry": "TOTAL_BIN",
    "sampler": "multinest",
}

overall_dict = {
    "meta": meta_db,
    "fit_instructions": fit_instructions_db,
}

db_dict = copy.deepcopy(overall_dict)


# -------------------------------------------------------------------------------------
# Do a resolved fit with a continuity bursty SFH

fit_instructions_resolved_bursty = copy.deepcopy(fit_instructions_continuity_bursty)

meta_resolved_bursty = {
    "run_name": "BURSTY_SFH_RESOLVED",
    "redshift": "photoz_delayed",
    "redshift_id": "TOTAL_BIN",
    "fit_photometry": "bin",
    "update_cont_bins": True,
}

overall_dict = {
    "meta": meta_resolved_bursty,
    "fit_instructions": fit_instructions_resolved_bursty,
}

resolved_dict_bursty = copy.deepcopy(overall_dict)

# -------------------------------------------------------------------------------------


def create_dicts(
    dict,
    num,
    override_meta=None,
    redshifts=None,
    cont_nbins=6,
    first_bin=10 * u.Myr,
    second_bin=None,
):
    """
    Create a list of dictionaries from a template dictionary.
    This is useful for creating multiple dictionaries with the same
    structure, but different values for certain keys.

    Parameters
    ----------
    dict : dict
        The template dictionary to copy.
    num : int
        The number of dictionaries to create.
    override_meta : dict, optional
        A dictionary of metadata to override in the template dictionary.
        The default is None. Options are:
        - "binmap_type": str
            Which photometry to fit.
        - "name_append": str
            A string to append to the run name.
        - "remove": list
            A list of keys to remove from the template dictionary.
        - "redshift": str
            The redshift to use for the fit. This can be a string or a list of strings. E.g. 'eazy', 'self', or the name of another Bagpipes run.
        - "redshift_id": str
            If getting the redshift from a previous Bagpipes run, this is the ID of the galaxy to use.
        - "redshift_sigma": str
            The redshift uncertainty to use. This can be a None, a float, a string or a list of strings. E.g. 'eazy', 'self', or the name of another Bagpipes run.
        - "min_redshift_sigma": float
            The minimum redshift uncertainty to use. The default is 0.5.
        - "sampler": str
            The sampler to use. The default is "multinest".
        - "run_name": str
            The name of the run.
        - "use_bpass": bool
            Whether to use BPASS models or not. The default is True.
        - "fit_photometry": str
            The photometry to fit. The default is "TOTAL_BIN". Overrides what is set in run_bagpipes if it exists.
        - "update_cont_bins": bool
            Whether to update the continuity bin edges based on the redshift. The default is False.
        - "cont_nbins": int
            The number of bins to use for the continuity SFH. The default is 6.
        - "cont_first_bin": astropy Quantity
            The first bin edge for the continuity SFH. The default is 10 * u.Myr.

    redshifts : list, optional
        A list of redshifts to use for updating the continuity bin edges.
        The default is None.
    cont_nbins : int, optional
        The number of bins to use for the continuity SFH. The default is 6.
    first_bin : astropy Quantity, optional
        The first bin edge for the continuity SFH. The default is 10 * u.Myr.
    second_bin : astropy Quantity, optional
        The second bin edge for the continuity SFH. The default is None.
        If None, the bin edges will be calculated based on the redshift.


    """
    dict = copy.deepcopy(dict)
    override_meta = copy.deepcopy(override_meta)

    if override_meta:
        if "name_append" in override_meta.keys():
            dict["meta"]["run_name"] += override_meta["name_append"]
            override_meta.pop("name_append")

        if "remove" in override_meta.keys():
            for key in override_meta["remove"]:
                dict["meta"].pop(key)
            override_meta.pop("remove")

        dict["meta"].update(override_meta)

    results = [copy.deepcopy(dict) for i in range(num)]
    if redshifts is not None and "continuity" in dict["fit_instructions"].keys():
        assert len(redshifts) == num
        print("Updating continuity bin edges for redshifts")
        for redshift, result in zip(redshifts, results):
            result["fit_instructions"]["continuity"]["bin_edges"] = list(
                calculate_bins(
                    redshift=redshift,
                    num_bins=cont_nbins,
                    first_bin=first_bin,
                    second_bin=second_bin,
                    return_flat=True,
                    output_unit="Myr",
                    log_time=False,
                )
            )

    return results
