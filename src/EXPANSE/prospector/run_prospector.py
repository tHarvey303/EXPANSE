try:
    import time

    import fsps
    import prospect
    import prospect.io.read_results as reader
    from prospect import prospect_args
    from prospect.fitting import fit_model, lnprobfn
    from prospect.io import write_results as writer
    from prospect.models import PolySpecModel, SpecModel, priors, transforms
    from prospect.models.priors import (
        ClippedNormal,
        LogNormal,
        LogUniform,
        Normal,
        TopHat,
        Uniform,
    )
    from prospect.models.sedmodel import SedModel
    from prospect.models.templates import (
        TemplateLibrary,
        adjust_continuity_agebins,
    )
    from prospect.models.transforms import (
        logsfr_ratios_to_agebins,
        logsfr_ratios_to_masses,
        logsfr_ratios_to_masses_flex,
        logsfr_ratios_to_sfrs,
        psb_logsfr_ratios_to_agebins,
        tage_from_tuniv,
        zfrac_to_masses,
        zfrac_to_sfr,
        zfrac_to_sfrac,
    )
    from prospect.plotting import FigureMaker
    from prospect.plotting.corner import allcorner
    from prospect.plotting.sfh import nonpar_mwa, nonpar_recent_sfr, sfh_to_cmf
    from prospect.sources import CSPSpecBasis, FastStepBasis
    from prospect.utils.obsutils import fix_obs
    from sedpy import observate

    assert prospect.__version__ == "1.4.0"
except ImportError:
    pass

import copy
import json
import os
import sys
from typing import List, Optional, Union

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
from joblib import Parallel, delayed

from EXPANSE.bagpipes import calculate_bins

from .Plotspector import Plotspector
from .utils import filter_catalog, find_bands, load_spectra, provide_phot

sampling_default_settings = {
    "nlive_init": 400,  # dynesty -->
    "nested_sample": "rwalk",
    "nested_target_n_effective": 10000,  #  A value of 10,000 for this keyword specifies high-quality posteriors, whereas a value of 3,000 will produce reasonable but approximate posteriors.
    "nlive_batch": 200,
    "nested_dlogz_init": 0.05,
    "nested_maxcall": int(1e7),  # Can reduce this to see how fit is going
    "nwalkers": 28,  # emcee -->
    "niter": 2048,
    "nburn": [16, 32, 64],
}


# are we in MPI

try:
    from mpi4py import MPI

    rank = MPI.COMM_WORLD.Get_rank()
    size = MPI.COMM_WORLD.Get_size()
    from mpi4py.futures import MPIPoolExecutor

    if size > 1:
        if rank == 0:
            print("Running with mpirun/mpiexec detected.")
        MPI.COMM_WORLD.Barrier()
        print(f"Message from process {rank}")
        sys.stdout.flush()
        MPI.COMM_WORLD.Barrier()
except ImportError:
    rank = 0
    size = 1


# This works for Prospector 1.4.0. When v2.0.0 is released, this will need to be updated to handle the new model setup.

# allowed parameters

# This is the default model.
# It is a delayed-tau model with a Calzetti dust law, Kroupa IMF, and a uniform redshift prior between 0 and 25.
# Nebular emission is included with a uniform logU prior between -3 and -1 and a uniform logZ prior between -2 and 0.2.
# The IGM absorption is included but fixed at the Madau 1995 prescription.

default_model_priors = {
    "dust": {
        "type": "calzetti",
        "prior": "uniform",
        "prior_range": [0, 6],
        "add_dust_emission": True,
        "add_dust": True,
    },
    "imf": {"type": "kroupa"},
    "zred": {"prior": "uniform", "prior_range": [0, 25]},  # redshift prior
    "igm": {
        "prior": "normal",
        "prior_range": [0, 2],
        "mean": 1.0,
        "sigma": 0.3,
        "add_igm": True,
        "vary_igm": False,
    },  # IGM absorption
    "nebular": {
        "add_neb": True,
        "type": "nebular",  # Can also be 'nebular_marginalization' or to marginalize over emission line fluxes
        "redshift": "stellar",  # Can be 'stellar' or 'fit' to fit for redshift offset
        "logu": {"prior": "uniform", "prior_range": [-3, -1]},
        "logz": {"prior": "uniform", "prior_range": [-2, 0.2], "link_to_stars": False},
    },
    "use_builtin": False,  # Don't use a built-in model like prospector alpha or beta
    "sfh": {
        "model": "delayed-tau",
        "use_default": False,  # If True, defaults will be used for all parameters
        "tau": {"prior": "log_10", "prior_range": [0.1, 30]},
        "mass": {"prior": "log_10", "prior_range": [1e6, 1e12]},
        "tage": {
            "prior": "log_10",
            "prior_range": [0.001, 1],
        },  # in terms of available lookback time
        "logz": {"prior": "uniform", "prior_range": [-4, 0.19]},  # metallicity
    },
}


def generate_model_name_from_run_params(run_params):
    """Generate a model name from the run_params dictionary."""
    model_name = run_params["sfh"]["model"]
    model_name += f"_{run_params['imf']['type']}"
    model_name += f"_{run_params['dust']['type']}" if run_params["dust"]["add_dust"] else ""
    model_name += "_neb" if run_params["nebular"]["add_neb"] else ""
    model_name += f"_z_{run_params['zred']['prior']}"

    return model_name


def build_sps(run_params):
    if run_params["model"]["sfh"]["model"] in [
        "continuity",
        "continuity_bursty",
        "continuity_flex",
        "continuity_psb",
        "dirichlet",
        "alpha",
        "beta",
    ]:
        sps = FastStepBasis(zcontinuous=1)
    elif run_params["model"]["sfh"]["model"] in [
        "rising",
        "exp",
        "delayed",
        "const",
        "ssp",
        "delayed-tau",
        "delayed-exp",
    ]:
        sps = CSPSpecBasis(zcontinuous=1)
    else:
        print(f'{run_params["sfh_model"]} not understood.')
    return sps


def sanity_check(run_params):
    """Validate that all required parameters are present in the run_params dictionary."""
    required_params = {
        "sfh": "SFH model not specified in run_params",
        "imf": "IMF not specified in run_params",
        "dust": "Dust model not specified in run_params",
        "zred": "Redshift not specified in run_params",
    }

    for param, error_msg in required_params.items():
        if param not in run_params:
            raise ValueError(error_msg)

    if "model" not in run_params["sfh"]:
        raise ValueError("SFH model not specified in run_params")


def is_varying_parameter(param_value):
    """Check if a parameter should vary based on its type."""
    return isinstance(param_value, dict)


def setup_parameter(param_name, param_value, model_params, prior_conv=None, mini_sigma=3):
    """Set up a model parameter based on its configuration.

    Args:
        param_name: The name of the parameter
        param_value: The parameter configuration (dict) or fixed value
        model_params: The model parameters dictionary to update
        prior_conv: Dictionary mapping prior names to prior classes
    """
    if prior_conv is None:
        prior_conv = {
            "uniform": TopHat,
            "log_10": LogUniform,
            "normal": Normal,
            "clipped_normal": ClippedNormal,
        }

    if isinstance(param_value, dict):
        # Parameter should vary
        if param_name not in model_params:
            # Create parameter entry if it doesn't exist
            model_params[param_name] = {"N": 1, "init": 0.0}

        model_params[param_name]["isfree"] = True

        # Set initial value if provided
        if "init" in param_value:
            model_params[param_name]["init"] = param_value["init"]

        if "isfree" in param_value:
            model_params[param_name]["isfree"] = param_value["isfree"]

        if "depends_on" in param_value:
            model_params[param_name]["depends_on"] = param_value["depends_on"]

        if "N" in param_value:
            model_params[param_name]["N"] = param_value["N"]

        # Set prior if provided
        if "prior" in param_value:
            prior_type = param_value["prior"]

            if prior_type in prior_conv:
                if prior_type == "normal" or prior_type == "clipped_normal":
                    assert "init" not in param_value, "Cannot specify both init and mean/sigma"
                    mean = param_value["mean"]
                    sigma = param_value["sigma"]
                    if prior_type == "clipped_normal":
                        if "prior_range" in param_value:
                            prior_range = param_value["prior_range"]
                            assert (
                                "mini" not in param_value
                            ), "Cannot specify both prior_range and mini/maxi"
                            assert (
                                "maxi" not in param_value
                            ), "Cannot specify both prior_range and mini/maxi"
                            param_value["mini"] = prior_range[0]
                            param_value["maxi"] = prior_range[1]

                        mini = param_value.get("mini", -1 * mini_sigma * sigma)
                        maxi = param_value.get("maxi", mini_sigma * sigma)

                        model_params[param_name]["prior"] = prior_conv[prior_type](
                            mean=mean, sigma=sigma, mini=mini, maxi=maxi
                        )
                    else:
                        model_params[param_name]["prior"] = prior_conv[prior_type](
                            mean=mean, sigma=sigma
                        )
                else:
                    if "prior_range" in param_value:
                        prior_range = param_value["prior_range"]
                        assert len(prior_range) == 2, "Prior range must have two values"
                        assert (
                            "mini" not in param_value
                        ), "Cannot specify both prior_range and mini/maxi"
                        assert (
                            "maxi" not in param_value
                        ), "Cannot specify both prior_range and mini/maxi"

                        mini = prior_range[0]
                        maxi = prior_range[1]
                    else:
                        mini = param_value["mini"]
                        maxi = param_value["maxi"]
                    model_params[param_name]["prior"] = prior_conv[prior_type](mini=mini, maxi=maxi)
            else:
                raise ValueError(f"Unrecognized prior type: {prior_type}")
    else:
        # Parameter is fixed
        if param_name not in model_params:
            # Create parameter entry if it doesn't exist
            model_params[param_name] = {"N": 1}

        model_params[param_name]["isfree"] = False
        model_params[param_name]["init"] = param_value


def build_model(run_params):
    """Build a SED model from run_params dictionary.

    Args:
        run_params: Dictionary containing model parameters

    Returns:
        SedModel instance
    """
    # Apply default parameters
    from copy import deepcopy

    model_run_params = deepcopy(default_model_priors)

    # Update with provided parameters, merging dictionaries recursively
    def update_dict_recursive(d, u):
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                update_dict_recursive(d[k], v)
            else:
                d[k] = v

    update_dict_recursive(model_run_params, run_params["model"])
    run_params["model"] = model_run_params

    if "model" not in run_params:
        raise ValueError("No model parameters provided")

    # Check if a pre-built model is provided
    if isinstance(model_run_params, SedModel):
        print("Overriding all other model parameters with provided SedModel")
        print(model_run_params.description)
        return model_run_params
    else:
        if rank == 0:
            print("Building model from run_params and default priors")

    sanity_check(model_run_params)

    # Dictionary mappings
    imf_dict = {
        "salpeter": 0,
        "chabrier": 1,
        "kroupa": 2,
        "van_dokkum": 3,
        "dave": 4,
        "tabulated": 5,
    }
    dust_dict = {
        "powerlaw": 0,
        "cardelli": 1,
        "mw": 1,
        "calzetti": 2,
        "wgp": 3,
        "kriek_conroy": 4,
        "gordon": 5,
        "smc": 5,
        "reddy": 6,
    }
    prior_conv = {
        "uniform": TopHat,
        "log_10": LogUniform,
        "normal": Normal,
        "clipped_normal": ClippedNormal,
    }

    # Handle built-in models
    if model_run_params.get("use_builtin", False):
        if model_run_params["use_builtin"] == "alpha":
            model_params = TemplateLibrary["alpha"]
            model_params["zred"]["init"] = model_run_params.get("redshift", 0)

            if model_run_params.get("vary_redshift", False):
                model_params["zred"]["isfree"] = True
                if is_varying_parameter(model_run_params["zred"]):
                    setup_parameter("zred", model_run_params["zred"], model_params, prior_conv)
                else:
                    sigma = model_run_params.get("redshift_sigma", 0.1)
                    model_params["zred"]["prior"] = Normal(
                        mean=model_run_params["redshift"], sigma=sigma
                    )

        elif model_run_params["use_builtin"] == "beta":
            model_params = TemplateLibrary["beta"]

        else:
            raise ValueError(f"Unrecognized built-in model: {model_run_params['use_builtin']}")

    else:
        # Create a model from scratch
        sfh_model = model_run_params["sfh"]["model"]

        # Select SFH template based on model type
        if sfh_model in ["alpha", "beta"]:
            model_params = TemplateLibrary[sfh_model]

        elif sfh_model in [
            "continuity",
            "continuity_bursty",
            "continuity_flex",
            "continuity_psb",
            "dirichlet",
            "stochastic",
        ]:
            # Map the SFH model to the appropriate template
            if sfh_model == "continuity_bursty":
                template_model = "continuity_sfh"
            else:
                template_model = f"{sfh_model}_sfh"

            # Get the template model parameters
            model_params = TemplateLibrary[template_model]

            # Configure SFH parameters
            num_bins = model_run_params["sfh"].get("num_sfh_bins", 6)
            redshift = model_run_params.get("redshift", 0)
            use_default_sfh_settings = model_run_params["sfh"].get("use_default", False)
            if not use_default_sfh_settings:
                if rank == 0:
                    print("Using custom SFH settings")
                if sfh_model in ["continuity", "continuity_bursty"]:
                    # Configure continuity SFH
                    model_params["mass"] = {
                        "N": num_bins,
                        "isfree": False,
                        "init": 1e6,
                        "units": r"M$_\odot$",
                        "depends_on": transforms.logsfr_ratios_to_masses,
                    }

                    bins = calculate_bins(redshift, num_bins=num_bins)

                    model_params["agebins"] = {
                        "N": num_bins,
                        "isfree": False,
                        "init": bins,
                        "units": "log(yr)",
                    }

                    model_params["logmass"] = {
                        "N": 1,
                        "isfree": True,
                        "init": 9,
                        "units": "Msun",
                        "prior": priors.TopHat(mini=6, maxi=12),
                    }

                    # Set scale based on SFH model type
                    scale = 1.0 if sfh_model == "continuity_bursty" else 0.3

                    model_params["logsfr_ratios"] = {
                        "N": num_bins - 1,
                        "isfree": True,
                        "init": [0.0] * (num_bins - 1),
                        "prior": priors.StudentT(
                            mean=np.full(num_bins - 1, 0.0),
                            scale=np.full(num_bins - 1, scale),
                            df=np.full(num_bins - 1, 2),
                        ),
                    }

                elif sfh_model == "continuity_flex":
                    bins = calculate_bins(redshift, num_bins=num_bins)

                    model_params["logsfr_ratio_young"] = {
                        "N": 1,
                        "isfree": True,
                        "init": 0.0,
                        "units": r"dlogSFR (dex)",
                        "prior": priors.StudentT(mean=0.0, scale=0.3, df=2),
                    }

                    model_params["logsfr_ratio_old"] = {
                        "N": 1,
                        "isfree": True,
                        "init": 0.0,
                        "units": r"dlogSFR (dex)",
                        "prior": priors.StudentT(mean=0.0, scale=0.3, df=2),
                    }

                    model_params["logsfr_ratios"] = {
                        "N": num_bins - 3,
                        "isfree": True,
                        "init": [0.0] * (num_bins - 3),
                        "units": r"dlogSFR (dex)",
                        "prior": priors.StudentT(
                            mean=np.full(num_bins - 3, 0.0),
                            scale=np.full(num_bins - 3, 0.3),
                            df=np.full(num_bins - 3, 2),
                        ),
                    }

                    model_params["mass"] = {
                        "N": num_bins,
                        "isfree": False,
                        "init": 1e6,
                        "units": r"M$_\odot$",
                        "depends_on": transforms.logsfr_ratios_to_masses_flex,
                    }

                    model_params["agebins"] = {
                        "N": num_bins,
                        "isfree": False,
                        "depends_on": transforms.logsfr_ratios_to_agebins,
                        "init": bins,
                        "units": "log(yr)",
                    }

                elif sfh_model == "continuity_psb":
                    # prospect.models.templates.TemplateLibrary
                    # Should allow for varying bins and redshift a bit
                    if rank == 0:
                        print("Custom PSB SFH not configured, using default params")
                elif sfh_model == "stochastic":
                    if rank == 0:
                        print("Custom stochastic SFH not configured, using default params")

                elif sfh_model == "dirichlet":
                    bins = calculate_bins(redshift, num_bins=num_bins)

                    model_params["mass"] = {
                        "N": num_bins,
                        "isfree": False,
                        "init": 1.0,
                        "units": r"M$_\odot$",
                        "depends_on": transforms.zfrac_to_masses,
                    }

                    model_params["agebins"] = {
                        "N": num_bins,
                        "isfree": False,
                        "init": bins,
                        "units": "log(yr)",
                    }

                    model_params["z_fraction"] = {
                        "N": num_bins - 1,
                        "isfree": True,
                        "init": [0] * (num_bins - 1),
                        "units": None,
                        "prior": priors.Beta(alpha=1.0, beta=1.0, mini=0.0, maxi=1.0),
                    }

                else:
                    raise ValueError(f"SFH model {sfh_model} not recognised.")

        elif sfh_model in ["rising", "exp", "delayed", "const", "delayed-tau", "delayed-exp"]:
            template_model = "parametric_sfh"
            model_params = TemplateLibrary[template_model]

            # Set coefficients based on SFH model
            const_coeff = {"const": 1}
            sfh_coeff = {
                "rising": 4,
                "exp": 1,
                "delayed": 4,
                "delayed-tau": 4,
                "delayed-exp": 4,
                "const": 0,
            }

            model_params["const"] = {"init": const_coeff.get(sfh_model, 0), "isfree": False}
            model_params["sfh"] = {"init": sfh_coeff[sfh_model], "isfree": False}

            # Configure tau parameter
            if sfh_model not in ["const"]:
                if is_varying_parameter(model_run_params["sfh"].get("tau", {})):
                    setup_parameter("tau", model_run_params["sfh"]["tau"], model_params, prior_conv)
                else:
                    model_params["tau"] = {"init": 1, "isfree": False}

            # Configure mass parameter
            if is_varying_parameter(model_run_params["sfh"].get("mass", {})):
                setup_parameter("mass", model_run_params["sfh"]["mass"], model_params, prior_conv)
            else:
                mass_value = model_run_params["sfh"].get("mass", 1e10)
                model_params["mass"] = {"init": mass_value, "isfree": False}

            # Configure age parameters
            model_params["tage"]["isfree"] = False
            model_params["tage_tuniv"] = {
                "isfree": True,
                "init": 0.5,
                "prior": TopHat(mini=0, maxi=1.0),
            }
            model_params["tage"]["depends_on"] = tage_from_tuniv

            # Configure age prior if it should vary
            if is_varying_parameter(model_run_params["sfh"].get("tage", {})):
                setup_parameter(
                    "tage_tuniv", model_run_params["sfh"]["tage"], model_params, prior_conv
                )

        else:
            raise ValueError(f"SFH model {sfh_model} not recognized")

        # Add nebular emission
        if "nebular" in model_run_params and model_run_params["nebular"].get("add_neb", True):
            if rank == 0:
                print("Adding nebular emission.")
            nebtype = model_run_params["nebular"].get("type", "nebular")
            model_params.update(TemplateLibrary[nebtype])

            neb_redshift = model_run_params["nebular"].get("redshift", "stellar")
            if neb_redshift == "fit":
                model_params.update(TemplateLibrary["fit_eline_redshift"])
                if rank == 0:
                    print("Fitting redshift offset for margninalized emission lines.")

            if nebtype == "nebular_marginalization":
                # Em line names to match $SPS_HOME/data/emlines_info.dat
                # Copy over 'elines_to_fit, elines_to_fix and elines_to_ignore if they exist
                for key in [
                    "elines_to_fit",
                    "elines_to_fix",
                    "elines_to_ignore",
                    "eline_sigma",
                    "eline_delta_zred",
                ]:
                    if key in model_run_params["nebular"]:
                        model_params[key] = model_run_params["nebular"][key]

        # Add dust emission if needed
        if model_run_params["dust"].get("add_dust_emission", False):
            model_params.update(TemplateLibrary["dust_emission"])

        # Add IGM absorption if needed
        if model_run_params.get("add_igm", True):
            model_params.update(TemplateLibrary["igm"])

        # Configure IGM parameters
        if model_run_params.get("vary_igm", False):
            igm_config = model_run_params.get("igm", {})
            mean = igm_config.get("mean", 1.0)
            sigma = igm_config.get("sigma", 0.3)
            mini = igm_config.get("prior_range", [0.0, 2.0])[0]
            maxi = igm_config.get("prior_range", [0.0, 2.0])[1]

            model_params["igm_factor"]["prior"] = ClippedNormal(
                mean=mean, sigma=sigma, mini=mini, maxi=maxi
            )
            model_params["igm_factor"]["isfree"] = True

        # Set metallicity (logzsol) from SFH parameters
        if "logz" in model_run_params["sfh"]:
            if is_varying_parameter(model_run_params["sfh"]["logz"]):
                setup_parameter(
                    "logzsol", model_run_params["sfh"]["logz"], model_params, prior_conv
                )
            else:
                model_params["logzsol"]["init"] = model_run_params["sfh"]["logz"]
                model_params["logzsol"]["isfree"] = False
        else:
            # Default metallicity settings
            model_params["logzsol"]["init"] = -2
            model_params["logzsol"]["prior"] = TopHat(mini=-4, maxi=0.19)
            model_params["logzsol"]["isfree"] = True

        # Configure gas parameters
        gas_config = model_run_params.get("nebular", {})

        # Unlink gas and stellar metallicity if specified
        if "logz" in gas_config and not gas_config["logz"].get("link_to_stars", True):
            if rank == 0:
                print("Unlinking gas and stellar metallicity.")
            if is_varying_parameter(gas_config.get("logz", {})):
                setup_parameter("gas_logz", gas_config["logz"], model_params, prior_conv)
            else:
                model_params["gas_logz"] = {}
                model_params["gas_logz"]["init"] = gas_config.get("logz", -2)
                model_params["gas_logz"]["isfree"] = False

            if "depends_on" in model_params["gas_logz"]:
                model_params["gas_logz"].pop("depends_on")

        # Configure ionization parameter
        if is_varying_parameter(gas_config.get("logu", {})):
            setup_parameter("gas_logu", gas_config["logu"], model_params, prior_conv)
        else:
            model_params["gas_logu"] = {}
            model_params["gas_logu"]["init"] = gas_config.get("logu", -2)
            model_params["gas_logu"]["isfree"] = False

        # Set IMF type
        imf_type = model_run_params.get("imf", {}).get("type", "kroupa")
        model_params["imf_type"]["init"] = imf_dict.get(imf_type, 2.0)  # Default to Kroupa

        # Set dust type
        dust_type = model_run_params.get("dust", {}).get("type", "calzetti")
        model_params["dust_type"]["init"] = dust_dict.get(dust_type, 2.0)  # Default to Calzetti

        # Configure dust attenuation parameters
        dust_config = model_run_params.get("dust", {})

        # Handle dust2 parameter (attenuation)
        if is_varying_parameter(dust_config.get("dust2", None)):
            setup_parameter("dust2", dust_config["dust2"], model_params, prior_conv)
        elif "prior" in dust_config:
            # Backward compatibility for top-level dust_prior
            dust_prior_type = dust_config.get("prior", "uniform")
            dust_range = dust_config.get("prior_range", [0, 6])

            model_params["dust2"]["isfree"] = True
            if dust_prior_type == "log_10":
                model_params["dust2"]["prior"] = LogUniform(
                    mini=max(1e-2, dust_range[0]), maxi=dust_range[1]
                )
            elif dust_prior_type == "uniform":
                model_params["dust2"]["prior"] = TopHat(mini=dust_range[0], maxi=dust_range[1])

        # Process other dust parameters not explicitly handled
        for key, value in dust_config.items():
            if key not in ["type", "prior", "prior_range", "add_dust_emission", "add_dust"]:
                dust_param = key
                if rank == 0:
                    print(f"Adding custom dust parameter: {dust_param}")
                setup_parameter(dust_param, value, model_params, prior_conv)

        # Set redshift
        redshift = model_run_params["zred"]
        setup_parameter("zred", redshift, model_params, prior_conv)

        # Configure redshift prior if varying
        if model_run_params.get("vary_redshift", False):
            model_params["zred"]["isfree"] = True

            if model_run_params.get("no_redshift_prior", False):
                model_params["zred"]["prior"] = TopHat(mini=0, maxi=25)
            else:
                sigma = model_run_params.get("redshift_sigma", 0.1)
                mini = max(0.001, redshift - 3 * sigma)
                maxi = redshift + 3 * sigma

                model_params["zred"]["prior"] = ClippedNormal(
                    mean=redshift, sigma=sigma, mini=mini, maxi=maxi
                )

        # Add AGN component if needed
        if model_run_params.get("add_agn", False):
            model_params.update(TemplateLibrary["agn"])  # Add AGN component
            model_params.update(TemplateLibrary["agn_eline"])  # Add AGN emission lines
            model_params["fagn"]["isfree"] = True
            model_params["agn_tau"]["isfree"] = True

        # Process any other parameters not explicitly handled
        for key, value in model_run_params.items():
            if key not in [
                "sfh",
                "dust",
                "imf",
                "zred",
                "nebular",
                "igm",
                "add_igm",
                "vary_igm",
                "add_agn",
                "use_builtin",
                "redshift",
                "vary_redshift",
                "spectral_smoothing",
                "calibration",
                "redshift_sigma",
                "no_redshift_prior",
                "vary_u",
                "dust_prior",
                "model",
            ]:
                if rank == 0:
                    print(f"Adding custom parameter: {key}")
                setup_parameter(key, value, model_params, prior_conv)

    fit_type = run_params["fit_type"]
    if fit_type == "phot":
        model = SedModel
    elif fit_type == "spec" or fit_type == "both":
        # smooth_type - options are 'vel', 'R', 'lambda', 'lsf'
        #   vel - velocity smoothing - resolution in km/s
        #   R - spectral resolution smoothing - resolution in lambda/sigma_lambda (dispersion, not FWHM)
        #   lambda - wavelength smoothing - resolution in Angstroms
        #   lsf - line spread function smoothing - arbitary, requires 'lsf' function or precomputed lsf stored in 'sigma_smooth' vector.

        # fftsmooth - boolean, whether to use FFT smoothing - faster, but may have boundary effects
        # Add any smoothing parameters
        model = SpecModel
        if "spectral_smoothing" in model_run_params:
            # Default is vel, tophat[10, 300], fftsmooth
            model_params.update(TemplateLibrary["spectral_smoothing"])
        else:
            print("No spectral smoothing parameters provided.")

        if fit_type == "both":
            # Add calibration chebyshev polynomial parameters if requested
            if "calibration" in model_run_params:
                # Options are optimize_speccal and fit_speccal
                cal_type = run_params["calibration"]

                model_params.update(TemplateLibrary[cal_type])
                model = PolySpecModel

    # Create and return the model

    model = model(model_params)

    if rank == 0:
        print(model.description)

    return model


def build_noise(run_params):
    # TODO: Implement noise model for photometry and spectra
    return (None, None)


def build_obs(run_params):
    # Build the observation dictionary with support for photometry and spectra
    assert ("flux" in run_params and "flux_err" in run_params and "filters" in run_params) or (
        "wavelength" in run_params and "spectrum" in run_params and "unc" in run_params
    ), "Must provide flux and flux error or spectrum and uncertainty"

    id = run_params["OBJID"]
    obs_dict = {}

    fit_type = None

    if "flux" in run_params:
        min_percentage_err = run_params["min_percentage_err"]
        flux = run_params["flux"]
        flux_err = run_params["flux_err"]
        filters = run_params["filters"]
        instruments = run_params.get("instruments", None)
        if instruments is None:
            assert all(
                len(f.split("_")) >= 2 for f in filters
            ), "Filters must be in the format instrument_band"
        else:
            assert len(instruments) == len(filters), "Instruments and filters must be same length"
            filters = [
                f"{instrument}_{band.lower()}" for instrument, band in zip(instruments, filters)
            ]

        assert len(flux) == len(flux_err), "Flux and flux error must be same length"
        assert len(flux) == len(filters), "Flux and filters must be same length"

        assert (
            type(flux) is u.Quantity or "flux_unit" in run_params
        ), "Flux must be astropy quantity"
        assert (
            type(flux_err) is u.Quantity or "flux_unit" in run_params
        ), "Flux error must be astropy quantity"

        if "flux_unit" in run_params:
            flux = flux * run_params["flux_unit"]
            flux_err = flux_err * run_params["flux_unit"]

        flux = copy.copy(flux.to(u.Jy).value)
        flux_err = copy.copy(flux_err.to(u.Jy).value)

        filterlist = observate.load_filters(filters)

        # Replace lower code with more efficient variant

        mask = (flux > 0) & (flux_err > 0) & (flux_err / flux < min_percentage_err / 100)
        flux_err[mask] = flux[mask] * min_percentage_err / 100
        flux /= 3631  # convert to maggies
        flux_err /= 3631

        obs_dict["filters"] = filterlist
        obs_dict["phot_wave"] = np.array([f.wave_effective for f in obs_dict["filters"]])

        # assert np.array_equal(np.sort(obs_dict["phot_wave"]), obs_dict["phot_wave"])
        obs_dict["maggies"] = np.array(flux)
        obs_dict["maggies_unc"] = np.array(flux_err)
        # Option to mask out certain filters
        if "phot_mask" in run_params:
            obs_dict["phot_mask"] = run_params["phot_mask"]

        fit_type = "phot"

    if "wavelength" in run_params:
        spectrum = run_params["spectrum"]
        unc = run_params["unc"]
        wavelength = run_params["wavelength"]

        assert (
            len(spectrum) == len(unc) == len(wavelength)
        ), "Spectrum, uncertainty and wavelength must be same length"
        assert (
            type(spectrum) is u.Quantity or "spectrum_unit" in run_params
        ), "Spectrum must be astropy quantity"
        assert (
            type(unc) is u.Quantity or "spectrum_unit" in run_params
        ), "Uncertainty must be astropy quantity"
        assert (
            type(wavelength) is u.Quantity or "wavelength_unit" in run_params
        ), "Wavelength must be astropy quantity"

        if "wavelength_unit" in run_params:
            wavelength = wavelength * run_params["wavelength_unit"]
        if "spectrum_unit" in run_params:
            spectrum = spectrum * run_params["spectrum_unit"]
            unc = unc * run_params["spectrum_unit"]

        obs_dict["wavelength"] = wavelength.to(u.Angstrom).value
        obs_dict["spectrum"] = spectrum.to(u.Jy).value / 3631  # convert to maggies
        obs_dict["unc"] = unc.to(u.Jy).value / 3631

        if "mask" in run_params:
            assert len(run_params["mask"]) == len(
                obs_dict["wavelength"]
            ), "Mask must be same length as spectrum"

            obs_dict["mask"] = run_params["mask"]

        fit_type = "spec" if fit_type is None else "both"

    assert fit_type is not None, "Must provide either photometry or spectrum"
    obs_dict["fit_type"] = fit_type
    run_params["fit_type"] = fit_type  # pass to model to add appropriate params

    if "meta" in run_params:
        # Store metadata
        obs_dict["meta"] = run_params["meta"]

    """
    if run_params['vary_redshift'] and not run_params['no_redshift_prior']:
        if type(run_params['set_redshift_sigma']) == str:
                if type (run_params['redshift_sigma_col']) == str:

                    run_params['redshift_sigma'] = float(catalog_obj[run_params['redshift_sigma_col']])
                elif type (run_params['redshift_sigma_col']) == list:
                    run_params['redshift_sigma'] = (float(catalog_obj[run_params['z_column']]-catalog_obj[run_params['redshift_sigma_col'][0]]) + float(catalog_obj[run_params['redshift_sigma_col'][1]]-catalog_obj[run_params['z_column']]))/2
        
        elif type(run_params['set_redshift_sigma']) in [list, np.ndarray]:
            if len(run_params['set_redshift_sigma']) == len(catalog):
                run_params['redshift_sigma'] = run_params['set_redshift_sigma'][val]
                print(f'Using custom redshift sigma of {run_params["set_redshift_sigma"][val]} for {id}')
            
        elif type(run_params['set_redshift_sigma']) in [float, int]:
            run_params['redshift_sigma'] = run_params['set_redshift_sigma']
            print(f'Using custom redshift sigma of {run_params["set_redshift_sigma"]} for {id}')
        else:
            'vary_redshift is on but no column name, list or value for redshift_sigma has been given, and no redshift prior. Setting redshift sigma to dz=1.'
            run_params['redshift_sigma'] = 1
    if run_params['set_redshift'] == None:
        run_params["redshift"] = float(catalog_obj[run_params['z_column']])
    elif len(run_params['set_redshift']) == len(catalog):
        print(f'Using custom redshift of {run_params["set_redshift"]} for {id}')
        run_params["redshift"] = run_params['set_redshift'][val]
    """

    fixed_obs = fix_obs(obs_dict)
    return fixed_obs


def build_all(run_params):
    obs = build_obs(run_params)
    model = build_model(run_params)
    if rank == 0:
        print("Loaded data.")
    sps = build_sps(run_params)
    if rank == 0:
        print("Built SPS model.")
    noise = build_noise(run_params)
    return obs, model, sps, noise


def sanity_check_run_params(run_params):
    required = ["OBJID", "model", "sampler"]

    for r in required:
        if r not in run_params:
            raise ValueError(f"{r} not found in run_params.")


def fix_params(params):
    for key, value in params.items():
        if isinstance(value, u.Quantity):
            params[key] = value.value.tolist()
        elif isinstance(value, np.ndarray):
            params[key] = value.tolist()
        elif isinstance(value, dict):
            fix_params(value)
        elif isinstance(value, bytes):
            params[key] = str(key)
        if type(value) not in [int, float, str, list, dict, bool]:
            print(f"Unrecognized type for {key}: {type(value)}")
            print(f"Value: {value}")


def remove_obs(obs):
    keys = ["unc", "spectrum", "wave", "flux", "flux_err", "wavelength"]
    for key in keys:
        if key in obs:
            obs.pop(key)


def main(config):
    """
    config: dictionary with the following keys
    OBJID: object ID
    run_path: path to save output
    flux: fluxes in Jy
    flux_err: flux errors in Jy
    redshift: redshift
    filters: list of filters
    load: if True, load from file

    """
    load = config.get("load", False)

    if not load:
        # Get the default argument parser
        # Don't parse in args
        # parser = prospect_args.get_parser()
        # args = parser.parse_args()
        # run_params = vars(args)
        run_params = {}

        run_params["param_file"] = __file__

        # Add sampling parameters
        run_params.update(sampling_default_settings)

        # Update run_params with user config
        run_params.update(config)

        # Check that required parameters are present
        sanity_check_run_params(run_params)

        # Run settings
        run_params["verbose"] = 2
        obs, model, sps, noise = build_all(run_params)

        # Add SPS libraries to run_params
        run_params["sps_libraries"] = sps.ssp.libraries

        # Get options for the sampler
        path = run_params["run_path"]
        id = run_params["OBJID"]
        sampler = run_params["sampler"]
        custom = run_params.get("custom", "")

        # build the fit ingredients
        if path.endswith(".h5"):
            if not os.path.exists(os.path.dirname(path)):
                os.makedirs(os.path.dirname(path))
        else:
            if not os.path.exists(path):
                os.makedirs(path)

            path = os.path.join(path, f"{id}{custom}.h5")

        # spec, phot, mfrac = model.predict(model.theta, obs=obs, sps=sps)

        # Do emcee/dynesty
        run_params["optimize"] = False
        if run_params["sampler"] == "dynesty":
            run_params["dynesty"] = True
        elif run_params["sampler"] == "emcee":
            run_params["emcee"] = True
        else:
            print(f"Sampler {sampler} not recognized.")
            return False

        # rename model to input_model in run_params
        run_params["input_model"] = model
        run_params.pop("model")

        try:
            print(f'Beginning {run_params["sampler"]} run. I may take a while.')
            output = fit_model(obs, model, sps, lnprobfn=lnprobfn, noise=noise, **run_params)
        except RuntimeError as e:
            print(
                f'Runtime Error. I am crashing. I am {run_params["OBJID"]} and it is {time.strftime("%y%b%d-%H.%M", time.localtime())}.'
            )
            print(f'I am sfh_model = {run_params["sfh_model"]}')
            if run_params["skip_errors"]:
                print(e)
                u = e.u
                theta = model.prior_transform(u)
                print("theta")
                print(theta)
                print("Passing to next galaxy")
                return False
            else:
                raise e

        # everything inside run_params needs to be serializable
        # so we can save it to disk

        remove_obs(run_params)
        fix_params(run_params)

        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))

        writer.write_hdf5(
            path,
            run_params,
            model,
            obs,
            output["sampling"][0],
            output["optimization"][0],
            tsample=output["sampling"][1],
            toptimize=output["optimization"][1],
            sps=sps,
        )

        print(f'done {run_params["sampler"]} in {output["sampling"][1]*3600}h')

    if True:
        # Load from file.
        if load:
            print(config["OBJID"])
            print(config["run_path"])
            hfile = config["run_path"]
            if not os.path.exists(hfile):
                print("Run not found.")
                return False
            else:
                print("Found .h5")

        res, obs, model = reader.results_from(hfile)
        fig_obj = Plotspector(results_file=hfile)

        # Trace plots
        run_params = res["run_params"]
        id = run_params["OBJID"]
        field = run_params["field"]
        sfh_model = run_params["sfh"]["model"]
        sampler = run_params["sampler"]
        plot = True
        custom = run_params.get("custom", "")
        vary_z = run_params.get("vary_z", "")
        add_agn = run_params.get("add_agn", False)
        if add_agn:
            add_agn = "_agn"
        else:
            add_agn = ""

        rpath = "/nvme/scratch/work/tharvey/prospector/output"
        path = f"{rpath}/{field}"
        name_path = f"{sfh_model}_{sampler}"
        try:
            print(f"Am I plotting? {plot}")
            if plot:
                tfig = reader.traceplot(res)
                tfig.savefig(f"{path}/plots/{id}_{name_path}_trace.png")
                print("Saved trace plot.")
                tfig.clf()
                # Corner figure of posterior PDFs
                # cfig = reader.sub(res)
                cfig = res
                cfig.savefig(f"{path}/plots/{id}_{name_path}__phot.png", dpi=300)
                cfig.clf()
                print("Saved  plot.")

                fig = fig_obj.pretty_plot()
                if fig:
                    fig.savefig(f"{path}/plots/{id}_{name_path}_summary.png")
                    fig.clf()
                    print("Saved summary plot.")
                else:
                    print("oops")
                    print(fig)

            print("Saving fits to catalogue.")

            field_cat_path = f"{path}/cats/{name_path}_cat.fits"
            all_cat_path = f"{rpath}/cats/{name_path}_cat.fits"

            fig_obj.add_to_cat(field_cat_path)
            fig_obj.add_to_cat(all_cat_path)
            sed_path = f"{path}/seds/{name_path}_sed"
            print(f"{path}/seds/{name_path}_sed.ecsv")
            fig_obj.save_sed(sed_path, save=True)
            print("Saved best-fitting SED.")

        except FileNotFoundError as e:  # IndexError normally
            print(
                "Somehing broke when saving plots. Continuing to next run, investigate when done."
            )
            print(e)


def main_parallel(config):
    load = config.get("load", False)

    if not load:
        # Get the default argument parser
        # parser = prospect_args.get_parser()
        # args = parser.parse_args()
        # run_params = vars(args)

        run_params = {}

        run_params["param_file"] = __file__

        # Add sampling parameters
        run_params.update(sampling_default_settings)

        # Update run_params with user config
        run_params.update(config)

        # Check that required parameters are present
        sanity_check_run_params(run_params)

        # Get options for the sampler
        path = run_params["run_path"]
        id = run_params["OBJID"]

        # Run settings
        run_params["verbose"] = 2
        obs, model, sps, noise = build_all(run_params)

        # Add SPS libraries to run_params
        run_params["sps_libraries"] = sps.ssp.libraries

        sampler = run_params["sampler"]
        assert sampler == "dynesty", "Only dynesty is supported for parallel runs."

        custom = run_params.get("custom", "")

        # spec, phot, mfrac = model.predict(model.theta, obs=obs, sps=sps)

        # Do emcee/dynesty
        run_params["optimize"] = False
        if run_params["sampler"] == "dynesty":
            run_params["dynesty"] = True
        elif run_params["sampler"] == "emcee":
            run_params["emcee"] = True
        else:
            print(f"Sampler {sampler} not recognized.")
            return False

        # Set up MPI communication
    try:
        import mpi4py
        from mpi4py import MPI
        from schwimmbad import MPIPool

        mpi4py.rc.threads = False
        mpi4py.rc.recv_mprobe = False

        comm = MPI.COMM_WORLD
        size = comm.Get_size()

        withmpi = comm.Get_size() > 1
    except ImportError:
        print("Failed to start MPI; are mpi4py and schwimmbad installed? Proceeding without MPI.")
        withmpi = False

    # Evaluate SPS over logzsol grid in order to get necessary data in cache/memory
    # for each MPI process. Otherwise, you risk creating a lag between the MPI tasks
    # caching data depending which can slow down the parallelization
    if (withmpi) & ("logzsol" in model.free_params):
        dummy_obs = dict(filters=None, wavelength=None)

        logzsol_prior = model.config_dict["logzsol"]["prior"]
        lo, hi = logzsol_prior.range
        logzsol_grid = np.around(np.arange(lo, hi, step=0.1), decimals=2)

        sps.update(**model.params)  # make sure we are caching the correct IMF / SFH / etc
        for logzsol in logzsol_grid:
            model.params["logzsol"] = np.array([logzsol])
            _ = model.predict(model.theta, obs=dummy_obs, sps=sps)

    # ensure that each processor runs its own version of FSPS
    # this ensures no cross-over memory usage
    from functools import partial

    from prospect.fitting import lnprobfn

    lnprobfn_fixed = partial(lnprobfn, sps=sps)

    # rename model to input_model in run_params
    run_params["input_model"] = model
    run_params.pop("model")

    remove_obs(run_params)
    fix_params(run_params)

    # attempt to json serialize the run_params
    try:
        json.dumps(run_params)
    except TypeError:
        print("Run_params not serializable. Exiting.")
        return False

    if withmpi:
        run_params["using_mpi"] = True
        with MPIPool() as pool:
            # The dependent processes will run up to this point in the code
            if not pool.is_master():
                pool.wait()
                sys.exit(0)
            nprocs = pool.size

            # The parent process will oversee the fitting
            output = fit_model(
                obs,
                model,
                sps,
                noise,
                pool=pool,
                queue_size=nprocs,
                lnprobfn=lnprobfn_fixed,
                **run_params,
            )
    else:
        # without MPI we don't pass the pool
        output = fit_model(obs, model, sps, noise, lnprobfn=lnprobfn_fixed, **run_params)

    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path), exist_ok=True)

    writer.write_hdf5(
        path,
        run_params,
        model,
        obs,
        output["sampling"][0],
        output["optimization"][0],
        tsample=output["sampling"][1],
        toptimize=output["optimization"][1],
        sps=sps,
    )

    try:
        path.close()
    except AttributeError:
        pass


def loadprospector(file_path):
    res, obs, model = reader.results_from(file_path)
    # Trace plots
    # tfig = reader.traceplot(res)
    # Corner figure of posterior PDFs
    cfig = reader.sub(res)
    plt.show()


def run_prospector_expanse(input_file_json):
    # This version loads a json file with all the parameters

    with open(input_file_json) as f:
        config = json.load(f)

    main(config)


def run_prospector_on_cat(
    file_path,
    model,
    run_name="auto",
    n_jobs=6,
    load=False,
    sampler="dynesty",
    filter_val=None,
    filter_field=None,
    run_dir="prospector/",
    existing_behaviour="skip",
    id_column="NUMBER",
    z_column=None,
    field_column=None,
    flux_column="FLUX_APER_band",
    flux_err_column="FLUXERR_APER_band",
    flux_units="uJy",
    skip_errors=False,
    custom="",
    plot=True,
    min_percentage_err=10,
    **kwargs,
):
    """Sets config for running Prospector

    Args:
        file_path (str, optional): Path of catalog to run. All rows will be run. Defaults to '/nvme/scratch/work/tharvey/catalogs/robust_and_good_gal_all_criteria_3sigma_all_fields.fits'.
        n_jobs (int, optional): Number of cores to use. Defaults to 6.
        load (bool, optional): Whether to load a pre-existing run or start a new one. Defaults to False.
        existing_behaviour - What to do if a run already exists. Options are 'skip', 'overwrite', 'unique'. Default 'skip'. If 'unique', will append a unique timestamp string to the run name.
        sampler (str, optional): Fitting algorithim used. Dynesty recommended. Defaults to 'dynesty'.
        filter_val, (flexible, optional): Along with filter_field, lets you filter the input catalog by any column. If filter_val is a list or array, then the catalog is filtered on the filter_field column to only objects which match that set (e.g. IDs).
            If a tuple of length 2, then the galaxies with [0] < filter_field <[1] are selected. If other, then a match is required to the filter_field column. Default None
        filter_field, (string, optional): Column name from input catalog. Used to filter catalog.
        skip_errors, (bool, optional): If fitting multiple galaxies and one crashes during sampling, will attempt to continue with others. Defaults to False.
        custom (str, optional): Append custom string to all paths - used for differentiating runs with parameters otherwise not in header. Default ''.
        plot (bool, optional.): Whether to generate , trace, SED plots etc. Default True.
        field_column - field column name in catalog. Default 'FIELDNAME'. If no FIELDNAME column, FIELDNAME is set to value of field_column.
        min_percentage_err - Minimum percentage error on photometry to use. Default 10.
    """
    # file_path = '/nvme/scratch/work/tharvey/catalogs/robust_and_good_gal_all_criteria_3sigma_all_fields.fits'
    # whether to load a pre-existing fit or rerun
    catalog = Table.read(file_path)
    catalog = filter_catalog(catalog, filter_val, filter_field)

    # needs to be true if loading previous runs
    if load == True:
        existing_behaviour = "overwrite"

    list_params = []

    print(f"Running Prospector on {len(catalog)} galaxies.")

    for row in catalog:
        id = row[id_column]

        run_params_init = {}
        run_params_init["OBJID"] = id

        try:
            field = row[field_column]
        except KeyError:
            field = field_column

        for param in [
            "field",
            "min_percentage_err",
            "model",
            "sampler",
            "use_builtin",
            "skip_errors",
            "plot",
            "custom",
            "flux_column",
            "flux_err_column",
            "flux_units",
        ]:
            run_params_init[param] = copy.deepcopy(locals()[param])

        run_params_init["catalog_path"] = file_path

        if run_name == "auto":
            model_name = generate_model_name_from_run_params(run_params_init["model"])
            run_name = f"{id}_{model_name}_{sampler}{custom}"

        if not run_name.endswith(".h5"):
            run_name = f"{run_name}.h5"

        if field is not None:
            field_val = f"/{field}/"
        else:
            field_val = "/"

        path = f"{run_dir}{field_val}{run_name}"

        if existing_behaviour == "unique" and os.path.exists(f"{path}.h5"):
            ts = time.strftime("%y%b%d-%H.%M", time.localtime())
            path = f"{path}_{ts}"

        path = f"{path}.h5"
        run_params_init["run_path"] = path

        if not os.path.exists(path) or existing_behaviour == "overwrite":
            list_params.append(run_params_init)

    print(f"{len(list_params)} to fit.")

    if len(list_params) == 0:
        if rank == 0:
            print(
                "No galaxies found to run. Either ID is not in catalog or the fit already exists and overwrite_existing is False."
            )
    elif len(list_params) > 1 and n_jobs > 1:
        print(f"Running in parallel with {n_jobs} cores. 1 galaxy per core.")
        Parallel(n_jobs=n_jobs)(delayed(main)(**run_param) for run_param in list_params)
    elif n_jobs == 1 and size > 1:
        if rank == 0:
            print(f"Running in multithreaded serial, fitting one galaxy over {size} cores.")
        for i in list_params:
            main_parallel(i)
    elif n_jobs == 1 and size == 1:
        for i in list_params:
            main(i)

    print("Fitting complete.")


def run_prospector_on_spec_phot(
    spec_path: Union[str, List[str]],  # Path or list of paths to spectrum file.
    model: Union[dict, List[dict]],  # Model or list of models to fit.
    phot_path: Optional[
        str
    ] = None,  # Optional Photometry catalog or list of paths to photometry catalogs
    set_IDs: Optional[
        Union[str, List[str]]
    ] = None,  # Optional list of IDs - not needed if ID column exists in catalog.
    run_name: str = "auto",  # Name of run. If auto, will generate from model and sampler
    load: bool = False,  # Whether to load a pre-existing run or start a new one
    sampler: str = "dynesty",  # Fitting algorithim used. Dynesty or emcee. Default dynesty.
    filter_val: Optional[
        Union[str, List[str]]
    ] = None,  # Filter value to select galaxies. If list, will select galaxies with IDs in list. If string, will select galaxies with that value in filter_field.
    filter_field: Optional[str] = None,  # Column name to filter on.
    run_dir: str = "prospector/",  # Directory to save output
    existing_behaviour: str = "skip",  # What to do if a run already exists. Options are 'skip', 'overwrite', 'unique'. Default 'skip'. If 'unique', will append a unique timestamp string to the run name.
    id_column: str = "NUMBER",  # Column name for object ID
    z_val_or_column: Optional[Union[str, float]] = None,  # Redshift value or column name.
    field_val_or_column: Optional[str] = None,  # Column name for field
    ignore_filters: List[str] = [],  # List of filters to ignore in photometry
    spec_mask: Optional[str] = None,  # Mask to apply to spectrum
    phot_flux_column: str = "FLUX_APER_band",  # Column name for photometry flux
    phot_flux_err_column: str = "FLUXERR_APER_band",  # Column name for photometry flux error
    spec_wav_units: Optional[
        u.Unit
    ] = None,  # Units of spectrum wavelength - if not provided will be inferred from file
    phot_flux_units: str = "uJy",  # Units of photometry flux
    spec_flux_units: Optional[
        u.Unit
    ] = None,  # Units of spectrum flux - if not provided will be inferred from file
    skip_errors: bool = False,  # If fitting multiple galaxies and one crashes during sampling, will attempt to continue with others. Defaults to False.
    custom: str = "",  # Append custom string to all paths - used for differentiating runs with parameters otherwise not in header. Default ''.
    plot: bool = True,  # Whether to generate , trace, SED plots etc. Default True.
    min_phot_percentage_err: int = 10,
    n_jobs: int = 1,  # Number of cores to use. Defaults to 1.
):  # Minimum percentage error on photometry to use. Default 10.
    """
    Sets config for running Prospector. This function is designed to fit one or more galaxies with spectroscopy and optionally photometry.
    """

    if size > 1:
        assert n_jobs == 1, "n_jobs must be 1 if running through mpirun."

    if type(spec_path) == str:
        spec_fit_number = 1
        spec_path = [spec_path]
    elif type(spec_path) == list:
        spec_fit_number = len(spec_path)

    if type(model) == list:
        assert (
            len(model) == spec_fit_number
        ), "Number of models must match number of spectrum files."
    else:
        model = [model] * spec_fit_number

    if set_IDs is not None:
        if type(set_IDs) is str:
            set_IDs = [set_IDs]
        assert len(set_IDs) == spec_fit_number, "Number of IDs must match number of spectrum files."

    if phot_path is not None:
        catalog = Table.read(phot_path)
        if filter_val is not None and filter_field is not None:
            catalog = filter_catalog(catalog, filter_val, filter_field)

        assert (
            len(catalog) == spec_fit_number
        ), "Number of galaxies in catalog must match number of spectrum files."

    list_params = []
    for i in range(spec_fit_number):
        model_i = model[i]
        spec_path_i = spec_path[i]
        if phot_path is not None:
            row = catalog[i]

        if set_IDs is not None:
            id = set_IDs[i]
        elif phot_path is not None and id_column in row.colnames:
            id = row[id_column]
        elif spec_fit_number == 1:
            id = run_name
            if id == "auto":
                raise ValueError(
                    "Please set either ID column in catalog, set_IDs, or manual run name"
                )
        else:
            id = f"{run_name}_{i}"

        run_params_init = {}
        run_params_init["OBJID"] = id

        try:
            field = row[field_val_or_column]
        except (KeyError, UnboundLocalError):
            field = field_val_or_column

        if phot_path is not None:
            filters = find_bands(row)
            filters = [f for f in filters if f not in ignore_filters]

            fluxes, errors = provide_phot(
                row,
                bands=filters,
                flux_wildcard=phot_flux_column,
                error_wildcard=phot_flux_err_column,
                min_percentage_error=min_phot_percentage_err,
                flux_unit=phot_flux_units,
            )
            run_params_init["flux"] = fluxes
            run_params_init["flux_err"] = errors
            run_params_init["filters"] = (
                filters  # this probably won't work as filter names won't match. Maybe an SVO lookup?
            )
            run_params_init["flux_units"] = phot_flux_units
            run_params_init["catalog_path"] = phot_path

        if z_val_or_column is not None:
            if (
                type(z_val_or_column) == str
                and phot_path is not None
                and z_val_or_column in row.colnames
            ):
                z = row[z_val_or_column]
            elif type(z_val_or_column) in [float, int]:
                z = z_val_or_column
            elif type(z_val_or_column) in [list, np.ndarray]:
                z = z_val_or_column[i]

            zred = model_i["zred"]
            if type("zred") is dict:
                if "mean" in zred:
                    zred["mean"] = z
                else:
                    zred["init"] = z

            else:
                zred = z

        wav, flux, flux_err = load_spectra(
            spec_path_i, input_flux_units=spec_flux_units, input_wav_units=spec_wav_units
        )
        run_params_init["model"] = model_i
        run_params_init["wavelength"] = wav
        run_params_init["spectrum"] = flux
        run_params_init["unc"] = flux_err
        if spec_mask is not None:
            run_params_init["mask"] = spec_mask

        if "meta" not in run_params_init:
            run_params_init["meta"] = {}

        meta = run_params_init["meta"]
        # Meta keys are stored in obs dictionary.
        meta["spec_path"] = spec_path_i
        meta["time"] = time.strftime("%y%b%d-%H.%M", time.localtime())
        meta["min_phot_percentage_err"] = min_phot_percentage_err
        meta["phot_path"] = phot_path
        meta["z_val_or_column"] = z_val_or_column
        meta["field"] = field
        if phot_path is not None:
            meta["phot_path"] = phot_path
            meta["phot_flux_column"] = phot_flux_column
            meta["phot_flux_err_column"] = phot_flux_err_column
            meta["phot_flux_units"] = phot_flux_units
            meta["ignore_filters"] = ignore_filters

        for param in [
            "field",
            "min_phot_percentage_err",
            "sampler",
            "skip_errors",
            "plot",
            "custom",
        ]:
            run_params_init[param] = copy.deepcopy(locals()[param])

        if run_name == "auto":
            model_name = generate_model_name_from_run_params(run_params_init["model"])
            run_name = f"{model_name}_{sampler}{custom}"

        if not run_name.endswith(".h5"):
            run_name = f"{run_name}.h5"

        if field is not None:
            field_val = f"/{field}/"
        else:
            field_val = "/"

        path = f"{run_dir}{field_val}{id}/{run_name}"

        print(f"Saving to {path}")

        if existing_behaviour == "unique" and os.path.exists(f"{path}.h5"):
            ts = time.strftime("%y%b%d-%H.%M", time.localtime())
            path = f"{path}_{ts}"

        run_params_init["run_path"] = path

        if not os.path.exists(path) or existing_behaviour == "overwrite":
            list_params.append(run_params_init)

    if rank == 0:
        print(f"{len(list_params)} to fit.")

    if len(list_params) == 0:
        if rank == 0:
            print(
                "No galaxies found to run. Either ID is not in catalog or the fit already exists and overwrite_existing is False."
            )
    elif len(list_params) > 1 and n_jobs > 1:
        print(f"Running in parallel with {n_jobs} cores. 1 galaxy per core.")
        Parallel(n_jobs=n_jobs)(delayed(main)(**run_param) for run_param in list_params)
    elif n_jobs == 1 and size > 1:
        if rank == 0:
            print(f"Running in multithreaded serial, fitting one galaxy over {size} cores.")
        for i in list_params:
            main_parallel(i)
    elif n_jobs == 1 and size == 1:
        for i in list_params:
            main(i)

    if rank == 0:
        print("Fitting complete.")
