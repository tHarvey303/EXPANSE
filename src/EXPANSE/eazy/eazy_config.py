import os
import astropy.units as u

filter_codes = {
    "F070W": 36,
    "F090W": 1,
    "F115W": 2,
    "F140M": 37,
    "F150W": 3,
    "F162M": 38,
    "F182M": 39,
    "F200W": 4,
    "F210M": 40,
    "F250M": 41,
    "F277W": 5,
    "F300M": 42,
    "F335M": 43,
    "F356W": 6,
    "F360M": 44,
    "F410M": 7,
    "F430M": 45,
    "F444W": 8,
    "F460M": 46,
    "F480M": 47,
    "F435W": 22,
    "F606W": 23,
    "F625W": 48,
    "F775W": 49,
    "F814W": 24,
    "F850LP": 50,
    "F105W": 25,
    "F125W": 26,
    "F140W": 27,
    "F160W": 28,
    "F0560W": 13,
    "F0770W": 14,
    "F1000W": 15,
    "F1130W": 16,
    "F1280W": 17,
    "F1500W": 18,
    "F1800W": 19,
    "F2100W": 20,
    "F2550W": 21,
}


def make_params(
    catalog_path,
    output_directory,
    template_name="fsps_larson",
    template_file=None,
    z_step=0.01,
    z_min=0.01,
    z_max=20,
    add_cgm=False,
    fix_zspec=False,
    cat_flux_unit=u.uJy,
    template_dir="",
):
    # JWST filter_file

    # path = pathlib.Path(os.getenv('EAZYCODE'))
    # currentdir = os.getcwd()
    # path_rel = str(path.relative_to(currentdir))

    templates_dict = {
        "fsps_larson": "templates/LarsonTemplates/tweak_fsps_QSF_12_v3_newtemplates.param",
        "fsps": "templates/fsps_full/tweak_fsps_QSF_12_v3.param",
        "nakajima_full": "templates/Nakajima2022/tweak_fsps_QSF_12_v3_larson_nakajima_all.param",
        "nakajima_subset": "templates/Nakajima2022/tweak_fsps_QSF_12_v3_larson_nakajima_subset.param",
        "BC03": "templates/bc03_chabrier_2003.param",
        "HOT_45K": "templates/fsps-hot/45k/fsps_45k.param",
        "HOT_60K": "templates/fsps-hot/60k/fsps_60k.param",
        "jades": "templates/jades/jades.param",
    }

    if template_file is None:
        template_file = os.path.join(
            template_dir, templates_dict[template_name]
        )
    else:
        raise ValueError(
            "Template name not recognized. Please provide a template file path."
        )

    params = {}

    params["TEMPLATES_FILE"] = template_file  # Template file

    # Get this code's path
    this_path = os.path.dirname(os.path.abspath(__file__))

    params["FILTERS_RES"] = os.path.join(this_path, "jwst_nircam_FILTER.RES")

    # Galactic extinction
    params["MW_EBV"] = 0  # Setting MW E(B-V) extinction
    params["CAT_HAS_EXTCORR"] = (
        False  # Catalog already corrected for reddening?
    )

    params["ADD_CGM"] = add_cgm  # Add Asada CGM damping wings?

    # Redshift stuff
    params["Z_STEP"] = z_step  # Setting photo-z step
    params["Z_MIN"] = z_min  # Setting minimum Z
    params["Z_MAX"] = z_max  # Setting maxium Z

    # Errors
    # params['WAVELENGTH_FILE'] = os.path.join(path_rel, 'templates/lambda.def')  # Wavelength grid definition file
    # params['TEMP_ERR_FILE'] = os.path.join(path_rel, 'templates/TEMPLATE_ERROR.eazy_v1.0') # Template error definition file
    params["TEMP_ERR_FILE"] = os.path.join(
        this_path, "TEMPLATE_ERROR.eazy_v1.0"
    )  # Template error definition file
    params["TEMP_ERR_A2"] = 0  # Template error amplitude
    params["SYS_ERR"] = 0
    # Priors
    params["APPLY_PRIOR"] = "n"  # Apply priors?
    params["PRIOR_ABZP"] = cat_flux_unit.to(u.ABmag)  # AB zeropoint

    print(f'Zeropoint: {params["PRIOR_ABZP"]}')
    params["PRIOR_FILTER"] = (
        28  # Filter from FILTER_RES corresponding to the columns in PRIOR_FILE
    )
    params["PRIOR_FILE"] = ""

    params["FIX_ZSPEC"] = fix_zspec  # Fix redshift to catalog zspec
    params["IGM_SCALE_TAU"] = 1.0  # Scale factor times Inoue14 IGM tau

    params["N_MIN_COLORS"] = 2  # Default is 5

    # Input files
    # -------------------------------------------------------------------------------------------------------------

    params["CATALOG_FILE"] = catalog_path

    return params
