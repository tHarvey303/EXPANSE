"""Automated script to create fitsmaps for multiple surveys and versions.

TODO:
- Fix RGB generation (needs to crop region used for scaling)
- Rename files to have nicer names
- Better catalogue information overlaid (more plots?)
    - Add DJA overlapping catalogues

"""

from EXPANSE.utils import create_fitsmap, PhotometryBandInfo, FieldInfo
import os
import traceback
from astropy.io import fits
import json


def dump_info_json(
    field_info,
    band="F444W",
    out_folder=".",
    keys=["DATE-OBS", "PI_NAME", "PROGRAM", "TARG_RA", "TARG_DEC"],
):
    """
    Dumps the specified keys from the photometry band info to a JSON file.
    """

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    im_path = field_info.im_paths[band]

    hdu = fits.open(im_path)
    header = hdu[0].header

    info = {key: header.get(key, "N/A") for key in keys}

    for key in info:
        if isinstance(info[key], float):
            info[key] = round(info[key], 3)

    info["Survey"] = field_info.survey
    info["Num. Bands"] = str(len(field_info))

    with open(os.path.join(out_folder, "info.json"), "w") as f:
        json.dump(info, f, indent=4)


subfolders = {
    "NEP-": "NEP",
    "JADES-DR3-": "JADES",
    "PRIMER-": "PRIMER",
    "G165": "PEARLS",
    "G191": "PEARLS",
    "CLIO": "PEARLS",
    "MACS-1423": "CANUCS",
    "ABELL370": "CANUCS",
    "WHL0137": "CANUCS",
    "MACS-1149": "CANUCS",
    "MACS-0416": "PEARLS",
    "CEERSP": "CEERS",
    "COSMOS-Web-": "COSMOS-Web",
    "JOF": "JADES",
    "WHL0137": "PEARLS",
}

# Caio
# MACS-1423 - v14 - NIRCam
# Abell 370 - v14 - NIRCam

# Me
# GRB230207A - v9 - NIRCam

# James
# COSOMOS-Web
# COSMOS-3D

# Unknown - MACS0416, MACS0417, SMACS0723, GLASS, MACS-1149

survey_properties = [
    ["CEERSP1", "v14", ["ACS_WFC", "NIRCam"], "austind"],  # done
    ["CEERSP2", "v14", ["ACS_WFC", "NIRCam"], "austind"],  # done
    ["CEERSP3", "v14", ["ACS_WFC", "NIRCam"], "austind"],  # done
    ["CEERSP4", "v14", ["ACS_WFC", "NIRCam"], "austind"],  # done
    ["CEERSP5", "v14", ["ACS_WFC", "NIRCam"], "austind"],  # done
    ["CEERSP6", "v14", ["ACS_WFC", "NIRCam"], "austind"],  # done
    ["CEERSP7", "v14", ["ACS_WFC", "NIRCam"], "austind"],  # done
    ["CEERSP8", "v14", ["ACS_WFC", "NIRCam"], "austind"],  # done
    ["CEERSP9", "v14", ["ACS_WFC", "NIRCam"], "austind"],  # done
    ["CEERSP10", "v14", ["ACS_WFC", "NIRCam"], "austind"],  # done
    ["NEP-1", "v14", ["ACS_WFC", "NIRCam"], "austind"],  # done
    ["NEP-2", "v14", ["ACS_WFC", "NIRCam"], "austind"],  # done
    ["NEP-3", "v14", ["ACS_WFC", "NIRCam"], "austind"],  # done
    ["NEP-4", "v14", ["ACS_WFC", "NIRCam"], "austind"],  # done
    ["JADES-DR3-GS-North", "v13", ["ACS_WFC", "NIRCam"], "austind"],  # done
    ["JADES-DR3-GS-South", "v13", ["ACS_WFC", "NIRCam"], "austind"],  # done
    ["JADES-DR3-GS-East", "v13", ["ACS_WFC", "NIRCam"], "austind"],  # done
    ["JADES-DR3-GS-West", "v13", ["ACS_WFC", "NIRCam"], "austind"],  # done
    ["JADES-DR3-GN-Deep", "v13", ["ACS_WFC", "NIRCam"], "austind"],  # done
    ["JADES-DR3-GN-Medium", "v13", ["ACS_WFC", "NIRCam"], "austind"],  # done
    ["JADES-DR3-GN-Parallel", "v13", ["ACS_WFC", "NIRCam"], "austind"],  # done
    ["PRIMER-COSMOS", "v12", ["ACS_WFC", "NIRCam"], "austind"],
    ["PRIMER-UDS", "v12", ["ACS_WFC", "NIRCam"], "austind"],
    ["G165", "v11", ["NIRCam"], "austind"],  # done
    ["G191", "v11", ["NIRCam"], "austind"],  # done
    ["NGDEEP2", "v11", ["NIRCam"], "austind"],  # done
    ["JOF", "v11", ["ACS_WFC", "NIRCam"], "austind"],  # done
    ["CEERSP1", "v9", ["ACS_WFC", "NIRCam"], "austind"],  # done
    ["CEERSP2", "v9", ["ACS_WFC", "NIRCam"], "austind"],  # done
    ["CEERSP3", "v9", ["ACS_WFC", "NIRCam"], "austind"],  # done
    ["CEERSP4", "v9", ["ACS_WFC", "NIRCam"], "austind"],  # done
    ["CEERSP5", "v9", ["ACS_WFC", "NIRCam"], "austind"],  # done
    ["CEERSP6", "v9", ["ACS_WFC", "NIRCam"], "austind"],  # done
    ["CEERSP7", "v9", ["ACS_WFC", "NIRCam"], "austind"],  # done
    ["CEERSP8", "v9", ["ACS_WFC", "NIRCam"], "austind"],  # done
    ["CEERSP9", "v9", ["ACS_WFC", "NIRCam"], "austind"],  # done
    ["CEERSP10", "v9", ["ACS_WFC", "NIRCam"], "austind"],  # done
]
survey_properties = [
    ["G165", "v11", ["NIRCam"], "austind"],  # done
    ["G191", "v11", ["NIRCam"], "austind"],  # done
    ["WHL0137", "v12", ["NIRCam"], "austind"],
    # ["MACS1423", "v14", ["NIRCam"], "goolsby"],  # done
    # ['MACS-0416', "v9", ["ACS_WFC", "NIRCam"], "austind"],
    # ['MACS-1149', 'v11', ['ACS_WFC', 'NIRCam'], 'austind'],
    # ["ABELL370", "v14", ["NIRCam"], "goolsby"],  # done
    # ["GRB230207A-FULL", "v12", ["NIRCam"], "tharvey"],  # done
    # ["GLIMPSE", "v14_caio_v2", ["ACS_WFC", "NIRCam"], "goolsby"],  # done
    # ["SMACS-0723", "v9", ['ACS_WFC', "NIRCam"], "austind"],
    # ["CLIO", "v9", ["NIRCam"], "austind"],  # done
]

"""
for row in ["A", "B"]:
    for i in range(17):
        field = f"COSMOS-Web-{i}{row}"
        survey_properties.append(
            [
                field,
                "COSMOS-3D_0.32as",
                [
                    "ACS_WFC",
                    "NIRCam",
                ],
                "jarcidia",
            ]
        )
"""
forced_phot_band = ["F277W", "F356W", "F444W"]
fitsmap_dir = "/nvme/scratch/work/tharvey/fitsmap/"
possible_bands = {
    "ACS_WFC": ["F435W", "F606W", "F775W", "F814W", "F850LP"],
    "NIRCam": [
        "F070W",
        "F090W",
        "F115W",
        "F150W",
        "F162M",
        "F182M",
        "F200W",
        "F210M",
        "F250M",
        "F277W",
        "F300M",
        "F335M",
        "F356W",
        "F360M",
        "F410M",
        "F430M",
        "F444W",
        "F460M",
        "F480M",
    ],
}


aper_diam = "0.32"  # 0.32
min_flux_pc_err = 10.0
filter_field = f"Austin+25_EAZY_fsps_larson_zfree_{aper_diam}as"
# filter_field = f"EPOCHS_NIRCam_EAZY_fsps_larson_zfree_{aper_diam}as"
filter_field = None
overwrite = True

morgan_version_to_dir = {
    "v8b": "mosaic_1084_wispfix",
    "v8c": "mosaic_1084_wispfix2",
    "v8d": "mosaic_1084_wispfix3",
    "v9": "mosaic_1084_wisptemp2",
    "v10": "mosaic_1084_wispscale",
    "v11": "mosaic_1084_wispnathan",
    "v12": "mosaic_1210_wispnathan",  # "v12",
    "v12test": "mosaic_1210_wispnathan_test",  # not sure if this is needed?
    "v13": "mosaic_1293_wispnathan",
    "v14": "mosaic_1364_wispnathan",
    "v12a": "v12",
    "COSMOS-3D_0.32as": "COSMOS-3D_0.32as",
    "v14_caio_v2": "v14_caio_v2",
}

override_dir_version = {
    "WHL0137": "",
}
pc_dirs = {"GRB230207A-FULL": "nvme"}


def main(
    survey,
    version,
    instrument_names,
    reducer,
    forced_phot_band=["F277W", "F356W", "F444W"],
    return_info=False,
    overwrite=False,
):
    print(f"Processing {survey} with version {version} and instruments {instrument_names}")

    out_dir = f"{fitsmap_dir}"

    for subfolder, prefix in subfolders.items():
        if survey.startswith(subfolder):
            out_dir = f"{fitsmap_dir}/{prefix}/"
            break

    if os.path.exists(f"{out_dir}/{survey}_{version}/index.html") and not overwrite:
        print(f"Fitsmap for {survey} version {version} already exists. Skipping.")
        return

    instrument_path = "+".join(instrument_names)
    forced_phot_path = "+".join(forced_phot_band)

    pc_dir = pc_dirs.get(survey, "raid")

    catalogue_path = f"/{pc_dir}/scratch/work/{reducer}/GALFIND_WORK/Catalogues/{version}/{instrument_path}/{survey}/({aper_diam})as/{survey}_MASTER_Sel-{forced_phot_path}_{version}.fits"

    galaxy_info = []

    dir_version = morgan_version_to_dir.get(version)

    if survey in override_dir_version:
        dir_version = override_dir_version.get(survey)

    for instrument in instrument_names:
        if instrument not in possible_bands:
            raise ValueError(
                f"Instrument {instrument} not recognized. Possible instruments: {list(possible_bands.keys())}"
            )

        for band in possible_bands[instrument]:
            try:
                if instrument == "ACS_WFC":
                    band_info = PhotometryBandInfo(
                        band_name=band,
                        survey=survey,
                        image_path=f"/raid/scratch/data/hst/{survey}/ACS_WFC/{dir_version}/30mas/",
                        wht_path=f"/raid/scratch/data/hst/{survey}/ACS_WFC/{dir_version}/30mas/wht/",
                        err_path=f"/raid/scratch/data/hst/{survey}/ACS_WFC/{dir_version}/30mas/",
                        seg_path=f"/{pc_dir}/scratch/work/{reducer}/GALFIND_WORK/SExtractor/ACS_WFC/{version}/{survey}/MAP_RMS/segmentation/",
                        psf_path="/nvme/scratch/work/tharvey/PSFs/JOF/",  # optional
                        psf_type="star_stack",  # optional
                        psf_kernel_path="/nvme/scratch/work/tharvey/PSFs/kernels/JOF/",
                        err_hdu_ext=1,
                        im_hdu_ext=1,
                    )
                elif instrument == "NIRCam":
                    band_info = PhotometryBandInfo(
                        band_name=band,
                        survey=survey,
                        wht_path="im",
                        err_path="im",
                        image_path=f"/raid/scratch/data/jwst/{survey}/NIRCam/{dir_version}/30mas/",
                        seg_path=f"/{pc_dir}/scratch/work/{reducer}/GALFIND_WORK/SExtractor/NIRCam/{version}/{survey}/MAP_RMS/segmentation/",
                        psf_path="/nvme/scratch/work/tharvey/PSFs/JOF/",  # optional
                        psf_kernel_path="/nvme/scratch/work/tharvey/PSFs/kernels/JOF/"
                        if band != "F444W"
                        else None,
                        psf_type="star_stack",  # optional
                    )
                else:
                    raise Exception(
                        f"Instrument {instrument} not recognized. Possible instruments: {list(possible_bands.keys())}"
                    )

                galaxy_info.append(band_info)

            except ValueError as e:
                print(f"Skipping band {band} for instrument {instrument} in survey {survey}: {e}")
                continue

    # add detection image
    # /raid/scratch/work/austind/GALFIND_WORK/Stacked_Images/v13/NIRCam/JADES-DR3-GS-West/rms_err/JADES-DR3-GS-West_F277W+F356W+F444W_v13_stack.fits

    detection_image_path = f"/{pc_dir}/scratch/work/{reducer}/GALFIND_WORK/Stacked_Images/{version}/NIRCam/{survey}//{survey}_{forced_phot_path}_{version}_stack.fits"
    if not os.path.exists(detection_image_path):
        raise FileNotFoundError(
            f"Detection image {detection_image_path} does not exist. Please check the path."
        )

    detection_band_info = PhotometryBandInfo(
        band_name=forced_phot_path,
        survey=survey,
        image_path=detection_image_path,
        wht_path="im",
        err_path="im",
        seg_path=f"/{pc_dir}/scratch/work/{reducer}/GALFIND_WORK/SExtractor/NIRCam/{version}/{survey}/MAP_RMS/segmentation/",
        detection_image=True,
    )
    galaxy_info.append(detection_band_info)

    field_info = FieldInfo(galaxy_info)

    if return_info:
        return field_info

    plot_folder = f"/{pc_dir}/scratch/work/{reducer}/GALFIND_WORK/Plots/{version}/{instrument_path}/{survey}/SED_plots/{aper_diam}as/"

    if not os.path.exists(plot_folder):
        print(f"Plot folder {plot_folder} does not exist. Please check the path.")

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    dump_info_json(
        field_info,
        out_folder=f"{out_dir}/{survey}_{version}/",
    )

    create_fitsmap(
        f"{survey}_{version}",
        field_info,
        catalogue_path=catalogue_path,
        filter_field=filter_field,
        filter_val=True,
        plot_path_column=plot_folder,
        out_dir=out_dir,
    )


if __name__ == "__main__":
    for survey, version, instruments, reducer in survey_properties:
        if "COSMOS-Web" in survey:
            forced_phot_band = ["F444W"]
        try:
            main(
                survey,
                version,
                instruments,
                reducer,
                forced_phot_band=forced_phot_band,
                overwrite=overwrite,
            )
        except Exception as e:
            print(
                f"Failed to create fitsmap for {survey} version {version} with instruments {instruments}. Skipping."
            )
            print(f"Error: {e}")
            traceback.print_exc()
