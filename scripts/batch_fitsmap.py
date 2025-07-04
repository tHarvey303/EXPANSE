from EXPANSE.utils import create_fitsmap, PhotometryBandInfo, FieldInfo
import os

subfolders = {
    "NEP-": "NEP",
    "JADES-DR3-": "JADES",
    "PRIMER-": "PRIMER",
    "G165": "PEARLS",
    "G191": "PEARLS",
    "CLIO": "PEARLS",
    "CEERSP": "CEERS",
    "COSMOS-Web-": "COSMOS-Web",
}

survey_properties = [
    ["NEP-1", "v14", ["ACS_WFC", "NIRCam"]],
    ["NEP-2", "v14", ["ACS_WFC", "NIRCam"]],
    ["NEP-3", "v14", ["ACS_WFC", "NIRCam"]],
    ["NEP-4", "v14", ["ACS_WFC", "NIRCam"]],
    ["JADES-DR3-GS-North", "v13", ["ACS_WFC", "NIRCam"]],
    ["JADES-DR3-GS-South", "v13", ["ACS_WFC", "NIRCam"]],
    ["JADES-DR3-GS-East", "v13", ["ACS_WFC", "NIRCam"]],
    ["JADES-DR3-GS-West", "v13", ["ACS_WFC", "NIRCam"]],
    ["JADES-DR3-GN-Deep", "v13", ["ACS_WFC", "NIRCam"]],
    ["JADES-DR3-GN-Medium", "v13", ["ACS_WFC", "NIRCam"]],
    ["JADES-DR3-GN-Parallel", "v13", ["ACS_WFC", "NIRCam"]],
    ["PRIMER-COSMOS", "v12", ["ACS_WFC", "NIRCam"]],
    ["PRIMER-UDS", "v12", ["ACS_WFC", "NIRCam"]],
    ["G165", "v11", ["NIRCam"]],
    ["G191", "v11", ["NIRCam"]],
    ["NGDEEP2", "v11", ["NIRCam"]],
    ["CEERSP1", "v9", ["ACS_WFC", "NIRCam"]],
    ["CEERSP2", "v9", ["ACS_WFC", "NIRCam"]],
    ["CEERSP3", "v9", ["ACS_WFC", "NIRCam"]],
    ["CEERSP4", "v9", ["ACS_WFC", "NIRCam"]],
    ["CEERSP5", "v9", ["ACS_WFC", "NIRCam"]],
    ["CEERSP6", "v9", ["ACS_WFC", "NIRCam"]],
    ["CEERSP7", "v9", ["ACS_WFC", "NIRCam"]],
    ["CEERSP8", "v9", ["ACS_WFC", "NIRCam"]],
    ["CEERSP9", "v9", ["ACS_WFC", "NIRCam"]],
    ["CEERSP10", "v9", ["ACS_WFC", "NIRCam"]],
]

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

forced_phot_band = ["F277W", "F356W", "F444W"]
aper_diam = 0.32
min_flux_pc_err = 10.0
filter_field = "Austin+25_EAZY_fsps_larson_zfree_0.32as"
overwrite = False

morgan_version_to_dir = {
    "v8b": "mosaic_1084_wispfix",
    "v8c": "mosaic_1084_wispfix2",
    "v8d": "mosaic_1084_wispfix3",
    "v9": "mosaic_1084_wisptemp2",
    "v10": "mosaic_1084_wispscale",
    "v11": "mosaic_1084_wispnathan",
    "v12": "mosaic_1210_wispnathan",
    "v12test": "mosaic_1210_wispnathan_test",  # not sure if this is needed?
    "v13": "mosaic_1293_wispnathan",
    "v14": "mosaic_1364_wispnathan",
}


def main(survey, version, instrument_names):
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

    reducer = "austind"

    catalogue_path = f"/raid/scratch/work/{reducer}/GALFIND_WORK/Catalogues/{version}/{instrument_path}/{survey}/({aper_diam})as/{survey}_MASTER_Sel-{forced_phot_path}_{version}.fits"

    galaxy_info = []

    dir_version = morgan_version_to_dir.get(version)

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
                        seg_path=f"/raid/scratch/work/{reducer}/GALFIND_WORK/SExtractor/ACS_WFC/{version}/{survey}/MAP_RMS/segmentation/",
                    )
                elif instrument == "NIRCam":
                    band_info = PhotometryBandInfo(
                        band_name=band,
                        survey=survey,
                        image_path=f"/raid/scratch/data/jwst/{survey}/NIRCam/{dir_version}/30mas/",
                        seg_path=f"/raid/scratch/work/{reducer}/GALFIND_WORK/SExtractor/NIRCam/{version}/{survey}/MAP_RMS/segmentation/",
                    )
                else:
                    raise Exception(
                        f"Instrument {instrument} not recognized. Possible instruments: {list(possible_bands.keys())}"
                    )

                galaxy_info.append(band_info)

            except ValueError as e:
                print(f"Skipping band {band} for instrument {instrument} in survey {survey}: {e}")
                continue

    field_info = FieldInfo(galaxy_info)

    plot_folder = f"/raid/scratch/work/{reducer}/GALFIND_WORK/Plots/{version}/{instrument_path}/{survey}/SED_plots/{aper_diam}as/"

    if not os.path.exists(plot_folder):
        print(f"Plot folder {plot_folder} does not exist. Please check the path.")

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

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
    for survey, version, instruments in survey_properties:
        try:
            main(survey, version, instruments)
        except Exception as e:
            print(
                f"Failed to create fitsmap for {survey} version {version} with instruments {instruments}. Skipping."
            )
            print(f"Error: {e}")
            continue
