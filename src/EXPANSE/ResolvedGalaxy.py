import ast
import contextlib

# can write astropy to h5
import copy
import glob
import os
import shutil
import sys
import traceback
import types
import typing
import tempfile
import time
import warnings
from io import BytesIO
from pathlib import Path
from matplotlib.patches import FancyArrow
import astropy.units as u
import h5py as h5
import matplotlib.pyplot as plt
import numpy as np
from astropy.convolution import convolve_fft
from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatLambdaCDM
from astropy.io import fits
from astropy.io.fits import Header
from astropy.io.misc.hdf5 import read_table_hdf5, write_table_hdf5
from astropy.nddata import Cutout2D, block_reduce
from astropy.table import Column, QTable, Table, vstack
from astropy.utils.exceptions import AstropyWarning
from astropy.utils.masked import Masked
from astropy.visualization import make_lupton_rgb, simple_norm
from astropy.wcs import WCS
from joblib import Parallel, delayed
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LogNorm, Normalize

# import Ellipse
from matplotlib.patches import Ellipse

# import ScalarFormatter
from matplotlib.ticker import ScalarFormatter

# import make_axis_locatable
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from scipy.interpolate import interp1d
from scipy.ndimage import zoom
from tqdm import tqdm

warnings.simplefilter("ignore", category=AstropyWarning)
import matplotlib.cm as mcm
import matplotlib.patheffects as PathEffects

# import FontProperties
from matplotlib.font_manager import FontProperties

# supress RuntimeWarning
warnings.filterwarnings("ignore", category=RuntimeWarning)


# Package imports

from .utils import (
    CLIInterface,
    is_cli,
    make_EAZY_SED_fit_params_arr,
    update_mpl,
    FieldInfo,
    PhotometryBandInfo,
    compass,
)

"""
try:
    from mpi4py import MPI

    rank = MPI.COMM_WORLD.Get_rank()
    size = MPI.COMM_WORLD.Get_size()
    from mpi4py.futures import MPIPoolExecutor

except ImportError:
    rank = 0
    size = 1"""

cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
# This class is designed to hold all the data for a galaxy, including the cutouts, segmentation maps, and RMS error maps.

"""TODO:
        1. Store aperture photometry, auto photometry from catalogue. 
        2. Pixel binning.
        3. Does ERR map need to be convolved with PSF?
"""

# What computer is this:
# Get path of this file
file_path = os.path.abspath(__file__)
db_dir = ""

if os.path.exists("/.singularity.d/Singularity"):
    computer = "singularity"
elif "nvme" in file_path:
    computer = "morgan"
elif "Users" in file_path:
    computer = "mac"
else:
    computer = "unknown"

if computer == "mac":
    bagpipes_dir = "/Users/user/Documents/PhD/bagpipes_dir/"
    print("Running on Mac.")
    bagpipes_filter_dir = bagpipes_dir + "inputs/filters/"
elif computer == "morgan":
    bagpipes_dir = "/nvme/scratch/work/tharvey/bagpipes/"
    db_dir = "/nvme/scratch/work/tharvey/dense_basis/pregrids/"
    print("Running on Morgan.")
    bagpipes_filter_dir = bagpipes_dir + "inputs/filters/"
elif computer == "singularity":
    print("Running in container.")
    bagpipes_filter_dir = "/mnt/filters/"
elif computer == "unknown":
    print("Unknown computer.")
    bagpipes_filter_dir = "filters/"

# CHECK FOR RESOlVED_GALAXY_DIR ENVIRONMENT VARIABLE
if "RESOLVED_GALAXY_DIR" in os.environ:
    resolved_galaxy_dir = os.environ["RESOLVED_GALAXY_DIR"]
else:
    # Get code directory
    resolved_galaxy_dir = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    resolved_galaxy_dir += "/galaxies/"
    if not os.path.exists(resolved_galaxy_dir):
        print("Can't find resolved galaxy directory, using default")
        print(resolved_galaxy_dir)


class ResolvedGalaxy:
    # Why is this attribute shared between all instances?

    def __init__(
        self,
        galaxy_id,
        sky_coord: SkyCoord,
        survey: str,
        bands,
        im_paths,
        im_exts,
        im_zps,
        seg_paths,
        rms_err_paths,
        rms_err_exts,
        im_pixel_scales,
        phot_imgs,
        phot_pix_unit,
        phot_img_headers,
        rms_err_imgs,
        aperture_dict=None,
        seg_imgs=None,
        galfind_version="v11",
        detection_band="F277W+F356W+F444W",
        psf_matched_data=None,
        psf_matched_rms_err=None,
        maps=None,
        binned_flux_map=None,
        binned_flux_err_map=None,
        photometry_table=None,
        sed_fitting_table=None,
        rms_background=None,
        psfs=None,
        psfs_meta=None,
        galaxy_region=None,
        psf_kernels=None,
        cutout_size=64,
        photometry_properties=None,
        photometry_meta_properties=None,
        total_photometry=None,
        dont_psf_match_bands=[],
        already_psf_matched=False,
        auto_photometry=None,
        flux_map=None,
        redshift=None,
        unmatched_data=None,
        unmatched_rms_err=None,
        unmatched_seg=None,
        meta_properties=None,
        det_data=None,
        resolved_mass=None,
        resolved_sfh=None,
        resolved_sed=None,
        resolved_sfr_10myr=None,
        resolved_sfr_100myr=None,
        h5_folder=resolved_galaxy_dir,
        psf_kernel_folder="internal",
        psf_folder="internal",
        psf_type="star_stack",
        overwrite=False,
        save_out=True,
        h5_path=None,
        interactive_outputs=None,
    ):
        self.internal_bagpipes_cache = {}

        self.galaxy_id = galaxy_id
        self.sky_coord = sky_coord
        self.survey = survey
        self.bands = bands
        self.im_paths = im_paths
        self.im_exts = im_exts
        self.im_zps = im_zps
        self.galfind_version = galfind_version
        self.detection_band = detection_band
        self.seg_paths = seg_paths
        self.rms_err_paths = rms_err_paths
        self.rms_err_exts = rms_err_exts
        self.im_pixel_scales = im_pixel_scales
        self.cutout_size = cutout_size
        self.aperture_dict = aperture_dict
        self.rms_background = rms_background
        # Actual cutouts
        self.phot_imgs = phot_imgs
        self.phot_pix_unit = phot_pix_unit
        self.phot_img_headers = phot_img_headers
        self.rms_err_imgs = rms_err_imgs
        self.seg_imgs = seg_imgs

        self.unmatched_data = unmatched_data
        self.unmatched_rms_err = unmatched_rms_err
        self.unmatched_seg = unmatched_seg

        self.meta_properties = meta_properties
        self.det_data = det_data

        self.redshift = redshift

        self.flux_map = flux_map

        self.auto_photometry = auto_photometry
        self.total_photometry = total_photometry

        self.dont_psf_match_bands = dont_psf_match_bands
        self.already_psf_matched = already_psf_matched

        self.interactive_outputs = interactive_outputs

        if photometry_properties is not None:
            self.photometry_properties = photometry_properties
            self.photometry_property_names = list(photometry_properties.keys())
            for property in photometry_properties:
                setattr(self, property, photometry_properties[property])
                setattr(
                    self,
                    f"{property}_meta",
                    photometry_meta_properties[property],
                )
                # print('Added property', property)
        else:
            self.photometry_property_names = []

        if h5_path is None:
            if not h5_folder.endswith("/"):
                h5_folder += "/"

            self.h5_path = h5_folder + f"{self.survey}_{self.galaxy_id}.h5"
        else:
            self.h5_path = h5_path

        self.save_out = save_out

        if os.path.exists(self.h5_path) and overwrite:
            print("deleting existing .h5 file.")
            os.remove(self.h5_path)

        # Check if dvipng is installed
        if shutil.which("dvipng") is None:
            # print('dvipng not found, disabling LaTeX')
            update_mpl(tex_on=False)
        else:
            update_mpl(tex_on=False)

        # Check sizes of images
        for band in self.bands:
            assert (
                self.phot_imgs[band].shape
                == (self.cutout_size, self.cutout_size)
            ), f"Image shape for {band} is {self.phot_imgs[band].shape}, not {(self.cutout_size, self.cutout_size)}"
            assert (
                self.rms_err_imgs[band].shape
                == (self.cutout_size, self.cutout_size)
            ), f"RMS error image shape for {band} is {self.rms_err_imgs[band].shape}, not {(self.cutout_size, self.cutout_size)}"

            if self.seg_imgs is not None and len(self.seg_imgs) > 0:
                assert (
                    self.seg_imgs[band].shape
                    == (self.cutout_size, self.cutout_size)
                ), f"Segmentation map shape for {band} is {self.seg_imgs[band].shape}, not {(self.cutout_size, self.cutout_size)}"
        # Bin the pixels
        if maps is not None:
            for key in maps:
                setattr(self, f"{key}_map", maps[key])
            # store names of maps
            self.maps = list(maps.keys())
        else:
            self.maps = []

        # print('type', type(pixedfit_map))
        self.psf_matched_data = psf_matched_data
        self.psf_matched_rms_err = psf_matched_rms_err
        self.psfs = psfs
        self.psfs_meta = psfs_meta
        # print(len(psfs_meta))
        self.resolved_mass = resolved_mass
        self.resolved_sfr_10myr = resolved_sfr_10myr
        self.resolved_sfr_100myr = resolved_sfr_100myr
        self.resolved_sfh = resolved_sfh
        self.resolved_sed = resolved_sed

        self.binned_flux_map = binned_flux_map
        self.binned_flux_err_map = binned_flux_err_map

        self.photometry_table = photometry_table

        self.sed_fitting_table = sed_fitting_table

        if psf_kernel_folder == "internal":
            self.psf_kernel_folder = (
                os.path.dirname(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                )
                + "/psfs/kernels/"
            )
        else:
            self.psf_kernel_folder = psf_kernel_folder

        if psf_folder == "internal":
            self.psf_folder = (
                os.path.dirname(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                )
                + "/psfs/psf_models/"
            )
        else:
            self.psf_folder = psf_folder
        # self.psf_kernels = {psf_type: {}}
        self.psf_kernels = psf_kernels

        self.gal_region = galaxy_region
        # Assume bands is in wavelength order, and that the largest PSF is in the last band
        self.use_psf_type = psf_type

        if self.save_out:
            self.dump_to_h5()

        if (
            self.psf_matched_data in [None, {}]
            or self.psf_matched_rms_err in [None, {}]
            or psf_type not in self.psf_matched_data.keys()
        ):
            # If no PSF matched data, then we need to get PSF kernels

            if psf_type == "webbpsf":
                print("Getting WebbPSF")
                self.get_webbpsf()
            elif psf_type == "star_stack":
                self.get_star_stack_psf()

            # print(f'Assuming {self.bands[-1]} is the band with the largest PSF, and convolving all bands with this PSF kernel.')
            # print(self.galaxy_id)
            print("Convolving images with PSF")
            self.convolve_with_psf(psf_type=psf_type, init_run=False)

        # if self.rms_background is None:
        #    self.estimate_rms_from_background()

        # Save to .h5

    @classmethod
    def init(
        cls,
        galaxy_id,
        survey,
        version,
        instruments=["ACS_WFC", "NIRCam"],
        excl_bands=[],
        cutout_size=64,
        forced_phot_band=["F277W", "F356W", "F444W"],
        aper_diams=[0.32] * u.arcsec,
        output_flux_unit=u.uJy,
        h5_folder=resolved_galaxy_dir,
        dont_psf_match_bands=[],
        already_psf_matched=False,
        save_out=True,
    ):
        """Initialise a galaxy object from either a .h5 file or GALFIND catalogue

        Parameters
        ----------

        galaxy_id : str, list, np.ndarray, Column
            The ID of the galaxy to load. If a list, will load multiple galaxies.
        survey : str
            The survey to load the galaxy from.
        version : str
            The version of the survey to load the galaxy from.
        instruments : list, optional
            The instruments to load the galaxy from. Default is ['ACS_WFC', 'NIRCam'].
        excl_bands : list, optional
            The bands to exclude from the galaxy. Default is [].
        cutout_size : int, optional
            The size of the cutout to load. Default is 64 pixels.
        forced_phot_band : list, optional
            The bands to use for forced photometry. Default is ['F277W', 'F356W', 'F444W'].
        aper_diams : list, optional
            The aperture diameters to use for aperture photometry. Default is [0.32] * u.arcsec.
        output_flux_unit : astropy.unit, optional
            The output flux unit to use. Default is u.uJy.
        h5_folder : str, optional
            The folder to save the .h5 files to. Default is resolved_galaxy_dir.
        dont_psf_match_bands : list, optional
            The bands to not PSF match. Default is [].
        already_psf_matched : bool, optional
            Whether the bands are already PSF matched. Default is False.
        save_out: bool, optional
            Whether to auto rewrite the .h5


        """

        if type(galaxy_id) not in [list, np.ndarray, Column]:
            galaxy_name = f"{survey}_{galaxy_id}"
            if os.path.exists(f"{h5_folder}/{galaxy_name}.h5"):
                print("Loading from .h5")
                return cls.init_from_h5(
                    galaxy_name,
                    h5_folder=h5_folder,
                    save_out=save_out,
                )
            else:
                print("Loading from GALFIND")
                return cls.init_from_galfind(
                    galaxy_id,
                    survey,
                    version,
                    instruments=instruments,
                    excl_bands=excl_bands,
                    cutout_size=cutout_size,
                    forced_phot_band=forced_phot_band,
                    aper_diams=aper_diams,
                    output_flux_unit=output_flux_unit,
                    h5_folder=h5_folder,
                    dont_psf_match_bands=dont_psf_match_bands,
                    already_psf_matched=already_psf_matched,
                )

        else:
            galaxy_names = [f"{survey}_{gal_id}" for gal_id in galaxy_id]
            found = [
                os.path.exists(f"{h5_folder}{galaxy_name}.h5")
                for galaxy_name in galaxy_names
            ]

            if all(found):
                print("Loading from .h5")
                return cls.init_multiple_from_h5(
                    galaxy_names,
                    h5_folder=h5_folder,
                    save_out=save_out,
                )
            else:
                # Load those that are found
                found_ids = [
                    gal_id for gal_id, f in zip(galaxy_id, found) if f
                ]

                if len(found_ids) > 0:
                    print("Loading some from .h5")
                    galaxy_names = [
                        f"{survey}_{gal_id}" for gal_id in found_ids
                    ]
                    found_galaxies = cls.init_multiple_from_h5(
                        galaxy_names,
                        h5_folder=h5_folder,
                        save_out=save_out,
                    )

                else:
                    found_galaxies = []

        if type(galaxy_id) in [list, np.ndarray, Column]:
            unfound_ids = [
                gal_id for gal_id in galaxy_id if gal_id not in found_ids
            ]
        else:
            unfound_ids = [galaxy_id]

        if type(cutout_size) in [list, np.ndarray, Column]:
            if type(galaxy_id) not in [list, np.ndarray, Column]:
                assert (
                    len(cutout_size) == 1
                ), f"Cutout size must be the same length as galaxy_id, {len(cutout_size)} != 1"
                filt_cutout_size = cutout_size[0]
            else:
                assert (
                    len(cutout_size) == len(galaxy_id)
                ), f"Cutout size must be the same length as galaxy_id, {len(cutout_size)} != {len(galaxy_id)}"
                filt_cutout_size = [
                    i
                    for i, f in zip(cutout_size, galaxy_id)
                    if f not in found_ids
                ]
        else:
            filt_cutout_size = cutout_size

        print(f"Loading from GALFIND for {unfound_ids}")
        unfound_galaxies = cls.init_from_galfind(
            unfound_ids,
            survey,
            version,
            instruments=instruments,
            excl_bands=excl_bands,
            cutout_size=filt_cutout_size,
            forced_phot_band=forced_phot_band,
            aper_diams=aper_diams,
            output_flux_unit=output_flux_unit,
            h5_folder=h5_folder,
            dont_psf_match_bands=dont_psf_match_bands,
            already_psf_matched=already_psf_matched,
        )

        # Match the order of the input galaxy_id
        galaxies = []
        for gal_id in galaxy_id:
            if gal_id in found_ids:
                galaxies.append(found_galaxies[found_ids.index(gal_id)])
            elif gal_id in unfound_ids:
                galaxies.append(unfound_galaxies[unfound_ids.index(gal_id)])
            else:
                raise Exception(f"Galaxy {gal_id} not found.")

        return galaxies

    @classmethod
    def init_from_galfind(
        cls,
        galaxy_id,
        survey,
        version,
        instruments=["ACS_WFC", "NIRCam"],
        excl_bands=[],
        cutout_size=64,
        forced_phot_band=["F277W", "F356W", "F444W"],
        aper_diams=[0.32] * u.arcsec,
        output_flux_unit=u.uJy,
        h5_folder=resolved_galaxy_dir,
        templates_arr=["fsps_larson"],
        lowz_zmax_arr=[[4.0, 6.0, None]],
        dont_psf_match_bands=[],
        already_psf_matched=False,
        psf_type="star_stack",
    ):
        # Imports here so only used if needed
        # from galfind import Data
        from galfind import EAZY, Catalogue
        from galfind.Catalogue_Creator import GALFIND_Catalogue_Creator

        if type(galaxy_id) in [list, np.ndarray, Column]:
            load_multiple = True
            crop_by = {"ID": galaxy_id}
        else:
            load_multiple = False
            crop_by = f"ID:{int(galaxy_id)}"
            galaxy_id = [galaxy_id]

        SED_code_arr = [EAZY()]
        SED_fit_params_arr = make_EAZY_SED_fit_params_arr(
            SED_code_arr, templates_arr, lowz_zmax_arr
        )
        # Make cat creator
        cat_creator = GALFIND_Catalogue_Creator("loc_depth", aper_diams[0], 10)
        # Load catalogue and populate galaxies
        cat = Catalogue.from_pipeline(
            survey=survey,
            version=version,
            instruments=instruments,
            aper_diams=aper_diams,
            cat_creator=cat_creator,
            SED_fit_params_arr=SED_fit_params_arr,
            forced_phot_band=forced_phot_band,
            excl_bands=excl_bands,
            loc_depth_min_flux_pc_errs=[10],
            crop_by=crop_by,
        )

        if load_multiple:
            if type(cutout_size) in [list, np.ndarray, Column]:
                assert (
                    len(cutout_size) == len(galaxy_id)
                ), f"Cutout size must be the same length as galaxy_id, {len(cutout_size)} != {len(galaxy_id)}"

                cutout_size_dict = {
                    galaxy_id: size
                    * cat.data.im_pixel_scales[forced_phot_band[0]].value
                    * u.arcsec
                    for galaxy_id, size in zip(galaxy_id, cutout_size)
                }

        if not load_multiple or type(cutout_size) not in [
            list,
            np.ndarray,
            Column,
        ]:
            cutout_size_dict = (
                cat.data.im_pixel_scales[forced_phot_band[0]]
                / u.pixel
                * cutout_size
            )

        band_properties_to_load = [
            "A_IMAGE",
            "B_IMAGE",
            "THETA_IMAGE",
            "FLUX_RADIUS",
            "MAG_AUTO",
            "MAGERR_AUTO",
            "MAG_BEST",
            "MAGERR_BEST",
            "MAG_ISO",
            "MAGERR_ISO",
        ]
        for prop in band_properties_to_load:
            cat.load_band_properties_from_cat(prop, prop)
        properties_to_load = [
            "ALPHA_J2000",
            "DELTA_J2000",
            "zbest_16_fsps_larson_zfree",
            "zbest_fsps_larson_zfree",
            "zbest_84_fsps_larson_zfree",
            "chi2_best_fsps_larson_zfree",
        ]
        for prop in properties_to_load:
            cat.load_property_from_cat(prop, prop)

        # Obtain galaxy object

        cat.make_cutouts(galaxy_id, cutout_size=cutout_size_dict)

        objects = []

        for position, gal_id in enumerate(galaxy_id):
            cutout_size_gal = (
                cutout_size_dict[gal_id] if load_multiple else cutout_size_dict
            )
            cutout_size_gal /= (
                cat.data.im_pixel_scales[forced_phot_band[0]].value * u.arcsec
            )

            # Make cutouts - this may not work currently as data.wht_types doesn't appear to be defined.

            galaxy = [gal for gal in cat.gals if gal.ID == gal_id]

            if len(galaxy) == 0:
                raise Exception(f"Galaxy {gal_id} not found")
            elif len(galaxy) > 1:
                raise Exception(f"Multiple galaxies with ID {gal_id} found")
            else:
                galaxy = galaxy[0]

            # Print all properties of galaxy
            # print(galaxy.__dict__.keys())

            cutout_paths = galaxy.cutout_paths
            # Settings things needed for init
            # Get things from Data object
            im_paths = cat.data.im_paths
            im_exts = cat.data.im_exts
            err_paths = cat.data.rms_err_paths
            rms_err_exts = cat.data.rms_err_exts
            seg_paths = cat.data.seg_paths
            im_zps = cat.data.im_zps
            im_pixel_scales = cat.data.im_pixel_scales
            bands = (
                galaxy.phot.instrument.band_names
            )  # should be bands just for galaxy!
            # cat.data.instrument.band_names # should be bands just for galaxy!
            # Get things from galaxy object
            galaxy_skycoord = galaxy.sky_coord
            # print('skycoord')
            # print(galaxy.sky_coord)
            bands_mask = galaxy.phot.flux_Jy.mask
            bands = bands[~bands_mask]
            # Get aperture photometry
            flux_aper = galaxy.phot.flux_Jy[~bands_mask]
            flux_err_aper = galaxy.phot.flux_Jy_errs[~bands_mask]
            depths = galaxy.phot.depths[~bands_mask]
            # Get the wavelegnths
            wave = galaxy.phot.wav  # [~bands_mask]

            # Get redshift
            SED_fit_params = SED_fit_params_arr[-1]
            print(
                SED_fit_params["code"].label_from_SED_fit_params(
                    SED_fit_params
                )
            )
            SED_results = galaxy.phot.SED_results[
                SED_fit_params["code"].label_from_SED_fit_params(
                    SED_fit_params
                )
            ]
            redshift = SED_results.z

            print(f"Redshift is {redshift}")

            # aperture_dict
            aperture_dict = {
                str(0.32 * u.arcsec): {
                    "flux": flux_aper,
                    "flux_err": flux_err_aper,
                    "depths": depths,
                    "wave": wave,
                }
            }
            phot_imgs = {}
            phot_pix_unit = {}
            rms_err_imgs = {}
            seg_imgs = {}
            phot_img_headers = {}

            auto_photometry = {}
            for band in bands:
                auto_photometry[band] = {}
                for prop in band_properties_to_load:
                    attr = getattr(galaxy, f"{prop}", None)
                    if attr is not None:
                        if band in attr.keys():
                            auto_photometry[band][prop] = attr[band].value

            meta_properties = {}
            for prop in properties_to_load:
                attr = getattr(galaxy, f"{prop}", None)
                if attr is not None:
                    if type(attr) in [list, np.ndarray] and len(attr) == 1:
                        attr = attr[0]
                    meta_properties[prop] = attr.value

            for band in bands:
                cutout_path = cutout_paths[band]
                with fits.open(cutout_path) as hdu:
                    assert (
                        hdu[1].header["NAXIS1"]
                        == hdu[1].header["NAXIS2"]
                        == cutout_size_gal
                    ), f"Cutout size is {hdu[1].header['NAXIS1']}, not {cutout_size_gal}"
                    # assert int(hdu[0].header['cutout_size_as'] / cat.data.im_pixel_scales[forced_phot_band[0]]) == cutout_size, f"Cutout size is {hdu[0].header['cutout_size_as'] / cat.data.im_pixel_scales[forced_phot_band[0]]}, not {cutout_size}"
                    data = hdu["SCI"].data
                    # CHECK if data is all O or NaN
                    if np.all(data == 0) or np.all(np.isnan(data)):
                        print(
                            f"All data is 0 or NaN for {band} - removing band from galaxy"
                        )
                        bands = bands[bands != band]
                        continue

                    try:
                        rms_data = hdu["RMS_ERR"].data
                    except KeyError:
                        weight_data = hdu["WHT"].data
                        rms_data = np.where(
                            weight_data == 0, 0, 1 / np.sqrt(weight_data)
                        )
                    try:
                        unit = u.Unit(hdu["SCI"].header["BUNIT"])
                        zero_point = None
                    except (KeyError, ValueError):
                        zeropoint = hdu["SCI"].header["ZEROPNT"]
                        unit = None

                    pix_scale = im_pixel_scales[band]
                    # convert to flux_unit
                    if unit == u.Unit("MJy/sr"):
                        if output_flux_unit == u.Unit("MJy/sr"):
                            data = data * u.MJy / u.sr
                            rms_data = rms_data * u.MJy / u.sr
                            unit = u.MJy / u.sr
                        else:
                            data = data * unit * pix_scale**2
                            rms_data = rms_data * unit * pix_scale**2

                            if output_flux_unit in [u.Jy, u.mJy, u.uJy, u.nJy]:
                                data = data.to(output_flux_unit)
                                rms_data = rms_data.to(output_flux_unit)
                                unit = output_flux_unit
                            elif (
                                output_flux_unit
                                == u.erg / u.s / u.cm**2 / u.AA
                            ):
                                data = data.to(
                                    output_flux_unit,
                                    equivalencies=u.spectral_density(
                                        wave[band]
                                    ),
                                )
                                rms_data = rms_data.to(
                                    output_flux_unit,
                                    equivalencies=u.spectral_density(
                                        wave[band]
                                    ),
                                )
                                unit = output_flux_unit
                            else:
                                raise Exception(
                                    "Output flux unit not recognised"
                                )

                            final_data = data
                            final_rms_data = rms_data
                            final_unit = unit
                    elif zeropoint is not None:
                        # print('Found zeropoint')
                        # import psutil

                        # print(proc.open_files())
                        ##proc = psutil.Process()

                        output_zeropoint = output_flux_unit.to(u.ABmag)
                        final_data = data * 10 ** (
                            (output_zeropoint - zeropoint) / 2.5
                        )
                        final_rms_data = rms_data * 10 ** (
                            (output_zeropoint - zeropoint) / 2.5
                        )
                        final_unit = output_flux_unit

                    else:
                        data = data * unit
                        rms_data = rms_data * unit

                        final_data = data
                        final_rms_data = rms_data
                        final_unit = unit

                    assert final_data is not None, "Final data is None"
                    assert final_rms_data is not None, "Final rms data is None"
                    assert final_unit is not None, "Final unit is None"

                    phot_imgs[band] = copy.deepcopy(final_data)
                    phot_pix_unit[band] = copy.deepcopy(final_unit)
                    rms_err_imgs[band] = copy.deepcopy(final_rms_data)
                    seg_imgs[band] = copy.deepcopy(hdu["SEG"].data)
                    phot_img_headers[band] = str(hdu["SCI"].header)

                    final_data = None
                    final_rms_data = None
                    final_unit = None

                    hdu.close()
                    # Close file
                    del (
                        hdu,
                        data,
                        rms_data,
                        final_data,
                        final_rms_data,
                        final_unit,
                    )

                # import psutil

                # proc = psutil.Process()
                # print(proc.open_files())

            if already_psf_matched:
                psf_matched_data = {psf_type: {}}
                psf_matched_rms_err = {psf_type: {}}
                for band in bands:
                    psf_matched_data[psf_type][band] = phot_imgs[band]
                    psf_matched_rms_err[psf_type][band] = rms_err_imgs[band]
            else:
                psf_matched_data = None
                psf_matched_rms_err = None

            if type(cutout_size_gal) is u.Quantity:
                cutout_size_gal = int(cutout_size_gal.value)

            object = cls(
                galaxy_id=gal_id,
                sky_coord=galaxy_skycoord,
                survey=survey,
                bands=bands,
                im_paths=im_paths,
                im_exts=im_exts,
                im_zps=im_zps,
                seg_paths=seg_paths,
                detection_band="+".join(forced_phot_band),
                galfind_version=version,
                rms_err_paths=err_paths,
                rms_err_exts=rms_err_exts,
                im_pixel_scales=im_pixel_scales,
                phot_imgs=phot_imgs,
                phot_pix_unit=phot_pix_unit,
                phot_img_headers=phot_img_headers,
                rms_err_imgs=rms_err_imgs,
                seg_imgs=seg_imgs,
                aperture_dict=aperture_dict,
                redshift=redshift,
                cutout_size=cutout_size_gal,
                dont_psf_match_bands=dont_psf_match_bands,
                auto_photometry=auto_photometry,
                psf_matched_data=psf_matched_data,
                psf_matched_rms_err=psf_matched_rms_err,
                meta_properties=meta_properties,
                already_psf_matched=already_psf_matched,
                overwrite=False,
            )
            objects.append(object)

        if load_multiple:
            return objects
        else:
            return objects[0]

    @classmethod
    def init_multiple_from_h5(
        cls,
        h5_names,
        h5_folder=resolved_galaxy_dir,
        save_out=True,
        n_jobs=1,
    ):
        if n_jobs == 1:
            galaxies = [
                cls.init_from_h5(
                    h5_name, h5_folder=h5_folder, save_out=save_out
                )
                for h5_name in tqdm(h5_names, desc="Loading galaxies")
            ]
        elif n_jobs >= 1:
            from joblib import Parallel, delayed

            galaxies = Parallel(n_jobs=n_jobs)(
                delayed(cls.init_from_h5)(
                    h5_name, h5_folder=h5_folder, save_out=save_out
                )
                for h5_name in tqdm(
                    h5_names, desc="Loading galaxies in parallel"
                )
            )

        return galaxies

    @classmethod
    def init_all_field_from_h5(
        cls,
        field,
        h5_folder=resolved_galaxy_dir,
        save_out=True,
        n_jobs=1,
    ):
        h5_names = glob.glob(f"{h5_folder}{field}*.h5")

        h5_names = [h5_name.split("/")[-1] for h5_name in h5_names]

        # sort them
        h5_names = sorted(h5_names)

        # Remove any with 'mock' in the name
        h5_names = [
            h5_name
            for h5_name in h5_names
            if "mock" not in h5_name and "temp" not in h5_name
        ]

        print("Found", len(h5_names), "galaxies in field", field)
        # print(h5_names)
        return cls.init_multiple_from_h5(
            h5_names, h5_folder=h5_folder, save_out=save_out, n_jobs=n_jobs
        )

    @classmethod
    def init_from_h5(
        cls,
        h5_name,
        h5_folder=resolved_galaxy_dir,
        return_attr=False,
        save_out=True,
    ):
        """Load a galaxy from an .h5 file"""
        if type(h5_name) is BytesIO:
            h5path = h5_name

        else:
            if not h5_name.endswith(".h5"):
                h5_name = f"{h5_name}.h5"
            if h5_name.startswith("/"):
                h5path = h5_name
            else:
                if not h5_folder.endswith("/"):
                    h5_folder += "/"

                h5path = f"{h5_folder}{h5_name}"

        # Check if file is locked by another process
        with h5.File(h5path, "r") as hfile:
            # Load meta data
            galaxy_id = hfile["meta"]["galaxy_id"][()].decode("utf-8")
            survey = hfile["meta"]["survey"][()].decode("utf-8")
            if hfile["meta"].get("sky_coord") is not None:
                sky_coord = hfile["meta"]["sky_coord"][()].split(b" ")
                sky_coord = SkyCoord(
                    ra=float(sky_coord[0]),
                    dec=float(sky_coord[1]),
                    unit=(u.deg, u.deg),
                )
            else:
                sky_coord = None
            if hfile["meta"].get("redshift") is not None:
                redshift = hfile["meta"]["redshift"][()]
            else:
                redshift = None

            bands = ast.literal_eval(
                hfile["meta"]["bands"][()].decode("utf-8")
            )
            cutout_size = int(hfile["meta"]["cutout_size"][()])
            im_zps = ast.literal_eval(hfile["meta"]["zps"][()].decode("utf-8"))
            if "galfind_version" in hfile["meta"].keys():
                galfind_version = hfile["meta"]["galfind_version"][()].decode(
                    "utf-8"
                )
            else:
                print("Warning! Assuming galfind version is v11")
                galfind_version = "v11"
            if "detection_band" in hfile["meta"].keys():
                detection_band = hfile["meta"]["detection_band"][()].decode(
                    "utf-8"
                )
            else:
                detection_band = "F277W+F356W+F444W"
                print("Warning! Assuming detection band is F277W+F356W+F444W")

            im_pixel_scales = ast.literal_eval(
                hfile["meta"]["pixel_scales"][()].decode("utf-8")
            )
            im_pixel_scales = {
                band: u.Quantity(scale)
                for band, scale in im_pixel_scales.items()
            }
            phot_pix_unit = ast.literal_eval(
                hfile["meta"]["phot_pix_unit"][()].decode("utf-8")
            )
            phot_pix_unit = {
                band: u.Unit(unit) for band, unit in phot_pix_unit.items()
            }
            if hfile["meta"].get("dont_psf_match_bands") is not None:
                dont_psf_match_bands = ast.literal_eval(
                    hfile["meta"]["dont_psf_match_bands"][()].decode("utf-8")
                )
            else:
                dont_psf_match_bands = []
            if hfile["meta"].get("already_psf_matched") is not None:
                already_psf_matched = ast.literal_eval(
                    hfile["meta"]["already_psf_matched"][()].decode("utf-8")
                )
            else:
                print("Warning! Assuming already PSF matched")
                already_psf_matched = True
            if hfile["meta"].get("meta_properties") is not None:
                meta_properties = {}
                for prop in hfile["meta"]["meta_properties"].keys():
                    meta_prop = hfile["meta"]["meta_properties"][prop][
                        ()
                    ].decode("utf-8")
                    try:
                        meta_prop = ast.literal_eval(
                            meta_prop.replace("unyt_quantity", "")
                        )
                    except:
                        meta_prop = meta_prop
                    meta_properties[prop] = meta_prop
            else:
                meta_properties = None
            if (
                "auto_photometry" in hfile.keys()
                and hfile["auto_photometry"].get("auto_photometry") is not None
            ):
                ap = (
                    hfile["auto_photometry"]["auto_photometry"][()]
                    .decode("utf-8")
                    .replace("<Quantity", "")
                    .replace(">", "")
                    .replace("array([", "([")
                    .replace(", dtype=float64", "")
                )
                auto_photometry = ast.literal_eval(str(ap))
            else:
                auto_photometry = None

            total_photometry = None

            if "total_photometry" in hfile.keys():
                if (
                    hfile["total_photometry"].get("total_photometry")
                    is not None
                ):
                    try:
                        total_photometry = ast.literal_eval(
                            hfile["total_photometry"]["total_photometry"][()]
                            .decode("utf-8")
                            .replace("<Quantity", "")
                            .replace(">", "")
                        )
                    except ValueError:
                        print(
                            f"Warning! Failed to load total photometry for {galaxy_id}"
                        )

            # Load paths and exts
            im_paths = ast.literal_eval(
                hfile["paths"]["im_paths"][()].decode("utf-8")
            )
            im_exts = ast.literal_eval(
                hfile["paths"]["im_exts"][()].decode("utf-8")
            )
            seg_paths = ast.literal_eval(
                hfile["paths"]["seg_paths"][()].decode("utf-8")
            )
            rms_err_paths = ast.literal_eval(
                hfile["paths"]["rms_err_paths"][()].decode("utf-8")
            )
            rms_err_exts = ast.literal_eval(
                hfile["paths"]["rms_err_exts"][()].decode("utf-8")
            )
            # Load aperture photometry
            # aperture_dict = ast.literal_eval(hfile['aperture_photometry']['aperture_dict'][()].decode('utf-8'))
            aperture_dict = {}
            if hfile.get("aperture_photometry") is not None:
                for aper in hfile["aperture_photometry"].keys():
                    aperture_dict[aper] = {}
                    for key in hfile["aperture_photometry"][aper].keys():
                        aperture_dict[aper][key] = hfile[
                            "aperture_photometry"
                        ][aper][key][()]
            # Load raw data
            phot_imgs = {}
            rms_err_imgs = {}
            seg_imgs = {}
            phot_img_headers = {}
            unmatched_data = {}
            unmatched_rms_err = {}
            unmatched_seg = {}

            for band in bands:
                phot_imgs[band] = hfile["raw_data"][f"phot_{band}"][()]
                rms_err_imgs[band] = hfile["raw_data"][f"rms_err_{band}"][()]
                if hfile["raw_data"].get(f"seg_{band}", None) is not None:
                    seg_imgs[band] = hfile["raw_data"][f"seg_{band}"][()]

                if hfile["headers"].get(band) is not None:
                    header = hfile["headers"][band][()].decode("utf-8")
                    phot_img_headers[band] = header
                if hfile.get("unmatched_data") is not None:
                    if len(hfile["unmatched_data"].keys()) > 0:
                        unmatched_data[band] = hfile["unmatched_data"][
                            f"phot_{band}"
                        ][()]
                        unmatched_rms_err[band] = hfile["unmatched_data"][
                            f"rms_err_{band}"
                        ][()]
                        if (
                            hfile["unmatched_data"].get(f"seg_{band}")
                            is not None
                        ):
                            unmatched_seg[band] = hfile["unmatched_data"][
                                f"seg_{band}"
                            ][()]

            if len(seg_imgs) == 0:
                seg_imgs = None

            if len(unmatched_data) == 0:
                unmatched_data = None
                unmatched_rms_err = None
                unmatched_seg = None

            if hfile.get("psf_matched_data") is not None:
                psf_matched_data = {}
                for psf_type in hfile["psf_matched_data"].keys():
                    psf_matched_data[psf_type] = {}
                    for band in bands:
                        psf_matched_data[psf_type][band] = hfile[
                            "psf_matched_data"
                        ][psf_type][band][()]
            else:
                psf_matched_data = None

            if hfile.get("psf_matched_rms_err") is not None:
                psf_matched_rms_err = {}
                for psf_type in hfile["psf_matched_rms_err"].keys():
                    psf_matched_rms_err[psf_type] = {}
                    for band in bands:
                        psf_matched_rms_err[psf_type][band] = hfile[
                            "psf_matched_rms_err"
                        ][psf_type][band][()]
            else:
                psf_matched_rms_err = None

            maps = {}
            if hfile.get("bin_maps") is not None:
                for key in hfile["bin_maps"].keys():
                    maps[key] = hfile["bin_maps"][key][()]

            binned_flux_map = None
            binned_flux_err_map = None
            if hfile.get("bin_fluxes") is not None:
                if hfile["bin_fluxes"].get("pixedfit") is not None:
                    binned_flux_map = (
                        hfile["bin_fluxes"]["pixedfit"][()]
                        * u.erg
                        / u.s
                        / u.cm**2
                        / u.AA
                    )

            if hfile.get("bin_flux_err") is not None:
                if hfile["bin_flux_err"].get("pixedfit") is not None:
                    binned_flux_err_map = (
                        hfile["bin_flux_err"]["pixedfit"][()]
                        * u.erg
                        / u.s
                        / u.cm**2
                        / u.AA
                    )

            possible_phot_keys = []
            photometry_table = {}
            if hfile.get("binned_photometry_table") is not None:
                for psf_type in hfile["binned_photometry_table"].keys():
                    photometry_table[psf_type] = {}
                    for binmap_type in hfile["binned_photometry_table"][
                        psf_type
                    ].keys():
                        if "__" not in binmap_type:
                            photometry_table[psf_type][binmap_type] = None
                            possible_phot_keys.append(
                                f"binned_photometry_table/{psf_type}/{binmap_type}"
                            )
            # else:
            #    print("No binned photometry table found")

            possible_sed_keys = []
            sed_fitting_table = {}
            if hfile.get("sed_fitting_table") is not None:
                for tool in hfile["sed_fitting_table"].keys():
                    sed_fitting_table[tool] = {}
                    for run in hfile["sed_fitting_table"][tool].keys():
                        if "__" not in run:
                            sed_fitting_table[tool][run] = None
                            possible_sed_keys.append(
                                f"sed_fitting_table/{tool}/{run}"
                            )

            rms_background = None
            if hfile.get("meta/rms_background") is not None:
                rms_background = ast.literal_eval(
                    hfile["meta/rms_background"][()].decode("utf-8")
                )

            # Get PSFs
            psfs = {}
            psfs_meta = {}
            if hfile.get("psfs") is not None:
                for psf_type in hfile["psfs"].keys():
                    psfs[psf_type] = {}
                    for band in bands:
                        if hfile["psfs"][psf_type].get(band) is not None:
                            psfs[psf_type][band] = hfile["psfs"][psf_type][
                                band
                            ][()]

            if hfile.get("psfs_meta") is not None:
                for psf_type in hfile["psfs_meta"].keys():
                    psfs_meta[psf_type] = {}
                    for band in bands:
                        if hfile["psfs_meta"][psf_type].get(band) is not None:
                            psfs_meta[psf_type][band] = hfile["psfs_meta"][
                                psf_type
                            ][band][()].decode("utf-8")

            galaxy_region = {}
            if hfile.get("galaxy_region") is not None:
                for binmap_type in hfile["galaxy_region"].keys():
                    galaxy_region[binmap_type] = hfile["galaxy_region"][
                        binmap_type
                    ][()]

            flux_map = {}
            if hfile.get("flux_map") is not None:
                for binmap_type in hfile["flux_map"].keys():
                    flux_map[binmap_type] = hfile["flux_map"][binmap_type][()]

            if hfile.get("det_data") is not None:
                det_data = {}
                det_data["phot"] = hfile["det_data"]["phot"][()]
                det_data["rms_err"] = hfile["det_data"]["rms_err"][()]
                det_data["seg"] = hfile["det_data"]["seg"][()]
            else:
                # print('no det data')
                det_data = None

            if hfile.get("resolved_mass") is not None:
                resolved_mass = {}
                for key in hfile["resolved_mass"].keys():
                    resolved_mass[key] = hfile["resolved_mass"][key][()]
            else:
                resolved_mass = None

            if hfile.get("resolved_sfr_100myr") is not None:
                resolved_sfr_100myr = {}
                for key in hfile["resolved_sfr_100myr"].keys():
                    resolved_sfr_100myr[key] = hfile["resolved_sfr_100myr"][
                        key
                    ][()]
            else:
                resolved_sfr_100myr = None

            if hfile.get("resolved_sfr_10myr") is not None:
                resolved_sfr_10myr = {}
                for key in hfile["resolved_sfr_10myr"].keys():
                    resolved_sfr_10myr[key] = hfile["resolved_sfr_10myr"][key][
                        ()
                    ]
            else:
                resolved_sfr_10myr = None

            # Read in resolved_sfh
            if hfile.get("resolved_sfh") is not None:
                resolved_sfh = {}
                for key in hfile["resolved_sfh"].keys():
                    resolved_sfh[key] = hfile["resolved_sfh"][key][()]

            else:
                resolved_sfh = None

            # Read in resolved SED
            if hfile.get("resolved_sed") is not None:
                resolved_sed = {}
                for key in hfile["resolved_sed"].keys():
                    resolved_sed[key] = hfile["resolved_sed"][key][()]
            else:
                resolved_sed = None

            # Read in PSF
            psf_kernels = {}
            if hfile.get("psf_kernels") is not None:
                for psf_type in hfile["psf_kernels"].keys():
                    psf_kernels[psf_type] = {}
                    for band in bands:
                        if (
                            hfile["psf_kernels"][psf_type].get(band)
                            is not None
                        ):
                            psf_kernels[psf_type][band] = hfile["psf_kernels"][
                                psf_type
                            ][band][()]

            interactive_outputs = None
            if hfile.get("interactive_outputs") is not None:
                interactive_outputs = {}
                for key in hfile["interactive_outputs"].keys():
                    interactive_outputs[key] = {}
                    # Get flux and flux_err
                    for prop in hfile["interactive_outputs"][key].keys():
                        interactive_outputs[key][prop] = hfile[
                            "interactive_outputs"
                        ][key][prop][()]
                        # Check for meta properties
                        meta = {}
                        for mkey in hfile["interactive_outputs"][key][
                            prop
                        ].attrs.keys():
                            meta[mkey] = hfile["interactive_outputs"][key][
                                prop
                            ].attrs[mkey]
                        if len(meta) > 0:
                            interactive_outputs[key]["meta"] = meta
                        else:
                            meta = None

            # hfile.close()
            # Read in photometry table(s)
            if len(possible_phot_keys) > 0:
                for key in possible_phot_keys:
                    table = read_table_hdf5(hfile, key)
                    psf_type, binmap_type = key.split("/")[1:]
                    photometry_table[psf_type][binmap_type] = table

            if len(possible_sed_keys) > 0:
                for key in possible_sed_keys:
                    table = read_table_hdf5(hfile, key)
                    tool, run = key.split("/")[1:]
                    sed_fitting_table[tool][run] = table

            photometry_properties = {}
            photometry_meta_properties = {}
            if hfile.get("photometry_properties") is not None:
                for prop in hfile["photometry_properties"].keys():
                    photometry_properties[prop] = hfile[
                        "photometry_properties"
                    ][prop][()]
                    meta = hfile["photometry_properties"][prop].attrs
                    # print('meta', meta)
                    photometry_meta_properties[prop] = {}
                    for key in meta:
                        # print(meta[key])
                        # print(key)
                        # print(meta[key])
                        photometry_meta_properties[prop][key] = meta[key]
                    if "unit" in photometry_meta_properties[prop].keys():
                        unit = u.Unit(photometry_meta_properties[prop]["unit"])
                    else:
                        unit = u.dimensionless_unscaled

                    # print('unit', unit)

                    photometry_properties[prop] = (
                        photometry_properties[prop] * unit
                    )
                    # print(f'Loaded {prop} with shape {photometry_properties[prop].shape}')

        out_dict = {
            "galaxy_id": galaxy_id,
            "sky_coord": sky_coord,
            "survey": survey,
            "bands": bands,
            "im_paths": im_paths,
            "im_zps": im_zps,
            "im_exts": im_exts,
            "seg_paths": seg_paths,
            "rms_err_paths": rms_err_paths,
            "rms_err_exts": rms_err_exts,
            "im_pixel_scales": im_pixel_scales,
            "phot_imgs": phot_imgs,
            "phot_pix_unit": phot_pix_unit,
            "phot_img_headers": phot_img_headers,
            "rms_err_imgs": rms_err_imgs,
            "seg_imgs": seg_imgs,
            "aperture_dict": aperture_dict,
            "psf_matched_data": psf_matched_data,
            "galfind_version": galfind_version,
            "psf_matched_rms_err": psf_matched_rms_err,
            "maps": maps,
            "binned_flux_map": binned_flux_map,
            "binned_flux_err_map": binned_flux_err_map,
            "photometry_table": photometry_table,
            "sed_fitting_table": sed_fitting_table,
            "rms_background": rms_background,
            "psfs": psfs,
            "psfs_meta": psfs_meta,
            "galaxy_region": galaxy_region,
            "cutout_size": cutout_size,
            "h5_folder": h5_folder,
            "psf_kernels": psf_kernels,
            "redshift": redshift,
            "flux_map": flux_map,
            "photometry_properties": photometry_properties,
            "photometry_meta_properties": photometry_meta_properties,
            "already_psf_matched": already_psf_matched,
            "dont_psf_match_bands": dont_psf_match_bands,
            "auto_photometry": auto_photometry,
            "unmatched_data": unmatched_data,
            "unmatched_rms_err": unmatched_rms_err,
            "unmatched_seg": unmatched_seg,
            "meta_properties": meta_properties,
            "det_data": det_data,
            "total_photometry": total_photometry,
            "resolved_mass": resolved_mass,
            "resolved_sfh": resolved_sfh,
            "resolved_sfr_100myr": resolved_sfr_100myr,
            "resolved_sfr_10myr": resolved_sfr_10myr,
            "resolved_sed": resolved_sed,
            "cutout_size": cutout_size,
            "h5_folder": h5_folder,
            "psf_kernels": psf_kernels,
            "interactive_outputs": interactive_outputs,
            "h5_path": h5path if type(h5_name) is not BytesIO else None,
            "save_out": save_out,
        }

        if return_attr:
            return out_dict

        else:
            return cls(**out_dict)
        """
        return cls(galaxy_id, sky_coord, survey, bands, im_paths, im_exts, im_zps,
                    seg_paths, rms_err_paths, rms_err_exts, im_pixel_scales, 
                    phot_imgs, phot_pix_unit, phot_img_headers, rms_err_imgs, seg_imgs, 
                    aperture_dict, psf_matched_data, psf_matched_rms_err, pixedfit_map, voronoi_map,
                    binned_flux_map, binned_flux_err_map, photometry_table, sed_fitting_table, rms_background,
                    psfs, psfs_meta, galaxy_region,
                    cutout_size, h5_folder)
        """

    @classmethod
    def init_from_basics(
        cls,
        galaxy_id: str,
        sky_coord: SkyCoord,
        survey: str,
        field_info: FieldInfo,
        cutout_size: typing.Union[int, u.Quantity, "auto"] = "auto",
        dont_psf_match_bands: typing.List[str] = [],
        redshift: float = -1,
        already_psf_matched: bool = False,
        forced_phot_band: typing.List[str] = ["F277W", "F356W", "F444W"],
    ):
        """
        This function will be the barebones function to initialize a Galaxy,
        without needing GALFIND.

        """

        # To Do

        # Make Cutouts from FieldInfo
        phot_imgs = {}
        phot_pix_unit = {}
        phot_img_headers = {}
        rms_err_imgs = {}
        seg_imgs = {}

        for band in field_info.band_names:
            im_path = field_info.im_paths[band]

            im_data = fits.getdata(im_path, field_info.im_exts[band])
            im_header = fits.getheader(im_path, field_info.im_exts[band])
            wcs = WCS(im_header)
            phot_img_headers[band] = str(im_header)

            cutout = Cutout2D(
                im_data,
                position=sky_coord,
                size=(cutout_size, cutout_size),
                wcs=wcs,
            )

            data = cutout.data

            # Check if data is all O or NaN
            if np.all(data == 0) or np.all(np.isnan(data)):
                raise Exception(f"All data is 0 or NaN for {band}")

            # Work out the unit or zero point (prefer zero point)
            if field_info.im_zps[band] is not None:
                zero_point = field_info.im_zps[band]
                # conversion_factor
                out_zeropoint = u.uJy.to(u.ABmag)
                scale_factor = 10 ** ((out_zeropoint - zero_point) / 2.5)
                final_data = data * scale_factor * u.uJy
                unit = u.uJy
            elif field_info.im_units[band] is not None:
                unit = u.Unit(field_info.im_units[band])
                if unit == u.Unit("MJy/sr"):
                    # Convert using pixel scale
                    scale_factor = field_info.im_pixel_scales[band] ** 2
                    data = data * unit * scale_factor

                    final_data = data.to(u.uJy)
                else:
                    scale_factor = 1
                    final_data = data * unit * scale_factor
                    final_data = final_data.to(u.uJy)

            phot_imgs[band] = copy.deepcopy(final_data)
            phot_pix_unit[band] = u.uJy

            del im_data

            # Check if we have rms, wht or seg files to cutout
            if field_info.err_paths[band] is not None:
                rms_err_path = field_info.err_paths[band]
                rms_err_data = fits.getdata(
                    rms_err_path, field_info.rms_err_exts[band]
                )
                rms_err_cutout = Cutout2D(
                    rms_err_data,
                    position=sky_coord,
                    size=(cutout_size, cutout_size),
                    wcs=wcs,
                )

                # Do the same conversion for the rms_err data
                rms_err_data = rms_err_cutout.data
                rms_err_data_final = rms_err_data * scale_factor * unit
                rms_err_imgs[band] = copy.deepcopy(
                    rms_err_data_final.to(u.uJy)
                )
                del rms_err_data

            if field_info.seg_paths[band] is not None:
                seg_path = field_info.seg_paths[band]
                seg_data = fits.getdata(seg_path)
                seg_cutout = Cutout2D(
                    seg_data,
                    position=sky_coord,
                    size=(cutout_size, cutout_size),
                    wcs=wcs,
                )

                seg_imgs[band] = copy.deepcopy(seg_cutout.data)
                del seg_data
        # Get the photometry from the cutouts

        if not already_psf_matched:
            psf_matched_data = None
            psf_matched_rms_err = None
        else:
            psf_matched_data = copy.deepcopy(phot_imgs)
            psf_matched_rms_err = copy.deepcopy(rms_err_imgs)

        meta_properties = {}

        return cls(
            galaxy_id=galaxy_id,
            sky_coord=sky_coord,
            survey=survey,
            bands=field_info.band_names,
            im_paths=field_info.im_paths,
            im_exts=field_info.im_exts,
            im_zps=field_info.im_zps,
            seg_paths=field_info.seg_paths,
            detection_band="+".join(forced_phot_band),
            galfind_version="",
            rms_err_paths=field_info.err_paths,
            rms_err_exts=field_info.rms_err_exts,
            im_pixel_scales=field_info.im_pixel_scales,
            phot_imgs=phot_imgs,
            phot_pix_unit=phot_pix_unit,
            phot_img_headers=phot_img_headers,
            rms_err_imgs=rms_err_imgs,
            seg_imgs=seg_imgs,
            aperture_dict={},
            redshift=redshift,
            cutout_size=cutout_size,
            dont_psf_match_bands=dont_psf_match_bands,
            auto_photometry={},
            psf_matched_data=psf_matched_data,
            psf_matched_rms_err=psf_matched_rms_err,
            meta_properties=meta_properties,
            already_psf_matched=already_psf_matched,
            overwrite=False,
        )

    def get_filter_wavs(
        self, facilities={"JWST": ["NIRCam"], "HST": ["ACS", "WFC3_IR"]}
    ):
        if getattr(self, "filter_wavs", None) is not None:
            return self.filter_wavs

        from astroquery.svo_fps import SvoFps

        filter_wavs = {}
        filter_instruments = {}
        filter_ranges = {}
        band_codes = {}

        done_band = False
        for facility in facilities:
            if done_band:
                break
            for instrument in facilities[facility]:
                try:
                    svo_table = SvoFps.get_filter_list(
                        facility=facility, instrument=instrument
                    )
                except:
                    continue
                bands_in_table = [
                    i.split(".")[-1] for i in svo_table["filterID"]
                ]
                for band in self.bands:
                    if band in bands_in_table:
                        if band in filter_wavs.keys():
                            raise Exception(
                                f"Band {band} found in multiple facilities"
                            )
                        else:
                            if instrument == "ACS":
                                instrument = "ACS_WFC"
                            filter_instruments[band] = instrument
                            mask = (
                                svo_table["filterID"]
                                == f"{facility}/{instrument}.{band}"
                            )
                            wav = svo_table[mask]["WavelengthCen"]
                            upper = (
                                svo_table[mask]["WavelengthCen"]
                                + svo_table[mask]["FWHM"] / 2.0
                            )
                            lower = (
                                svo_table[mask]["WavelengthCen"]
                                - svo_table[mask]["FWHM"] / 2.0
                            )
                            range = (lower * wav.unit, upper * wav.unit)

                            if len(wav) > 1:
                                raise Exception(
                                    f"Multiple profiles found for {band}"
                                )

                            filter_wavs[band] = wav[0] * wav.unit

                            band_codes[band] = svo_table[mask]["filterID"][0]

                            filter_ranges[band] = range

        assert len(filter_wavs.keys()) == len(
            self.bands
        ), f"Not all filters found {filter_wavs.keys()} vs {self.bands}"

        self.filter_wavs = filter_wavs
        self.filter_ranges = filter_ranges
        self.filter_instruments = filter_instruments
        self.band_codes = band_codes

    def add_psf_models(
        self,
        psf_dir,
        psf_type="star_stack",
        overwrite=False,
        file_ending=".fits",
    ):
        """
        This function will add PSF models to the galaxy object.
        """

        if getattr(self, "psfs", None) not in [None, {}] and not overwrite:
            if psf_type in self.psfs.keys():
                print(f"PSF models already exist for {psf_type}")
                return

        psf_dir = Path(psf_dir)
        if not psf_dir.exists():
            raise FileNotFoundError(f"PSF directory {psf_dir} not found")

        psfs = {}
        psfs_meta = {}

        for band in self.bands:
            paths = glob.glob(str(psf_dir / f"*{band}*{file_ending}"))
            if len(paths) == 0:
                print(f"No PSF found for {band} at {psf_dir}")
                continue
            elif len(paths) > 1:
                raise Exception(f"Multiple PSFs found for {band}: {paths}")

            psf_path = paths[0]

            psf_data = fits.getdata(psf_path)
            psf_header = fits.getheader(psf_path)

            psfs[band] = copy.deepcopy(psf_data)
            psfs_meta[band] = str(psf_header)

            self.add_to_h5(psf_data, f"psfs/{psf_type}/", band, overwrite=True)
            self.add_to_h5(
                str(psf_header), f"psfs_meta/{psf_type}/", band, overwrite=True
            )

        self.psfs = {psf_type: psfs}
        self.psfs_meta = {psf_type: psfs_meta}

    def run_autogalaxy(
        self,
        model_type="sersic",
        psf_type="star_stack",
        use_psf_matched_data=True,
        output_dir=None,
        overwrite=False,
        n_live_points=50,
        band=None,
        mask_type="circular",
        mask_radius=None,
        save_plots=True,
        make_diagnostic_plots=True,
        search_type="nest",
        return_results=False,
        override_redshift=None,
        path_prefix="modelling",
        model_priors={
            "centre_1": {"type": "Gaussian", "mean": 0.0, "sigma": 0.06},  # x
            "centre_0": {"type": "Gaussian", "mean": 0.0, "sigma": 0.06},  # y
            "effective_radius": {
                "type": "Uniform",
                "lower_limit": 1e-2,
                "upper_limit": 0.2,
            },
            "sersic_index": {"type": "Uniform", "lower": 0.1, "upper": 8.0},
        },
    ):
        """Run PyAutoGalaxy to perform Bayesian model fitting on galaxy images.

        All prior positions are in arcsec!

        Parameters
        ----------
        model_type : str
            Type of model to fit. Options are:
            - 'sersic': Single Sérsic profile
            - 'dev': de Vaucouleurs profile (n=4)
            - 'exp': Exponential profile (n=1)

        psf_type : str
            Type of PSF to use for convolution
        output_dir : str
            Directory to save output files. If None, uses resolved_galaxy_dir/autogalaxy/
        overwrite : bool
            Whether to overwrite existing results
        n_live_points : int
            Number of live points for nested sampling
        band : str
            List of bands to fit. If None, fits all bands
        mask_type : str
            Type of mask to apply:
            - Name of mask in galaxy.region (e.g. 'pixedfit', 'detection')
            - 'circular': Circular mask with given radius
            - 'elliptical': Elliptical mask based on Kron parameters
        mask_radius : float
            Radius of mask in arcsec. If None, uses Kron radius
        save_plots : bool
            Whether to save diagnostic plots
        search_type : str
            Type of parameter space search ('nest' or 'mcmc')
        return_results : bool
            Whether to return the full results object

        Returns
        -------
        dict
            Dictionary containing fitted parameters and uncertainties
            if return_results=True
        """
        import autogalaxy as ag
        import autogalaxy.plot as aplt
        import autofit as af
        import numpy as np
        import os

        if override_redshift is not None:
            z = override_redshift
        else:
            z = self.redshift

        if output_dir is None:
            output_dir = os.path.join(resolved_galaxy_dir, "autogalaxy")
        os.makedirs(output_dir, exist_ok=True)

        # move to output directory
        os.chdir(output_dir)

        if hasattr(self, "autogalaxy_results") and not overwrite:
            print(
                "AutoGalaxy results already exist. Set overwrite=True to rerun."
            )
            return

        # Set up bands to fit
        if band is None:
            print(f"Fitting {self.bands[-1]}")
            band = self.bands[-1]

            # Create temporary directory to store fits files
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create imaging data objects for each band
            # Create imaging object using from_fits
            pixel_scales = self.im_pixel_scales[band].to("arcsec")

            # Write image data, PSF and RMS map to temporary fits files
            data_path = os.path.join(tmpdir, f"data_{band}.fits")
            psf_path = os.path.join(tmpdir, f"psf_{band}.fits")
            noise_path = os.path.join(tmpdir, f"noise_{band}.fits")
            if use_psf_matched_data:
                im_data = self.psf_matched_data[psf_type][band]
                noise_data = self.psf_matched_rms_err[psf_type][band]
                psf_data = self.psfs[psf_type][self.bands[-1]]
            else:
                im_data = self.phot_imgs[band]
                noise_data = self.rms_err_imgs[band]
                self.psfs[psf_type][band]

            im_data *= self.phot_pix_unit[band]
            noise_data *= self.phot_pix_unit[band]

            # Convert from uJy to MJy/sr manually
            pix_sr = pixel_scales**2
            im_data = im_data.to(u.MJy) / (pix_sr.to(u.steradian))
            noise_data = noise_data.to(u.MJy) / (pix_sr.to(u.steradian))

            # Convert to float32 to avoid fits precision issues
            fits.writeto(data_path, im_data.value, overwrite=True)
            fits.writeto(noise_path, noise_data.value, overwrite=True)
            fits.writeto(psf_path, psf_data, overwrite=True)

            pixel_scales = float(pixel_scales.value)

            imaging = ag.Imaging.from_fits(
                pixel_scales=pixel_scales,
                data_path=data_path,
                psf_path=psf_path,
                noise_map_path=noise_path,
                data_hdu=0,
                psf_hdu=0,
                noise_map_hdu=0,
            )

            # Create mask
            if mask_type in self.gal_region:
                # Get pixel coordinates where region is True
                region_mask = self.gal_region[mask_type]
                y_coords, x_coords = np.where(region_mask > 0)
                pixel_coordinates = list(zip(y_coords, x_coords))

                mask = ag.Mask2D.from_pixel_coordinates(
                    shape_native=region_mask.shape,
                    pixel_coordinates=pixel_coordinates,
                    pixel_scales=pixel_scales,
                    invert=True,  # True = unmasked for pixel coordinates method
                    buffer=1,  # Small buffer to ensure edge pixels are included
                )

            elif mask_type == "circular":
                if mask_radius is None:
                    # Use Kron radius
                    if band in self.auto_photometry:
                        kron_radius = self.auto_photometry[band]["KRON_RADIUS"]
                        mask_radius = kron_radius * pixel_scales
                    else:
                        mask_radius = 3.0  # arcsec default
                print("Using circular mask.")
                print(f"Mask radius: {mask_radius} arcsec")
                print(im_data.shape, type(im_data.shape))

                mask = ag.Mask2D.circular(
                    shape_native=im_data.shape,
                    pixel_scales=pixel_scales,
                    radius=mask_radius,
                )
            elif mask_type == "elliptical":
                # Use Kron ellipse parameters if available
                if band in self.auto_photometry:
                    a = self.auto_photometry[band]["A_IMAGE"]
                    b = self.auto_photometry[band]["B_IMAGE"]
                    theta = self.auto_photometry[band]["THETA_IMAGE"]
                    kron_radius = self.auto_photometry[band]["KRON_RADIUS"]
                    mask = ag.Mask2D.elliptical(
                        shape_native=im_data.shape,
                        pixel_scales=pixel_scales,
                        major_axis_radius=a * kron_radius * pixel_scales,
                        axis_ratio=b / a,
                        angle=theta,
                    )
                else:
                    # Fall back to circular mask
                    mask = ag.Mask2D.circular(
                        shape_native=im_data.shape,
                        pixel_scales=pixel_scales,
                        radius=3.0,
                    )
            elif mask_type is None:
                mask = None
            else:
                raise ValueError(f"Mask type {mask_type} not recognized.")
            # Apply mask
            if mask is not None:
                imaging = imaging.apply_mask(mask=mask)

        if make_diagnostic_plots:
            dataset_plotter = aplt.ImagingPlotter(dataset=imaging)
            dataset_plotter.figures_2d(data=True)
        # Set up model
        total_model = af.Collection()

        str_to_model = {
            "sersic": ag.lp.Sersic,
            "exp": ag.lp.Exponential,
            "dev": ag.lp.DevVaucouleurs,
        }

        if model_type not in str_to_model.keys():
            raise ValueError(
                f"Model type {model_type} not recognized. Only {str_to_model.keys()} are currently supported."
            )

        model = af.Model(str_to_model[model_type])
        for prior in model_priors:
            prior_type = model_priors[prior]["type"]
            prior_obj = getattr(af, f"{prior_type}Prior")
            copy_dict = model_priors[prior].copy()
            # Remove type key
            del copy_dict["type"]
            setattr(model, prior, prior_obj(**copy_dict))

        model.galaxies = ag.Galaxy(
            redshift=z,
            light=model,
        )

        name = f"{self.galaxy_id}"
        unique_tag = f"{model_type}_{psf_type}"

        # Set up analysis pipeline
        if search_type == "nest":
            search = af.DynestyDynamic(
                path_prefix=path_prefix,
                name=name,
                unique_tag=unique_tag,
                iterations_per_update=iterations_per_update,
                **sampler_kwargs,
            )
        elif search_type == "mcmc":
            search = af.Emcee(
                path_prefix=path_prefix,
                name=name,
                unique_tag=unique_tag,
                iterations_per_update=iterations_per_updatem**sampler_kwargs,
            )
        elif search_type == "nautilus":
            search = af.Nautilus(
                path_prefix=path_prefix,
                name=name,
                unique_tag=unique_tag,
                iterations_per_update=iterations_per_update,
                **sampler_kwargs,
            )
        # Create analysis class for each band
        analysis = ag.AnalysisImaging(dataset=imaging)

        # Run fitting
        result = search.fit(model=total_model, analysis=analysis)

        # Save results
        self.autogalaxy_results = {
            "model_type": model_type,
            "mp_instance": result.max_log_likelihood_instance,
            "parameters": result.samples.parameter_lists,
            "log_evidence": result.log_evidence,
            "total_log_likelihood": result.total_log_likelihood,
        }

        # Generate model images and fit info

        fit = ag.FitImaging(
            dataset=imaging, light_profile=result.max_log_likelihood_instance
        )

        self.autogalaxy_results[f"{band}_model"] = fit.model_data
        self.autogalaxy_results[f"{band}_residuals"] = fit.residual_map
        self.autogalaxy_results[f"{band}_chi2"] = fit.chi_squared_map

        if save_plots:
            # Save diagnostic plots

            fit = ag.FitImaging(
                dataset=imaging,
                light_profile=result.max_log_likelihood_instance,
            )
            fit.figures.subplot_fit()
            plt.savefig(
                f"{output_dir}/{self.galaxy_id}_{band}_fit.png",
                bbox_inches="tight",
            )
            plt.close()

        # Save to h5 file
        self.add_to_h5(
            self.autogalaxy_results,
            "autogalaxy",
            "results",
            overwrite=overwrite,
        )

        if return_results:
            return self.autogalaxy_results

    def run_galfitm(
        self,
        model_type="sersic",
        psf_type="star_stack",
        output_dir=None,
        overwrite=False,
        binmap_type="pixedfit",
        save_models=True,
        return_models=False,
    ):
        """Run GALFITM on the galaxy images to perform multi-band morphological fitting.

        Parameters
        ----------
        model_type : str
            Type of model to fit. Options are:
            - 'sersic': Single Sérsic profile
            - 'double_sersic': Double Sérsic profile
            - 'psf': Point source
            - 'sersic_psf': Sérsic + point source
        psf_type : str
            Type of PSF to use for convolution
        output_dir : str
            Directory to save output files. If None, uses resolved_galaxy_dir/galfitm/
        overwrite : bool
            Whether to overwrite existing files
        binmap_type : str
            Type of binning map to use for determining galaxy region
        save_models : bool
            Whether to save the model images and residuals
        return_models : bool
            Whether to return the model images and GalfitM outputs

        Returns
        -------
        dict
            Dictionary containing the fitted model parameters and uncertainties
            if return_models=True
        """
        from pygalfitm import PyGalfitm
        import tempfile
        import os

        if output_dir is None:
            output_dir = os.path.join(resolved_galaxy_dir, "galfitm")
        os.makedirs(output_dir, exist_ok=True)

        if hasattr(self, "galfitm_results") and not overwrite:
            print(
                "GalfitM results already exist. Set overwrite=True to rerun."
            )
            return

        # Get effective wavelengths for each band
        self.get_filter_wavs()
        wavs = [self.filter_wavs[band].value for band in self.bands]

        # Initialize PyGalfitm object
        gal = PyGalfitm()

        # Set up base parameters
        gal.set_base(
            {
                "A": f"{output_dir}/input",
                "A1": ",".join(self.bands),
                "A2": ",".join([str(w) for w in wavs]),
                "B": ",".join(self.bands),
                "K": f"{self.im_pixel_scales[self.bands[0]].to('arcsec').value}, {self.im_pixel_scales[self.bands[0]].to('arcsec').value}",
                "J": ",".join([str(self.im_zps[band]) for band in self.bands]),
            }
        )

        # Set up model components based on model_type
        if model_type == "sersic":
            gal.activate_components(["sersic"])
            gal.set_component(
                "sersic",
                {
                    "1": ("1", 3, "cheb"),  # Position x
                    "2": ("1", 3, "cheb"),  # Position y
                    "3": ("1", 3, "cheb"),  # Magnitude
                    "4": ("1", 1, "cheb"),  # Re
                    "5": ("1", 1, "cheb"),  # n
                    "6": ("1", 1, "cheb"),  # q
                    "7": ("1", 1, "cheb"),  # PA
                },
            )
        elif model_type == "double_sersic":
            gal.activate_components(["sersic", "sersic"])
            # Set up double Sersic parameters
            for i in range(2):
                gal.set_component(
                    f"sersic{i+1}",
                    {
                        "1": ("1", 3, "cheb"),
                        "2": ("1", 3, "cheb"),
                        "3": ("1", 3, "cheb"),
                        "4": ("1", 1, "cheb"),
                        "5": ("1", 1, "cheb"),
                        "6": ("1", 1, "cheb"),
                        "7": ("1", 1, "cheb"),
                    },
                )
        elif model_type == "psf":
            gal.activate_components(["psf"])
            gal.set_component(
                "psf",
                {
                    "1": ("1", 3, "cheb"),
                    "2": ("1", 3, "cheb"),
                    "3": ("1", 3, "cheb"),
                },
            )
        elif model_type == "sersic_psf":
            gal.activate_components(["sersic", "psf"])
            gal.set_component(
                "sersic",
                {
                    "1": ("1", 3, "cheb"),
                    "2": ("1", 3, "cheb"),
                    "3": ("1", 3, "cheb"),
                    "4": ("1", 1, "cheb"),
                    "5": ("1", 1, "cheb"),
                    "6": ("1", 1, "cheb"),
                    "7": ("1", 1, "cheb"),
                },
            )
            gal.set_component(
                "psf",
                {
                    "1": ("1", 3, "cheb"),
                    "2": ("1", 3, "cheb"),
                    "3": ("1", 3, "cheb"),
                },
            )

        # Write input files
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write cutouts
            for band in self.bands:
                fits.writeto(
                    f"{tmpdir}/cutout_{band}.fits",
                    self.phot_imgs[band],
                    overwrite=True,
                )
                fits.writeto(
                    f"{tmpdir}/rms_{band}.fits",
                    self.rms_err_imgs[band],
                    overwrite=True,
                )
                fits.writeto(
                    f"{tmpdir}/psf_{band}.fits",
                    self.psfs[psf_type][band],
                    overwrite=True,
                )

            # Write feedme file
            gal.write_feedme(feedme_path=f"{tmpdir}/galfit.feedme")

            # Run GalfitM
            log = gal.run()

            # Read results
            from pygalfitm.read import read_output_to_class

            results = read_output_to_class(f"{tmpdir}/output.fits")

            if save_models:
                # Save model images
                for band in self.bands:
                    model = results.get_model(band)
                    residual = results.get_residual(band)
                    fits.writeto(
                        f"{output_dir}/{self.galaxy_id}_{band}_model.fits",
                        model,
                        overwrite=True,
                    )
                    fits.writeto(
                        f"{output_dir}/{self.galaxy_id}_{band}_residual.fits",
                        residual,
                        overwrite=True,
                    )

            # Save parameters to class
            self.galfitm_results = {
                "model_type": model_type,
                "parameters": results.get_parameters(),
                "uncertainties": results.get_uncertainties(),
                "chi2": results.get_chi2(),
                "reduced_chi2": results.get_reduced_chi2(),
            }

            # Save to h5 file
            self.add_to_h5(
                self.galfitm_results, "galfitm", "results", overwrite=overwrite
            )

            if return_models:
                return self.galfitm_results

    def _convolve_sed(self, flux, wav, filters="self"):
        """Convolve an SED with a set of filters"""

        self.get_filter_wavs()

        if filters == "self":
            filters = self.bands

        from astroquery.svo_fps import SvoFps

        filters = [self.band_codes[band] for band in filters]

        mask = np.isfinite(flux) & np.isfinite(wav)
        mags = flux[mask]
        wav = wav[mask]

        output_unit = flux.unit

        # if len(wav) < len_wav:
        # print('Removed nans from input SED')
        if len(flux) != len(wav) != len(filters):
            print("Inputs are different lengths!")
        if type(wav[0]) == float:
            print("Assuming microns")
            wav = [i * u.um for i in wav]

        output_phot = []
        for filt in filters:
            data = SvoFps.get_transmission_data(filt)
            filt_wav = data["Wavelength"]
            trans = data["Transmission"]

            filt_wav = filt_wav.to("micron")

            trap_object = lambda x: np.interp(x, filt_wav, trans)
            max_wav = np.max(filt_wav)
            min_wav = np.min(filt_wav)

            # print('Filter min_wav, max_wav', filt, min_wav, max_wav)

            mask = (wav < max_wav) & (wav > min_wav)
            wav_loop = wav[mask]
            flux_loop = flux[mask]

            if len(wav_loop) == 0:
                print(
                    f"Warning! No overlap between filter {filt} and SED. Filling with 99"
                )
                if output_unit == u.ABmag:
                    out_flux = 99 * u.ABmag
                else:
                    out_flux = 0 * output_unit
            else:
                fluxes = flux_loop.to(u.Jy)
                trans_ob = trap_object(wav_loop)
                trans_int = np.trapz(trans, filt_wav)
                # print(trans_ob*fluxes, wav_loop)
                top_int = np.trapz(trans_ob * fluxes, wav_loop)
                # print(top_int)
                frac = top_int / trans_int

                out_flux = frac.to(output_unit).value

            output_phot.append(out_flux)

        output_phot = np.array(output_phot) * output_unit
        return output_phot

    def property_in_mask(
        self,
        property_map,
        mask=None,
        mask_name=None,
        func="sum",
        density=False,
        return_total=False,
    ):
        """Given a map and mask of the same dimensions, calculate the sum, mean or median of the map within each segment of the mask
        For the mask  0 is background, positive integers are different segments"""
        assert (
            mask is not None or mask_name is not None
        ), "Need to provide either a mask or a mask_name"

        if mask is None:
            if mask_name not in self.gal_region.keys():
                raise ValueError(
                    f"Mask {mask_name} not found in galaxy region."
                )

            mask = self.gal_region[mask_name]

        assert (
            property_map.shape == mask.shape
        ), f"Property map and mask must have the same shape, {property_map.shape} vs {mask.shape}"

        if func == "sum":
            func = np.nansum
        elif func == "mean":
            func = np.nanmean
        elif func == "median":
            func = np.nanmedian

        unique = np.unique(mask)

        unique = unique[unique != 0]

        output = np.zeros_like(map)

        # Make output nans initially
        output = np.nan * np.zeros_like(property_map)

        for un in unique:
            output[mask == un] = func(property_map[mask == un])

        if density:
            pix_size = self.im_pixel_scales[self.bands[0]].to(u.arcsec).value
            print("Assuming pixel size is", pix_size, "arcsec")
            # Divide each region by pixel area
            for un in unique:
                num_pix = np.sum(mask == un)
                area = num_pix * pix_size**2
                output[mask == un] /= area

        if return_total:
            return func(np.unique(output))

        return output

    def get_star_stack_psf(
        self, match_band=None, scaled_psf=False, just_return_psfs=False
    ):
        """Get the PSF kernel from a star stack"""

        # Check for kernel

        psf_kernel_folder = (
            f"{self.psf_kernel_folder}/star_stack/{self.survey}/"
        )
        psf_folder = f"{self.psf_folder}/star_stack/{self.survey}/"

        if self.psfs is None:
            self.psfs = {}
        if self.psfs_meta is None:
            self.psfs_meta = {}

        scale = "" if not scaled_psf else "_norm"
        if "star_stack" not in self.psfs.keys():
            self.psfs["star_stack"] = {}
            self.psfs_meta["star_stack"] = {}

        run = False
        psfs = {}
        for band in self.bands:
            if band in self.dont_psf_match_bands or self.already_psf_matched:
                continue
            path = f"{psf_folder}/{band}_psf{scale}.fits"
            if not os.path.exists(path):
                raise Exception(f"No PSF found for {band} in {psf_folder}")
            else:
                psf = fits.getdata(path)
                psf_hdr = str(fits.getheader(path))
                self.psfs["star_stack"][band] = psf
                self.psfs_meta["star_stack"][band] = psf_hdr

            if just_return_psfs:
                psfs[band] = psf
            else:
                self.add_to_h5(psf, "psfs/star_stack/", band, overwrite=True)
                self.add_to_h5(
                    psf_hdr, "psfs_meta/star_stack/", band, overwrite=True
                )
                if match_band is None:
                    match_band = self.bands[-1]
                kernel_path = (
                    f"{psf_kernel_folder}/kernel_{band}_to_{match_band}.fits"
                )

                if os.path.exists(kernel_path):
                    kernel = fits.getdata(kernel_path)
                    if self.psf_kernels is None:
                        self.psf_kernels = {"star_stack": {}}
                    elif self.psf_kernels.get("star_stack") is None:
                        self.psf_kernels["star_stack"] = {}

                    self.psf_kernels["star_stack"][band] = kernel
                    self.add_to_h5(
                        kernel, "psf_kernels/star_stack/", band, overwrite=True
                    )
                elif band == match_band:
                    pass
                else:
                    print(f"{kernel_path} not found, generating kernel.")
                    run = True

        if just_return_psfs:
            return psfs

        if run:
            self.convert_psfs_to_kernels(psf_type="star_stack")

    def estimate_rms_from_background(
        self, cutout_size=250, object_distance=20, overwrite=True, plot=False
    ):
        """Estimate the RMS error from the background"""
        import cv2

        if self.rms_background is None or overwrite:
            self.rms_background = {}
            if plot:
                update_mpl(tex_on=False)
                max_in_row = 4
                fig, axs = plt.subplots(
                    nrows=len(self.bands) // max_in_row + 1,
                    ncols=max_in_row,
                    figsize=(20, 20),
                )
                axs = axs.flatten()
                # Delete extra axes
                for i in range(len(self.bands), len(axs)):
                    fig.delaxes(axs[i])

            for pos, band in enumerate(self.bands):
                image_path = self.im_paths[band]
                hdu = fits.open(image_path)
                ra, dec = self.sky_coord.ra.deg, self.sky_coord.dec.deg
                wcs = WCS(hdu[self.im_exts[band]].header)
                x_cent, y_cent = wcs.all_world2pix(ra, dec, 0)
                data = hdu[self.im_exts[band]].section[
                    int(y_cent - cutout_size / 2) : int(
                        y_cent + cutout_size / 2
                    ),
                    int(x_cent - cutout_size / 2) : int(
                        x_cent + cutout_size / 2
                    ),
                ]
                # Open seg
                seg_path = self.seg_paths[band]
                seg_hdu = fits.open(seg_path)
                seg_data = seg_hdu[0].data[
                    int(y_cent - cutout_size / 2) : int(
                        y_cent + cutout_size / 2
                    ),
                    int(x_cent - cutout_size / 2) : int(
                        x_cent + cutout_size / 2
                    ),
                ]

                # Dilate the segmentation map to be more than 20 pixels from
                seg_data[seg_data != 0] = 1
                kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (object_distance, object_distance)
                )
                seg_data = seg_data.astype(np.uint8)
                seg_mask = cv2.dilate(seg_data, kernel, iterations=1)
                seg_mask = seg_mask.astype(bool)
                # Get RMS of background
                rms = np.sqrt(np.nanmean(data[~seg_mask] ** 2))
                if plot:
                    # Plot histogram of background
                    # Don't use latex
                    ax = axs[pos]
                    ax.hist(
                        data[~seg_mask].flatten(),
                        bins=30,
                        histtype="step",
                        color="k",
                    )
                    ax.axvline(rms, color="r", linestyle="--")
                    # ax.set_xlabel('Background')
                    ax.set_title(f"{band} bckg RMS = {rms:.4f} MJy/sr")

                    err_data = hdu[self.rms_err_exts[band]].section[
                        int(y_cent - cutout_size / 2) : int(
                            y_cent + cutout_size / 2
                        ),
                        int(x_cent - cutout_size / 2) : int(
                            x_cent + cutout_size / 2
                        ),
                    ]
                    ax.hist(
                        err_data[~seg_mask].flatten(),
                        bins=30,
                        histtype="step",
                        color="b",
                    )

                self.rms_background[band] = rms

            # In MJy/sr - need to convert?

            self.add_to_h5(
                str(self.rms_background),
                "meta",
                "rms_background",
                overwrite=True,
            )

        # if plot:
        #    return fig

    def get_number_of_bins(self, binmap_type="pixedfit"):
        region = getattr(self, f"{binmap_type}_map", None)
        if region is not None:
            return len(np.unique(region)) - 1
        else:
            print("No galaxy region found")
            return None

    def dump_to_h5(
        self, h5_folder=resolved_galaxy_dir, mode="append", force=False
    ):
        """Dump the galaxy data to an .h5 file"""
        # for strings

        if not self.save_out and not force:
            print("Skipping writing to .h5")
            return

        if not os.path.exists(h5_folder):
            print("Making directory", h5_folder)
            os.makedirs(h5_folder)

        str_dt = h5.string_dtype(encoding="utf-8")
        # Convert most dictionaries to strings
        # 'meta' - galaxy ID, survey, sky_coord,    version, instruments, excl_bands, cutout_size, zps, pixel_scales, phot_pix_unit
        # 'paths' - im_paths, seg_paths, rms_err_paths
        # 'raw_data' - phot_imgs, rms_err_imgs, seg_imgs
        # 'headers' - phot_img_headers
        # 'bin_maps' - voronoi_map, pixedfit_map
        # 'psf_matched_data'
        # 'aperture_photometry'
        # 'auto_photometry'
        # 'sed_fitting'[tool] =

        if os.path.exists(self.h5_path):
            append = "_temp"
        else:
            append = ""
        new_h5_path = self.h5_path.replace(".h5", f"{append}.h5")

        if not os.path.exists(os.path.dirname(new_h5_path)):
            os.makedirs(os.path.dirname(new_h5_path))

        try:
            with h5.File(new_h5_path, "w") as hfile:
                # print('append is', append)

                groups = [
                    "meta",
                    "paths",
                    "raw_data",
                    "aperture_photometry",
                    "auto_photometry",
                    "headers",
                    "bin_maps",
                    "bin_fluxes",
                    "bin_flux_err",
                ]
                for group in groups:
                    hfile.create_group(group)

                hfile["meta"].create_dataset(
                    "galaxy_id", data=str(self.galaxy_id), dtype=str_dt
                )
                hfile["meta"].create_dataset(
                    "survey", data=self.survey, dtype=str_dt
                )
                hfile["meta"].create_dataset("redshift", data=self.redshift)
                if self.sky_coord is not None:
                    hfile["meta"].create_dataset(
                        "sky_coord",
                        data=self.sky_coord.to_string(
                            style="decimal", precision=8
                        ),
                        dtype=str_dt,
                    )
                hfile["meta"].create_dataset(
                    "bands", data=str(list(self.bands)), dtype=str_dt
                )
                hfile["meta"].create_dataset(
                    "cutout_size", data=self.cutout_size
                )
                hfile["meta"].create_dataset(
                    "zps", data=str(self.im_zps), dtype=str_dt
                )
                hfile["meta"].create_dataset(
                    "pixel_scales",
                    data=str(
                        {
                            band: str(scale)
                            for band, scale in self.im_pixel_scales.items()
                        }
                    ),
                    dtype=str_dt,
                )
                hfile["meta"].create_dataset(
                    "phot_pix_unit",
                    data=str(
                        {
                            band: str(pix_unit)
                            for band, pix_unit in self.phot_pix_unit.items()
                        }
                    ),
                    dtype=str_dt,
                )
                hfile["meta"].create_dataset(
                    "dont_psf_match_bands",
                    data=str(self.dont_psf_match_bands),
                    dtype=str_dt,
                )
                hfile["meta"].create_dataset(
                    "already_psf_matched",
                    data=str(self.already_psf_matched),
                    dtype=str_dt,
                )
                hfile["meta"].create_dataset(
                    "detection_band", data=self.detection_band, dtype=str_dt
                )
                hfile["meta"].create_dataset(
                    "galfind_version", data=self.galfind_version, dtype=str_dt
                )
                if self.meta_properties is not None:
                    hfile["meta"].create_group("meta_properties")
                    for prop in self.meta_properties.keys():
                        hfile["meta"]["meta_properties"].create_dataset(
                            prop,
                            data=str(self.meta_properties[prop]),
                            dtype=str_dt,
                        )

                if self.auto_photometry is not None:
                    hfile["auto_photometry"].create_dataset(
                        "auto_photometry",
                        data=str(self.auto_photometry),
                        dtype=str_dt,
                    )
                if self.total_photometry is not None:
                    hfile.create_group("total_photometry")
                    hfile["total_photometry"].create_dataset(
                        "total_photometry",
                        data=str(self.total_photometry),
                        dtype=str_dt,
                    )
                # Save paths and exts
                keys_to_check = [
                    "im_paths",
                    "seg_paths",
                    "rms_err_paths",
                    "im_exts",
                    "rms_err_exts",
                ]

                hfile["paths"].create_dataset(
                    "im_paths", data=str(self.im_paths), dtype=str_dt
                )
                hfile["paths"].create_dataset(
                    "seg_paths", data=str(self.seg_paths), dtype=str_dt
                )
                hfile["paths"].create_dataset(
                    "rms_err_paths", data=str(self.rms_err_paths), dtype=str_dt
                )
                hfile["paths"].create_dataset(
                    "im_exts", data=str(self.im_exts), dtype=str_dt
                )
                hfile["paths"].create_dataset(
                    "rms_err_exts", data=str(self.rms_err_exts), dtype=str_dt
                )

                if self.aperture_dict is not None:
                    for aper in self.aperture_dict.keys():
                        hfile["aperture_photometry"].create_group(aper)
                        for key in self.aperture_dict[aper].keys():
                            data = self.aperture_dict[aper][key]
                            hfile["aperture_photometry"][aper].create_dataset(
                                f"{key}", data=data
                            )

                # Save raw data
                for band in self.bands:
                    hfile["raw_data"].create_dataset(
                        f"phot_{band}",
                        data=self.phot_imgs[band],
                        compression="gzip",
                    )
                    hfile["raw_data"].create_dataset(
                        f"rms_err_{band}",
                        data=self.rms_err_imgs[band],
                        compression="gzip",
                    )

                    if self.seg_imgs is not None:
                        hfile["raw_data"].create_dataset(
                            f"seg_{band}",
                            data=self.seg_imgs[band],
                            compression="gzip",
                        )

                if self.unmatched_data is not None:
                    hfile.create_group("unmatched_data")
                    for band in self.bands:
                        hfile["unmatched_data"].create_dataset(
                            f"phot_{band}",
                            data=self.unmatched_data[band],
                            compression="gzip",
                        )
                        hfile["unmatched_data"].create_dataset(
                            f"rms_err_{band}",
                            data=self.unmatched_rms_err[band],
                            compression="gzip",
                        )
                        if self.unmatched_seg is not None:
                            hfile["unmatched_data"].create_dataset(
                                f"seg_{band}",
                                data=self.unmatched_seg[band],
                                compression="gzip",
                            )

                if self.det_data is not None:
                    hfile.create_group("det_data")
                    hfile["det_data"].create_dataset(
                        "phot",
                        data=self.det_data["phot"],
                        compression="gzip",
                    )
                    hfile["det_data"].create_dataset(
                        "rms_err",
                        data=self.det_data["rms_err"],
                        compression="gzip",
                    )
                    hfile["det_data"].create_dataset(
                        "seg",
                        data=self.det_data["seg"],
                        compression="gzip",
                    )

                # Save headers
                for band in self.bands:
                    if self.phot_img_headers.get(band, None) is not None:
                        hfile["headers"].create_dataset(
                            f"{band}",
                            data=str(self.phot_img_headers[band]),
                            dtype=str_dt,
                        )

                if self.psf_matched_data is not None:
                    hfile.create_group("psf_matched_data")
                    for psf_type in self.psf_matched_data.keys():
                        hfile["psf_matched_data"].create_group(psf_type)
                        for band in self.bands:
                            # print(band)
                            # print(self.psf_matched_data[psf_type])
                            hfile["psf_matched_data"][psf_type].create_dataset(
                                band,
                                data=self.psf_matched_data[psf_type][band],
                                compression="gzip",
                            )

                if self.psf_matched_rms_err is not None:
                    hfile.create_group("psf_matched_rms_err")
                    for psf_type in self.psf_matched_rms_err.keys():
                        hfile["psf_matched_rms_err"].create_group(psf_type)
                        for band in self.bands:
                            hfile["psf_matched_rms_err"][
                                psf_type
                            ].create_dataset(
                                band,
                                data=self.psf_matched_rms_err[psf_type][band],
                                compression="gzip",
                            )

                # or here
                # Save galaxy region

                # Save binned maps
                for map in self.maps:
                    hfile["bin_maps"].create_dataset(
                        map,
                        data=getattr(self, f"{map}_map"),
                        compression="gzip",
                    )

                if self.binned_flux_map is not None:
                    hfile["bin_fluxes"].create_dataset(
                        "pixedfit",
                        data=self.binned_flux_map,
                        compression="gzip",
                    )
                if self.binned_flux_err_map is not None:
                    hfile["bin_flux_err"].create_dataset(
                        "pixedfit",
                        data=self.binned_flux_err_map,
                        compression="gzip",
                    )
                if self.rms_background is not None:
                    hfile.create_dataset(
                        "meta/rms_background",
                        data=str(self.rms_background),
                        compression="gzip",
                    )

                # Small memory leak here - 0.2 MB per save

                # Save PSFs
                if self.psfs is not None and self.psfs != {}:
                    hfile.create_group("psfs")
                    for psf_type in self.psfs.keys():
                        hfile["psfs"].create_group(psf_type)
                        for band in self.bands:
                            if self.psfs[psf_type].get(band) is not None:
                                hfile["psfs"][psf_type].create_dataset(
                                    band,
                                    data=self.psfs[psf_type][band],
                                    compression="gzip",
                                )

                if self.psfs_meta is not None and self.psfs_meta != {}:
                    hfile.create_group("psfs_meta")
                    for psf_type in self.psfs_meta.keys():
                        hfile["psfs_meta"].create_group(psf_type)
                        for band in self.bands:
                            if self.psfs_meta[psf_type].get(band) is not None:
                                data = str(self.psfs_meta[psf_type][band])
                                hfile["psfs_meta"][psf_type].create_dataset(
                                    band,
                                    data=data,
                                    dtype=str_dt,
                                )

                # Add psf_Kernels
                if self.psf_kernels is not None and self.psf_kernels != {}:
                    hfile.create_group("psf_kernels")
                    for psf_type in self.psf_kernels.keys():
                        hfile["psf_kernels"].create_group(psf_type)
                        for band in self.bands:
                            if (
                                self.psf_kernels[psf_type].get(band)
                                is not None
                            ):
                                hfile["psf_kernels"][psf_type].create_dataset(
                                    band,
                                    data=self.psf_kernels[psf_type][band],
                                    compression="gzip",
                                )

                # Add galaxy region
                if self.gal_region is not None:
                    hfile.create_group("galaxy_region")
                    for binmap_type in self.gal_region.keys():
                        hfile["galaxy_region"].create_dataset(
                            binmap_type,
                            data=self.gal_region[binmap_type],
                            compression="gzip",
                        )
                # Add flux_map
                if self.flux_map is not None:
                    hfile.create_group("flux_map")
                    for binmap_type in self.flux_map.keys():
                        hfile["flux_map"].create_dataset(
                            binmap_type,
                            data=self.flux_map[binmap_type],
                            compression="gzip",
                        )

                for pos, property in enumerate(self.photometry_property_names):
                    if pos == 0:
                        hfile.create_group("photometry_properties")
                    hfile["photometry_properties"].create_dataset(
                        property, data=getattr(self, property)
                    )
                    # Add meta data
                    for key in getattr(self, f"{property}_meta").keys():
                        hfile["photometry_properties"][property].attrs[key] = (
                            getattr(self, f"{property}_meta")[key]
                        )

                # Add resolved mass
                if self.resolved_mass is not None:
                    hfile.create_group("resolved_mass")
                    for key in self.resolved_mass.keys():
                        hfile["resolved_mass"].create_dataset(
                            key, data=self.resolved_mass[key]
                        )

                if self.resolved_sfr_10myr is not None:
                    hfile.create_group("resolved_sfr_10myr")
                    for key in self.resolved_sfr_10myr.keys():
                        hfile["resolved_sfr_10myr"].create_dataset(
                            key,
                            data=self.resolved_sfr_10myr[key],
                            compression="gzip",
                        )

                if self.resolved_sfr_100myr is not None:
                    hfile.create_group("resolved_sfr_100myr")
                    for key in self.resolved_sfr_100myr.keys():
                        hfile["resolved_sfr_100myr"].create_dataset(
                            key,
                            data=self.resolved_sfr_100myr[key],
                            compression="gzip",
                        )

                # Add resolved SFH
                if self.resolved_sfh is not None:
                    hfile.create_group("resolved_sfh")
                    for key in self.resolved_sfh.keys():
                        hfile["resolved_sfh"].create_dataset(
                            key,
                            data=self.resolved_sfh[key],
                            compression="gzip",
                        )

                # Add resolved SED
                if self.resolved_sed is not None:
                    hfile.create_group("resolved_sed")
                    for key in self.resolved_sed.keys():
                        hfile["resolved_sed"].create_dataset(
                            key,
                            data=self.resolved_sed[key],
                            compression="gzip",
                        )

                # Add MockGalaxy properties
                if type(self) is MockResolvedGalaxy:
                    # Copy over the mock galaxy properties if they exist
                    hfile.create_group("mock_galaxy")

                    if self.noise_images is not None:
                        hfile["mock_galaxy"].create_group("noise_images")
                        for key in self.noise_images.keys():
                            hfile["mock_galaxy"][
                                "noise_images"
                            ].create_dataset(
                                key,
                                data=self.noise_images[key],
                                compression="gzip",
                            )
                    if self.property_images is not None:
                        hfile["mock_galaxy"].create_group("property_images")
                        for key in self.property_images.keys():
                            hfile["mock_galaxy"][
                                "property_images"
                            ].create_dataset(
                                key,
                                data=self.property_images[key],
                                compression="gzip",
                            )
                    if self.seds is not None:
                        hfile["mock_galaxy"].create_group("seds")
                        for key in self.seds.keys():
                            if type(self.seds[key]) is np.ndarray:
                                hfile["mock_galaxy"]["seds"].create_dataset(
                                    key,
                                    data=self.seds[key],
                                    compression="gzip",
                                )
                            else:
                                hfile["mock_galaxy"]["seds"].create_group(key)
                                for key2 in self.seds[key].keys():
                                    hfile["mock_galaxy"]["seds"][
                                        key
                                    ].create_dataset(
                                        key2,
                                        data=self.seds[key][key2],
                                        compression="gzip",
                                    )

                    if self.sfh is not None:
                        hfile["mock_galaxy"].create_group("sfh")
                        for key in self.sfh.keys():
                            if type(self.sfh[key]) is np.ndarray:
                                hfile["mock_galaxy"]["sfh"].create_dataset(
                                    key, data=self.sfh[key], compression="gzip"
                                )
                            else:
                                hfile["mock_galaxy"]["sfh"].create_group(key)
                                for key2 in self.sfh[key].keys():
                                    hfile["mock_galaxy"]["sfh"][
                                        key
                                    ].create_dataset(
                                        key2,
                                        data=self.sfh[key][key2],
                                        compression="gzip",
                                    )

                # Write interactive outputs
                if self.interactive_outputs is not None:
                    hfile.create_group("interactive_outputs")
                    for region_id in self.interactive_outputs.keys():
                        hfile["interactive_outputs"].create_group(region_id)
                        for key in self.interactive_outputs[region_id].keys():
                            if key == "meta":
                                pass
                            else:
                                data = self.interactive_outputs[region_id][key]
                                hfile["interactive_outputs"][
                                    region_id
                                ].create_dataset(
                                    key,
                                    data=data,
                                    compression="gzip",
                                )
                                if key == "eazy_fit":
                                    meta = self.interactive_outputs[region_id][
                                        "meta"
                                    ]
                                    for meta_key in meta.keys():
                                        hfile["interactive_outputs"][
                                            region_id
                                        ][key].attrs[key] = str(meta[meta_key])

                # Write photometry table(s)
                if self.photometry_table is not None:
                    for psf_type in self.photometry_table.keys():
                        for binmap_type in self.photometry_table[
                            psf_type
                        ].keys():
                            write_table_hdf5(
                                self.photometry_table[psf_type][binmap_type],
                                new_h5_path,
                                f"binned_photometry_table/{psf_type}/{binmap_type}",
                                serialize_meta=True,
                                overwrite=True,
                                append=True,
                            )
                # Write sed fitting table(s)
                if self.sed_fitting_table is not None:
                    for tool in self.sed_fitting_table.keys():
                        for run in self.sed_fitting_table[tool].keys():
                            write_table_hdf5(
                                self.sed_fitting_table[tool][run],
                                new_h5_path,
                                f"sed_fitting_table/{tool}/{run}",
                                serialize_meta=True,
                                overwrite=True,
                                append=True,
                            )

                # Add anything else from the old file to the new file
                exists = False
                if os.path.exists(self.h5_path) and append != "":
                    exists = True
                    with h5.File(self.h5_path, "r") as old_hfile:
                        # print("Removing temp", self.h5_path)
                        for key in old_hfile.keys():
                            # Check subgroups
                            if key not in hfile.keys():
                                print("Copying", key)
                                old_hfile.copy(key, hfile)

                            elif type(key) is h5.Group:
                                for subkey in key.keys():
                                    if subkey not in hfile[key].keys():
                                        print("Copying", key, subkey)
                                        old_hfile.copy(
                                            f"{key}/{subkey}", hfile[key]
                                        )

        except ValueError as e:
            print(f"Blocking Error: {e}")
            os.remove(new_h5_path)

            return False

        if exists:
            os.remove(self.h5_path)
            os.rename(
                self.h5_path.replace(".h5", f"{append}.h5"), self.h5_path
            )

    def convolve_with_psf(self, psf_type="webbpsf", init_run=False):
        """Convolve the images with the PSF
        psf_type: str - the type of PSF to use
        init_run: bool - if this is the inital run, don't load from .h5 or create .h5 file"""

        if (
            getattr(self, "psf_matched_data", None) in [None, {}]
            or getattr(self, "psf_matched_rms_err", None) in [None, {}]
            or psf_type not in self.psf_matched_data.keys()
        ):
            run = False
            # Try and load from .h5
            if not init_run and os.path.exists(self.h5_path):
                print("Creating PSF matched data", self.galaxy_id)

                h5file = h5.File(self.h5_path, "a")
                if "psf_matched_data" in h5file.keys():
                    if psf_type in h5file["psf_matched_data"].keys():
                        if len(self.bands) != len(
                            list(h5file["psf_matched_data"][psf_type].keys())
                        ):
                            run = True
                        else:
                            self.psf_matched_data = {psf_type: {}}
                            for band in self.bands:
                                self.psf_matched_data[psf_type][band] = h5file[
                                    "psf_matched_data"
                                ][psf_type][band][()]
                    else:
                        run = True
                else:
                    h5file.create_group("psf_matched_data")
                    run = True

                if "psf_matched_rms_err" in h5file.keys():
                    if psf_type in h5file["psf_matched_rms_err"].keys():
                        if len(self.bands) != len(
                            list(
                                h5file["psf_matched_rms_err"][psf_type].keys()
                            )
                        ):
                            run = True
                        else:
                            self.psf_matched_rms_err = {psf_type: {}}
                            for band in self.bands:
                                self.psf_matched_rms_err[psf_type][band] = (
                                    h5file[
                                        "psf_matched_rms_err"
                                    ][psf_type][band][()]
                                )
                    else:
                        run = True
                else:
                    h5file.create_group("psf_matched_rms_err")
                    run = True
            else:
                run = True

            if run:
                # Do the convolution\
                if getattr(self, "psf_matched_data", None) is None:
                    self.psf_matched_data = {psf_type: {}}
                else:
                    self.psf_matched_data[psf_type] = {}
                if getattr(self, "psf_matched_rms_err", None) is None:
                    self.psf_matched_rms_err = {psf_type: {}}
                else:
                    self.psf_matched_rms_err[psf_type] = {}

                for band in self.bands[:-1]:
                    print(f"Convolving {band} with PSF")
                    if (
                        band in self.dont_psf_match_bands
                        or self.already_psf_matched
                    ):
                        psf_matched_img = self.phot_imgs[band]
                        psf_matched_rms_err = self.rms_err_imgs[band]
                    else:
                        kernel = self.psf_kernels[psf_type][band]
                        # kernel = fits.open(kernel_path)[0].data
                        # Convolve the image with the PSF

                        psf_matched_img = convolve_fft(
                            self.phot_imgs[band], kernel, normalize_kernel=True
                        )
                        psf_matched_rms_err = convolve_fft(
                            self.rms_err_imgs[band],
                            kernel,
                            normalize_kernel=True,
                        )

                    try:
                        from piXedfit.piXedfit_images.images_utils import (
                            remove_naninfzeroneg_image_2dinterpolation,
                        )

                        psf_matched_rms_err = (
                            remove_naninfzeroneg_image_2dinterpolation(
                                psf_matched_rms_err
                            )
                        )
                    except Exception as e:
                        print(e)
                        print("Didnt work")
                        pass

                    # Save to psf_matched_data
                    self.psf_matched_data[psf_type][band] = psf_matched_img
                    self.psf_matched_rms_err[psf_type][band] = (
                        psf_matched_rms_err
                    )

                    if not init_run:
                        if (
                            h5file.get(f"psf_matched_data/{psf_type}/{band}")
                            is not None
                        ):
                            del h5file[f"psf_matched_data/{psf_type}/{band}"]
                        if (
                            h5file.get(
                                f"psf_matched_rms_err/{psf_type}/{band}"
                            )
                            is not None
                        ):
                            del h5file[
                                f"psf_matched_rms_err/{psf_type}/{band}"
                            ]

                        h5file.create_dataset(
                            f"psf_matched_data/{psf_type}/{band}",
                            data=psf_matched_img,
                        )
                        h5file.create_dataset(
                            f"psf_matched_rms_err/{psf_type}/{band}",
                            data=psf_matched_rms_err,
                        )

                    # h5file['psf_matched_data'][psf_type] = self.psf_matched_data

                # print(self.phot_imgs.keys(), self.bands)
                self.psf_matched_data[psf_type][self.bands[-1]] = (
                    self.phot_imgs[self.bands[-1]]
                )  # No need to convolve the last band
                self.psf_matched_rms_err[psf_type][self.bands[-1]] = (
                    self.rms_err_imgs[self.bands[-1]]
                )

                if not init_run:
                    # Deal with last band
                    if (
                        h5file.get(
                            f"psf_matched_data/{psf_type}/{self.bands[-1]}"
                        )
                        is not None
                    ):
                        del h5file[
                            f"psf_matched_data/{psf_type}/{self.bands[-1]}"
                        ]
                    if (
                        h5file.get(
                            f"psf_matched_rms_err/{psf_type}/{self.bands[-1]}"
                        )
                        is not None
                    ):
                        del h5file[
                            f"psf_matched_rms_err/{psf_type}/{self.bands[-1]}"
                        ]
                    data = data = self.phot_imgs[self.bands[-1]]
                    data = data.value if type(data) is u.Quantity else data

                    h5file.create_dataset(
                        f"psf_matched_data/{psf_type}/{self.bands[-1]}",
                        data=data,
                    )
                    data = self.rms_err_imgs[self.bands[-1]]
                    data = data.value if type(data) is u.Quantity else data
                    h5file.create_dataset(
                        f"psf_matched_rms_err/{psf_type}/{self.bands[-1]}",
                        data=data,
                    )

            else:
                print("not running")

            if not init_run:
                h5file.close()

        else:
            print("Already PSF matched data.")

    def __str__(self):
        str = f"Resolved Galaxy {self.galaxy_id} from {self.survey} survey\n"
        str += f"SkyCoord: {self.sky_coord}\n"
        str += f"Bands: {self.bands}\n"
        str += f"Cutout size: {self.cutout_size}\n"
        str += f"Aperture photometry: {self.aperture_dict}\n"
        if hasattr(self, "pixedfit_map") and self.pixedfit_map is not None:
            str += f"Number of bins: {self.get_number_of_bins()}\n"
            # Make an ascii map of the bins
            test_map = copy.copy(self.pixedfit_map)
            unique_vals = np.unique(test_map)
            test_map = test_map.astype(object)
            for val in unique_vals:
                if val in [0, np.nan]:
                    fill_val = " "
                else:
                    fill_val = chr(int(val) + 65)

                test_map[test_map == val] = fill_val
            str += "Pixel-fit map\n"
            for row in test_map:
                str += f"{''.join(row)}\n"

        return str

    def __repr__(self):
        return self.__str__()

    def plot_cutouts(
        self,
        bands=None,
        psf_matched=False,
        save=False,
        psf_type=None,
        save_path=None,
        show=False,
        facecolor="white",
        fig=None,
    ):
        """Plot the cutouts for the galaxy"""
        if bands is None:
            bands = self.bands
        nrows = len(bands) // 6 + (1 if len(bands) % 6 else 0)
        ncols = min(6, len(bands))

        if fig is None:
            fig = plt.figure(
                figsize=(ncols * 3, nrows * 3), facecolor=facecolor
            )

        # Create axes within the figure
        axes = fig.subplots(nrows, ncols, squeeze=False)
        axes = axes.flatten()

        for i, (ax, band) in enumerate(zip(axes, bands)):
            if psf_matched:
                if psf_type is None:
                    psf_type = self.use_psf_type
                data = self.psf_matched_data[psf_type][band]
            else:
                if self.unmatched_data is not None:
                    data = self.unmatched_data[band]
                else:
                    data = self.phot_imgs[band]

            if type(data) is u.Quantity:
                data = data.value

            # Set normalization by brightest pixel in central 30x30 pixels

            central_data = np.copy(data)[
                self.cutout_size // 2 - 15 : self.cutout_size // 2 + 15,
                self.cutout_size // 2 - 15 : self.cutout_size // 2 + 15,
            ]
            norm = simple_norm(central_data, stretch="log", max_percent=99.9)
            im = ax.imshow(
                data, origin="lower", interpolation="none", norm=norm
            )
            ax.text(
                0.9,
                0.9,
                band,
                color="w",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=12,
                fontweight="bold",
                path_effects=[
                    PathEffects.Stroke(linewidth=2, foreground="black"),
                    PathEffects.Normal(),
                ],
            )
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("equal", adjustable="box")

        # Remove any unused subplots
        for ax in axes[len(bands) :]:
            fig.delaxes(ax)

        # Adjust spacing between subplots
        fig.subplots_adjust(wspace=0, hspace=0)

        if save:
            plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
        if show:
            plt.show()
        else:
            return fig

    def get_webbpsf(
        self,
        plot=False,
        overwrite=False,
        fov=4,
        og_fov=10,
        oversample=4,
        PATH_SW_ENERGY="psfs/Encircled_Energy_SW_ETCv2.txt",
        PATH_LW_ENERGY="psfs/Encircled_Energy_LW_ETCv2.txt",
    ):
        skip = False
        if getattr(self, "psfs", None) not in [None, {}, []]:
            skip = True
        if "webbpsf" in self.psfs.keys():
            skip = True

        if not skip or overwrite:
            import webbpsf

            self.psfs["webbpsf"] = {}
            self.psfs_meta["webbpsf"] = {}
            psfs = {}
            psf_headers = {}
            for band in self.bands:
                # Get dimensions from header
                header = fits.open(self.im_paths[band])[
                    self.im_exts[band]
                ].header
                header_0 = fits.open(self.im_paths[band])[0].header
                # Get dimensions

                xdim = header["NAXIS1"]
                ydim = header["NAXIS2"]

                if 10000 < xdim < 10400 & 4200 < ydim < 4400:
                    print(
                        "Dimensions consistent with a single NIRCam pointing"
                    )

                x_pos, y_pos = WCS(header).all_world2pix(
                    self.sky_coord.ra.deg, self.sky_coord.dec.deg, 0
                )
                # Calculate which NIRCam detector the galaxy is on
                if float(band[1:-1]) < 240:
                    wav = "SW"
                    jitter = 0.022
                    # 17 corresponds with 2" radius (i.e. 4" FOV)
                    energy_table = ascii.read(PATH_SW_ENERGY)
                    row = np.argmin(
                        abs(fov / 2.0 - energy_table["aper_radius"])
                    )
                    encircled = energy_table[row][filt]
                    norm_fov = energy_table["aper_radius"][row] * 2
                    print(
                        f'Will normalize PSF within {norm_fov}" FOV to {encircled}'
                    )

                else:
                    wav = "LW"
                    jitter = 0.034
                    energy_table = ascii.read(PATH_LW_ENERGY)
                    row = np.argmin(
                        abs(fov / 2.0 - energy_table["aper_radius"])
                    )
                    encircled = energy_table[row][filt]
                    norm_fov = energy_table["aper_radius"][row] * 2
                    print(
                        f'Will normalize PSF within {norm_fov}" FOV to {encircled}'
                    )

                if wav == "LW" and x_pos < 10244 / 2:
                    det = "A5"
                else:
                    det = "B5"

                if wav == "SW":
                    if x_pos < 10244 / 2:
                        det = "A"
                        center_rot = (2190, 2190)
                    else:
                        det = "B"
                        center_rot = (8000, 2250)

                    # Calculate rotation angle from vertical
                    rot = np.arctan(
                        (y_pos - center_rot[1]) / (x_pos - center_rot[0])
                    )
                    rot = rot * 180 / np.pi
                    if 0 < rot < 90:
                        det += "1"
                    elif 90 < rot < 180:
                        det += "3"
                    elif -90 < rot < 0:
                        det += "2"
                    elif -180 < rot < -90:
                        det += "4"

                    print(
                        f"Galaxy at {self.sky_coord.ra.deg} ({x_pos}), {self.sky_coord.dec.deg} ({y_pos}) is on NIRCam {wav} detector {det}"
                    )

                print(f'{filt} at {fov}" FOV')

                # If consistent with a single NIRCam pointing
                nircam = webbpsf.NIRCam()
                date = header_0["DATE-OBS"]
                nircam.load_wss_opd_by_date(date, plot=False)
                # nircam = webbpsf.setup_sim_to_match_file(self.im_paths[band])
                nircam.options["detector"] = f"NRC{det}"
                # Can set nircam.options['source_offset_theta'] = position_angle if not oriented vertical

                nircam.filter = band
                nircam.options["output_mode"] = "detector sampled"
                nircam.options["parity"] = "odd"
                nircam.options["jitter_sigma"] = jitter
                print("Calculating PSF")
                fov = (
                    self.cutout_size
                    * self.im_pixel_scales[band].to(u.arcsec).value
                )
                print(f"Size: {fov} arcsec")

                nircam.pixel_scale = (
                    self.im_pixel_scales[band].to(u.arcsec).value
                )

                psf = nircam.calc_psf(
                    fov_arcsec=og_fov, normalize="exit_pupil", oversample=4
                )
                # Drop the first element from the HDU

                psf_data = psf["DET_SAMP"].data

                clip = int(
                    (og_fov - fov)
                    / 2
                    / self.im_pixel_scales[band].to(u.arcsec).value
                )
                psf_data = psf_data[clip:-clip, clip:-clip]

                w, h = np.shape(psf_data)
                Y, X = np.ogrid[:h, :w]
                r = norm_fov / 2.0 / nc.pixelscale
                center = [w / 2.0, h / 2.0]
                dist_from_center = np.sqrt(
                    (X - center[0]) ** 2 + (Y - center[1]) ** 2
                )
                psf_data /= np.sum(psf_data[dist_from_center < r])
                psf_data *= encircled  # to get the missing flux accounted for
                print(f"Final stamp normalization: {rotated.sum()}")

                psfs[band] = psf_data
                psf_headers[band] = str(psf[1].header)
                if plot:
                    webbpsf.display_psf(psf)
                    # plt.show()
                psf = fits.PrimaryHDU(psf_data)

                dir = f"{self.psf_folder}/webbpsf/{self.survey}_{self.galaxy_id}/"
                os.makedirs(dir, exist_ok=True)
                psf.writeto(f"{dir}/webbpsf_{band}.fits", overwrite=True)

                self.add_to_h5(psf.data, "psfs/webbpsf/", band, overwrite=True)
                self.add_to_h5(
                    str(psf.header), "psfs_meta/webbpsf/", band, overwrite=True
                )

                self.psfs["webbpsf"][band] = psf.data
                self.psfs_meta["webbpsf"][band] = str(
                    psf.header
                )  # This seems to be large? Maybe just save essentials

        else:
            print("Webbpsf PSFs already calculated")
            print("Saving to run pypher")
            for band in self.bands:
                dir = f"{self.psf_folder}/{self.survey}_{self.galaxy_id}/"
                os.makedirs(dir, exist_ok=True)
                hdu = fits.ImageHDU(
                    self.psfs["webbpsf"][band],
                    header=fits.Header.fromstring(
                        self.psfs_meta["webbpsf"][band], sep="\n"
                    ),
                )
                hdu.writeto(f"{dir}/webbpsf_{band}.fits", overwrite=True)

        self.convert_psfs_to_kernels(
            match_band=self.bands[-1], psf_type="webbpsf"
        )

    def convert_psfs_to_kernels(
        self, match_band=None, psf_type="webbpsf", oversample=3
    ):
        if match_band is None:
            match_band = self.bands[-1]

        target_psf = self.psfs[psf_type][match_band]

        dir = f"{self.psf_kernel_folder}/{psf_type}/{self.survey}"
        dir += f"_{self.galaxy_id}/" if psf_type == "webbpsf" else "/"

        kernel_dir = f"{self.psf_kernel_folder}/{psf_type}/{self.survey}"
        kernel_dir += f"_{self.galaxy_id}/" if psf_type == "webbpsf" else "/"

        # target_psf = fits.getdata(f'{dir}/{psf_type}_{match_band}.fits')

        if oversample > 1:
            print(f"Oversampling PSF by {oversample}x...")
            target_psf = zoom(target_psf, oversample)

        print("Normalizing PSF to unity...")
        target_psf /= target_psf.sum()
        os.makedirs(dir, exist_ok=True)
        fits.writeto(f"{dir}/{psf_type}_a.fits", target_psf, overwrite=True)

        command = [
            "addpixscl",
            f"{dir}/{psf_type}_a.fits",
            f"{self.im_pixel_scales[match_band].to(u.arcsec).value}",
        ]
        os.system(" ".join(command))
        print("Computing kernels for PSF matching to ", match_band)
        for band in self.bands[:-1]:
            if band in self.dont_psf_match_bands:
                continue
            # filt_psf = fits.getdata(f'{dir}/{psf_type}_{match_band}.fits')
            filt_psf = self.psfs[psf_type][band]
            if oversample:
                filt_psf = zoom(filt_psf, oversample)

            filt_psf /= filt_psf.sum()

            fits.writeto(f"{dir}/{psf_type}_b.fits", filt_psf, overwrite=True)

            # Need ! pip install pypher first if not installed

            command = [
                "addpixscl",
                f"{dir}/{psf_type}_b.fits",
                f"{self.im_pixel_scales[band].to(u.arcsec).value}",
            ]
            os.system(" ".join(command))
            try:
                os.remove(f"{dir}/kernel.fits")
            except:
                pass
            command = [
                "pypher",
                f"{dir}/{psf_type}_b.fits",
                f"{dir}/{psf_type}_a.fits",
                f"{dir}/kernel.fits",
                "-r",
                "3e-3",
            ]
            # print(' '.join(command))
            os.system(" ".join(command))
            if not os.path.exists(f"{dir}/kernel.fits"):
                print("Pypher hasn't made a file!")
            kernel = fits.getdata(f"{dir}/kernel.fits")

            os.remove(f"{dir}/{psf_type}_b.fits")

            if oversample > 1:
                kernel = block_reduce(
                    kernel, block_size=oversample, func=np.sum
                )
                kernel /= kernel.sum()
            os.makedirs(kernel_dir, exist_ok=True)
            fits.writeto(
                f"{kernel_dir}/kernel_{band}_to_{match_band}.fits",
                kernel,
                overwrite=True,
            )
            if self.psf_kernels is None:
                self.psf_kernels = {}
            if self.psf_kernels.get(psf_type) is None:
                self.psf_kernels[psf_type] = {}
            self.psf_kernels[psf_type][band] = kernel
            # f'{self.psf_kernel_dir}/{psf_type}/kernel_{band}_to_{match_band}.fits'

            os.remove(f"{dir}/kernel.fits")
            os.remove(f"{dir}/kernel.log")

        os.remove(f"{dir}/{psf_type}_a.fits")

    def add_det_galaxy_region(self, force=False, overwrite=False):
        if self.det_data is not None:
            det_galaxy_region = copy.copy(self.det_data["seg"])
            # Value of segmap in center
            center = int(self.cutout_size // 2)
            possible_vals = np.unique(det_galaxy_region)
            if len(possible_vals) == 2:
                det_gal_mask = det_galaxy_region == np.max(possible_vals)
            elif len(possible_vals) > 2:
                center_val = det_galaxy_region[center, center]
                mask = det_galaxy_region == center_val
                det_gal_mask = np.zeros_like(det_galaxy_region)
                det_gal_mask[mask] = True
            else:
                raise ValueError(
                    f"No detection pixels found. Only value in map is {possible_vals}"
                )

            det_gal_mask = det_gal_mask.astype(int)

            self.gal_region["detection"] = det_gal_mask

            if len(np.unique(det_gal_mask)) == 1:
                print(
                    f"WARNING! Only one value in detection region for {self.galaxy_id}. Check!"
                )

            self.add_to_h5(
                det_gal_mask,
                "galaxy_region",
                "detection",
                force=force,
                overwrite=overwrite,
            )

    def plot_lupton_rgb(
        self,
        red=[],
        green=[],
        blue=[],
        q=1,
        stretch=1,
        figsize=(8, 8),
        use_psf_matched=False,
        override_psf_type=None,
        return_array=True,
        save=False,
        save_path=None,
        show=False,
        fig=None,
        ax=None,
        add_compass=False,
        compass_arrow_length=0.5 * u.arcsec,
        compass_arrow_width=0.5,
        compass_center=(2, 17),
        compass_text_scale_factor=1.15,
        text_fontsize="large",
        add_scalebar=False,
        label_bands=False,
    ):
        """Plot the galaxy in Lupton RGB"""

        if hasattr(self, "use_psf_type") and override_psf_type is None:
            psf_type = self.use_psf_type
        else:
            psf_type = override_psf_type

        if type(red) is str:
            red = [red]
        if type(green) is str:
            green = [green]
        if type(blue) is str:
            blue = [blue]

        if use_psf_matched:
            img = self.psf_matched_data[psf_type]
        else:
            if self.unmatched_data is not None:
                img = self.unmatched_data
            else:
                img = self.phot_imgs
        if len(red) == 0:
            r = np.zeros((self.cutout_size, self.cutout_size))
        else:
            r = np.sum([img[band] for band in red], axis=0)
        if len(green) == 0:
            g = np.zeros((self.cutout_size, self.cutout_size))
        else:
            g = np.sum([img[band] for band in green], axis=0)

        if len(blue) == 0:
            b = np.zeros((self.cutout_size, self.cutout_size))
        else:
            b = np.sum([img[band] for band in blue], axis=0)

        rgb = make_lupton_rgb(r, g, b, Q=q, stretch=stretch)
        if return_array:
            return rgb

        if fig is None:
            fig = plt.figure(figsize=figsize, dpi=200)
        if ax is None:
            ax = fig.add_subplot(111)

        ax.imshow(rgb, origin="lower")

        # disable x and y ticks
        ax.set_xticks([])
        ax.set_yticks([])

        self.plot_kron_ellipse(
            ax=ax, center=self.cutout_size / 2, band="detection"
        )
        if label_bands:
            ax.text(
                0.03,
                0.98,
                f'{"+".join(red)}',
                color="red",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=text_fontsize,
                fontweight="bold",
                path_effects=[
                    PathEffects.Stroke(linewidth=2, foreground="white"),
                    PathEffects.Normal(),
                ],
            )
            ax.text(
                0.03,
                0.9,
                f'{"+".join(green)}',
                color="green",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=text_fontsize,
                fontweight="bold",
                path_effects=[
                    PathEffects.Stroke(linewidth=2, foreground="white"),
                    PathEffects.Normal(),
                ],
            )
            ax.text(
                0.03,
                0.82,
                f'{"+".join(blue)}',
                color="blue",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=text_fontsize,
                fontweight="bold",
                path_effects=[
                    PathEffects.Stroke(linewidth=2, foreground="white"),
                    PathEffects.Normal(),
                ],
            )

        # ra, dec, wcs, axis, scale=0.10, x_ax ='ra', y_ax='dex',ang_text=False, arrow_width=200, arrow_color="black", text_color="black", fontsize="large", return_ang=False):

        if add_compass:
            wcs = WCS(self.phot_img_headers["F444W"])
            # Crop WCS to cutout size centered on galaxy
            # convert skycoord back to pixel coordinates and then to WCS
            x_pix, y_pix = wcs.all_world2pix(
                self.sky_coord.ra.deg, self.sky_coord.dec.deg, 0
            )
            wcs = wcs[
                int(y_pix - self.cutout_size / 2) : int(
                    y_pix + self.cutout_size / 2
                ),
                int(x_pix - self.cutout_size / 2) : int(
                    x_pix + self.cutout_size / 2
                ),
            ]

            # Put arrow in bottom corner - calculate wcs coords of (5, 10)

            ra, dec = wcs.all_pix2world(
                compass_center[0], compass_center[1], 0
            )

            compass(
                ra,
                dec,
                wcs,
                ax,
                arrow_length=compass_arrow_length,
                x_ax="ra",
                ang_text=False,
                arrow_width=compass_arrow_width,
                arrow_color="white",
                text_color="white",
                fontsize=text_fontsize,
                return_ang=False,
                compass_text_scale_factor=compass_text_scale_factor,
            )
        if add_scalebar:
            re = 15  # pixels
            d_A = cosmo.angular_diameter_distance(self.redshift)
            pix_scal = u.pixel_scale(
                self.im_pixel_scales["F444W"].value * u.arcsec / u.pixel
            )
            re_as = (re * u.pixel).to(u.arcsec, pix_scal)
            re_kpc = (re_as * d_A).to(u.kpc, u.dimensionless_angles())

            # First scalebar
            scalebar = AnchoredSizeBar(
                ax.transData,
                0.5 / self.im_pixel_scales["F444W"].value,
                '0.5"',
                "lower right",
                pad=0.3,
                color="white",
                frameon=False,
                size_vertical=1,
                fontproperties=FontProperties(size=text_fontsize),
            )
            ax.add_artist(scalebar)
            # Plot scalebar with physical size
            scalebar = AnchoredSizeBar(
                ax.transData,
                re,
                f"{re_kpc:.1f}",
                "lower left",
                pad=0.3,
                color="white",
                frameon=False,
                size_vertical=1,
                fontproperties=FontProperties(size=text_fontsize),
            )
            scalebar.set(
                path_effects=[
                    PathEffects.withStroke(linewidth=3, foreground="white")
                ]
            )
            ax.add_artist(scalebar)

        if save:
            plt.savefig(save_path)
        if show:
            plt.show()

    def pixedfit_processing(
        self,
        use_galfind_seg=True,
        seg_combine=["F277W", "F356W", "F444W"],
        gal_region_use="pixedfit",
        dir_images=resolved_galaxy_dir,
        override_psf_type=None,
        use_all_pixels=False,
        overwrite=False,
        load_anyway=True,  # Won't overwrite, but will create the object.
    ):
        if hasattr(self, "use_psf_type") and override_psf_type is None:
            psf_type = self.use_psf_type
        else:
            psf_type = override_psf_type

        from piXedfit.piXedfit_images import images_processing

        if not load_anyway and (
            self.gal_region is not None
            and gal_region_use in self.gal_region.keys()
            and not overwrite
        ):
            print("Galaxy regions already calculated")
            return

        if not os.path.exists(dir_images):
            os.makedirs(dir_images)

        instruments = [
            "hst_acs"
            if band.lower() in ["f435w", "f606w", "f775w", "f814w", "f850lp"]
            else "jwst_nircam"
            for band in self.bands
        ]
        filters = [
            f"{instrument}_{band.lower()}"
            for instrument, band in zip(instruments, self.bands)
        ]

        sci_img = {}
        var_img = {}
        img_unit = {}
        scale_factors = {}

        for f, band in zip(filters, self.bands):
            data = self.psf_matched_data[psf_type][band]
            err = self.psf_matched_rms_err[psf_type][band]
            var = np.square(err)
            if (
                self.phot_img_headers is None
                or band not in self.phot_img_headers.keys()
            ):
                header = fits.Header()
            else:
                header = Header.fromstring(
                    self.phot_img_headers[band], sep="\n"
                )
            if type(data) is u.Quantity:
                data = data.value
            if type(var) is u.Quantity:
                var = var.value

            hdu = fits.PrimaryHDU(data, header=header)
            path = f"{dir_images}/crop_{band}_sci.fits"
            hdu.writeto(path, overwrite=True)
            sci_img[f] = Path(path).name

            err_path = f"{dir_images}/crop_{band}_var.fits"
            hdu = fits.PrimaryHDU(var, header=header)
            hdu.writeto(err_path, overwrite=True)
            var_img[f] = Path(err_path).name
            unit = self.phot_pix_unit[band]
            unit_str = {
                u.Jy: "Jy",
                u.MJy / u.sr: "MJy/sr",
                u.erg / u.s / u.cm**2 / u.AA: "erg/s/cm2/A",
            }

            img_unit[f] = unit_str.get(unit, unit)

            scale_factor = (1 * unit) / (1 * u.Jy)
            scale_factor = scale_factor.decompose()
            if scale_factor.unit == u.dimensionless_unscaled:
                scale_factor = scale_factor.value
                img_unit[f] = "Jy"
            else:
                scale_factor = 1

            scale_factors[f] = scale_factor
            # If Unit is some power of 10 of Jy, calculate scale factor to Jy

        img_pixsizes = {
            f: float(self.im_pixel_scales[band].to(u.arcsec).value)
            for f, band in zip(filters, self.bands)
        }

        if self.sky_coord is None:
            gal_ra = 0.0
            gal_dec = 0.0
        else:
            gal_ra = self.sky_coord.ra.deg
            gal_dec = self.sky_coord.dec.deg

        gal_z = -1  # PLACEHOLDER -  NOT USED

        flag_psfmatch = True
        flag_reproject = True
        flag_crop = True

        remove_files = True

        stamp_size = (self.cutout_size, self.cutout_size)

        # print("sci_img", sci_img)
        # print("var_img", var_img)
        # print("img_unit", img_unit)
        # print("img_pixsizes", img_pixsizes)
        # print("scale_factors", scale_factors)

        img_process = images_processing(
            filters,
            sci_img,
            var_img,
            gal_ra,
            gal_dec,
            dir_images=dir_images,
            img_unit=img_unit,
            img_scale=scale_factors,
            img_pixsizes=img_pixsizes,
            run_image_processing=True,
            stamp_size=(self.cutout_size, self.cutout_size),
            flag_psfmatch=flag_psfmatch,
            flag_reproject=flag_reproject,
            flag_crop=flag_crop,
            kernels=None,
            gal_z=gal_z,
            remove_files=remove_files,
        )

        seg_type = "galfind"
        # Get galaxy region from segmentation map
        if not use_galfind_seg:
            print("Making segmentation maps")
            img_process.segmentation_sep()
            self.seg_imgs = img_process.seg_maps
            seg_type = "pixedfit_sep"
            # self.add_to_h5(img_process.segm_maps, 'seg_maps', 'pixedfit', ext='SEG_MAPS')

        if use_all_pixels:
            segm_maps = [
                np.ones_like(self.seg_imgs[band]) for band in self.bands
            ]
            img_process.segm_maps = segm_maps
            segm_maps_ids = None
            seg_type = "all_pixels"
        else:
            segm_maps = []
            for band in self.bands:
                segm = self.seg_imgs[band]
                # change to 0 is background, 1 is galaxy
                # Get value in center
                # print(self.cutout_size//2)
                center = int(self.cutout_size // 2)
                center = segm[center, center]

                segm[segm == center] = 1
                segm[segm != 1] = 0
                # print(np.count_nonzero(segm))
                segm_maps.append(segm)

            img_process.segm_maps = segm_maps

            if seg_combine is not None:
                segm_maps_ids = np.argwhere(
                    np.array([band in seg_combine for band in self.bands])
                ).flatten()
            else:
                segm_maps_ids = None

        if self.gal_region is None:
            self.gal_region = {}
        galaxy_region = img_process.galaxy_region(segm_maps_ids=segm_maps_ids)

        self.gal_region[gal_region_use] = galaxy_region

        if self.det_data is not None:
            det_galaxy_region = copy.copy(self.det_data["seg"])
            # Value of segmap in center
            center = int(self.cutout_size // 2)
            possible_vals = np.unique(det_galaxy_region)
            if len(possible_vals) == 2:
                det_gal_mask = det_galaxy_region == np.max(
                    possible_vals
                )  # Since 0 is background
            elif len(possible_vals) > 2:
                center_val = det_galaxy_region[center, center]
                mask = det_galaxy_region == center_val
                det_gal_mask = np.zeros_like(det_galaxy_region)
                det_gal_mask[mask] = True
            else:
                raise ValueError(
                    f"Detected only one value in detection region for {self.galaxy_id}. Check!"
                )

            det_gal_mask = det_gal_mask.astype(int)

            self.gal_region["detection"] = det_gal_mask
            if gal_region_use == "detection":
                seg_type = "detection"

        # Difference between gal_region - which I think is one image for all bands.

        # Calculate maps of multiband fluxes
        flux_maps_fits = (
            f"{dir_images}/{self.survey}_{self.galaxy_id}_fluxmap.fits"
        )
        Gal_EBV = 0  # Placeholder

        # Check gal_region is same shape as flux map
        assert (
            np.shape(self.gal_region[gal_region_use])
            == (self.cutout_size, self.cutout_size)
        ), f"Galaxy region shape {np.shape(self.gal_region[gal_region_use])} not same as cutout size {self.cutout_size}"
        mmap = copy.copy(self.gal_region[gal_region_use])
        assert (
            len(np.unique(mmap)) > 1
        ), f"Galaxy region is not binary {np.unique(mmap)}"

        img_process.flux_map(
            mmap,
            Gal_EBV=Gal_EBV,
            name_out_fits=flux_maps_fits,
        )
        if remove_files:
            files = glob.glob("*crop_*")
            for file in files:
                os.remove(file)
            files = glob.glob(dir_images + "/crop_*")
            for file in files:
                os.remove(file)

        self.flux_map_path = flux_maps_fits
        self.img_process = img_process

        meta_dict = {
            "stacked_bands": "+".join(seg_combine),
            "seg_type": seg_type,
            "use_all_pixels": use_all_pixels,
            "gal_region_use": gal_region_use,
        }

        if gal_region_use == "pixedfit":
            self.add_to_h5(
                galaxy_region,
                "galaxy_region",
                "pixedfit",
                meta=meta_dict,
                overwrite=overwrite,
            )
        else:
            self.add_to_h5(
                galaxy_region,
                "galaxy_region",
                gal_region_use,
                meta=meta_dict,
                overwrite=overwrite,
            )

        ## What I have named 'galaxy_region' is actually flux_map_fits
        self.add_to_h5(
            flux_maps_fits, "flux_map", gal_region_use, overwrite=overwrite
        )

        # Copy flux map to h5 file - TODO
        self.dir_images = dir_images

        return img_process

    def plot_voronoi_map(self):
        if self.voronoi_map is None:
            print("No Voronoi map found")
            return

        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        mappable = ax.imshow(
            self.voronoi_map,
            origin="lower",
            interpolation="none",
            cmap="nipy_spectral_r",
        )
        fig.colorbar(mappable, ax=ax)
        ax.set_title("Voronoi Map")
        # plt.show()

    def plot_snr_map(
        self, band="All", override_psf_type=None, facecolor="white", show=False
    ):
        if hasattr(self, "use_psf_type") and override_psf_type is None:
            psf_type = self.use_psf_type
        else:
            psf_type = override_psf_type

        bands = self.bands if band == "All" else [band]
        nrows = len(bands) // 6 + 1
        fig, axes = plt.subplots(
            nrows, 6, figsize=(18, 4 * nrows), facecolor=facecolor
        )
        axes = axes.flatten()
        # Remove empty axes
        for i in range(len(bands), len(axes)):
            fig.delaxes(axes[i])

        for i, band in enumerate(bands):
            snr_map = (
                self.psf_matched_data[psf_type][band]
                / self.psf_matched_rms_err[psf_type][band]
            )
            mappable = axes[i].imshow(
                snr_map, origin="lower", interpolation="none"
            )
            cax = make_axes_locatable(axes[i]).append_axes(
                "right", size="5%", pad=0.05
            )
            fig.colorbar(mappable, ax=axes[i], cax=cax)
            axes[i].set_title(f"{band} SNR Map")
            # Turn of ticklabels
            axes[i].set_xticklabels([])
            axes[i].set_yticklabels([])

        if show:
            plt.show()
        else:
            return fig
        # plt.show()

    def voronoi_binning(
        self,
        galaxy_region,
        SNR_reqs=10,
        ref_band="combined",
        plot=True,
        override_psf_type=None,
        overwrite=False,
        wvt=True,
        cvt=True,
        quiet=True,
        min_band="auto",
        band_instrument="NIRCam",
        min_snr_wav=1216 * u.AA,
        redshift="self",
        use_only_widebands=False,
    ):
        """
        Perform Voronoi binning on the galaxy region.

        Parameters
        ----------
        galaxy_region : str or np.ndarray
            If str, must be a key in self.gal_region. If np.ndarray, must be a 2D array with 1s for galaxy pixels and 0s for background pixels.
        SNR_reqs : float, optional
            SNR/pixel required for binning, by default 10
        ref_band : str or List[str], optional
            Band(s) to use as reference for SNR calculation. If list, will sum the bands.
            If 'combined_min', will use the band with the lowest SNR for each pixel. If 'combined_sum', will sum the bands.
        plot : bool, optional
            Whether to plot the SNR map, by default True
        override_psf_type : str, optional
            If not None, will use this PSF type instead of self.use_psf_type, by default None
        overwrite : bool, optional
            If True, will overwrite existing Voronoi map, by default False
        wvt : bool, optional
            If True, will use wavelet transform to calculate SNR, by default True
        cvt : bool, optional
            If True, will use curvelet transform to calculate SNR, by default True
        quiet : bool, optional
            If True, will suppress output from voronoi_2d_binning, by default True
        min_band : str, optional
            If 'ref_band' starts with 'combined', will use this band to calculate SNR, by default 'auto'
        band_instrument : str, optional
            if 'min_band' is 'auto', will use this instrument only to calculate bands to include in SNR, by default 'NIRCam'
        min_snr_wav : Quantity, optional
            if 'min_band' is 'auto', will use this wavelength to calculate bands to include in SNR, by default 1216*u.AA
        redshift : float or str, optional
            If 'min_band' is 'auto', will use this redshift to calculate bands to include in SNR, by default 'self'
        use_only_widebands : bool, optional
            If True, will only use widebands in SNR calculation, by default False

        """

        if hasattr(self, "use_psf_type") and override_psf_type is None:
            psf_type = self.use_psf_type
        else:
            psf_type = override_psf_type

        if hasattr(self, "voronoi_map") and not overwrite:
            print("Voronoi map already exists. Set overwrite=True to re-run")
            return

        from vorbin.voronoi_2d_binning import voronoi_2d_binning

        if type(galaxy_region) is str:
            if galaxy_region not in self.gal_region.keys():
                raise ValueError(
                    f"Galaxy region {galaxy_region} not found. Available regions: {self.gal_region.keys()}"
                )
            galaxy_region = copy.copy(self.gal_region[galaxy_region])

        x = np.arange(self.cutout_size)
        y = np.arange(self.cutout_size)
        x, y = np.meshgrid(x, y)
        # appy galaxy region mask to remove background pixels
        x = x[galaxy_region == 1]
        y = y[galaxy_region == 1]

        x = x.flatten()
        y = y.flatten()

        if ref_band.startswith("combined"):
            print(f"start {self.bands[-1]}")
            if ref_band.endswith("min"):
                signal = self.psf_matched_data[psf_type][self.bands[-1]]
                noise = self.psf_matched_rms_err[psf_type][self.bands[-1]]
            elif ref_band.endswith("sum"):
                signal = np.zeros_like(
                    self.psf_matched_data[psf_type][self.bands[-1]]
                )
                noise = np.zeros_like(
                    self.psf_matched_rms_err[psf_type][self.bands[-1]]
                )
            elif ref_band.endswith("average"):
                signal = []
                noise = []

            if redshift == "self":
                z = self.redshift
            elif type(redshift) in [float, int]:
                z = redshift
            else:
                raise ValueError(f"Redshift {redshift} not understood.")

            no_use_bins = self._calculate_min_wav_band(
                min_snr_wav=min_snr_wav,
                only_snr_instrument=band_instrument,
                redshift=z,
            )
            print("No use bins:", no_use_bins)
            count = 1
            for band in self.bands:
                if band in no_use_bins:
                    continue
                elif band.endswith("M") and use_only_widebands:
                    continue
                else:
                    count += 1
                    if ref_band.endswith("min"):
                        # Take signal and noise from lowest SNR for each pixel.
                        snr = signal / noise
                        snr_band = (
                            self.psf_matched_data[psf_type][band]
                            / self.psf_matched_rms_err[psf_type][band]
                        )
                        mask = snr_band < snr
                        signal[mask] = self.psf_matched_data[psf_type][band][
                            mask
                        ]
                        noise[mask] = self.psf_matched_rms_err[psf_type][band][
                            mask
                        ]
                    elif ref_band.endswith("sum"):
                        signal += self.psf_matched_data[psf_type][band]
                        noise += self.psf_matched_rms_err[psf_type][band] ** 2
                    elif ref_band.endswith("average"):
                        signal.append(self.psf_matched_data[psf_type][band])
                        noise.append(self.psf_matched_rms_err[psf_type][band])

            if ref_band.endswith("average"):
                signal = np.nanmean(signal, axis=0)
                noise = np.nanmean(noise, axis=0)

            if ref_band.endswith("sum"):
                noise = np.sqrt(noise)

        elif type(ref_band) is list:
            signal = np.zeros_like(
                self.psf_matched_data[psf_type][self.bands[-1]]
            )
            noise = np.zeros_like(
                self.psf_matched_rms_err[psf_type][self.bands[-1]]
            )

            for band in ref_band:
                signal += self.psf_matched_data[psf_type][band]
                noise += self.psf_matched_rms_err[psf_type][band]

            ref_band = "+".join(ref_band)

        else:
            signal = self.psf_matched_data[psf_type][ref_band]
            noise = self.psf_matched_rms_err[psf_type][ref_band]

        # replace any NaNs with 0
        np.nan_to_num(signal, copy=False)
        np.nan_to_num(noise, copy=False, nan=np.nanmedian(noise))

        if plot:
            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
            mappable = ax.imshow(
                signal / noise,
                origin="lower",
                interpolation="none",
                cmap="nipy_spectral_r",
            )
            fig.colorbar(mappable, ax=ax)
            ax.set_title(f"SNR Map (ref: {ref_band})")
            # Plot galaxy region
            ax.imshow(galaxy_region, origin="lower", alpha=0.5, cmap="Greys")
            plt.show()

        signal = signal[galaxy_region == 1].flatten()
        noise = noise[galaxy_region == 1].flatten()

        total_snr = np.nansum(signal) / np.sqrt(np.nansum(noise**2))

        print(f"Total SNR of input image: {total_snr}")

        assert (
            len(x) == len(signal) == len(noise) == len(y)
        ), f"Lengths of x, y, signal, noise not equal: {len(x)}, {len(y)}, {len(signal)}, {len(noise)}"

        print(f"Number of pixels: {len(x)}")

        if total_snr < SNR_reqs:
            print(
                f"WARNING! Total SNR of input image within mask: {total_snr} is less than required SNR: {SNR_reqs}"
            )
            # make all pixels a bin
            bin_number = np.ones_like(signal)

        elif np.all(signal / noise > SNR_reqs):
            print(f"All pixels have SNR > {SNR_reqs}, no binning required.")
            # Make each pixel a bin
            # Assign bin numbers from highest SNR to lowest SNR
            bin_number = np.argsort(signal / noise)[::-1] + 1
            print(bin_number)
        else:
            # sn_func = lambda index, flux, flux_err: print(index) #flux[index] / flux_err[index]
            try:
                bin_number, x_gen, y_gen, x_bar, y_bar, sn, nPixels, scale = (
                    voronoi_2d_binning(
                        x,
                        y,
                        signal,
                        noise,
                        SNR_reqs,
                        cvt=cvt,
                        wvt=wvt,
                        pixelsize=1,  # self.im_pixel_scales[ref_band].to(u.arcsec).value,
                        plot=plot,
                        quiet=quiet,
                    )
                )  # sn_func = sn_func)
                # move from 0 to n-1 bins to 1 to n bins
                bin_number += 1
            except (ValueError, IndexError):
                print(
                    "Voronoi binning failed. Using all pixels as a single bin."
                )
                bin_number = np.ones_like(signal)

        print(f"Number of bins: {np.max(bin_number)}")
        # Reshape bin_number to 2D given the galaxy region mask
        bin_number_2d = np.zeros_like(galaxy_region)
        # check it is 2D
        assert (
            np.ndim(bin_number_2d) == 2
        ), f"Bin number map is not 2D: {np.shape(bin_number_2d)}"
        # bin_number_2d[galaxy_region != 1] =
        for i, (xi, yi, bin) in enumerate(zip(x, y, bin_number)):
            bin_number_2d[yi, xi] = int(bin)

        self.maps.append("voronoi")

        meta_dict = {
            "ref_band": ref_band,
            "SNR_reqs": SNR_reqs,
            "psf_type": psf_type,
            "wvt": wvt,
            "cvt": cvt,
            "use_only_widebands": use_only_widebands,
            "min_band": min_band,
            "band_instrument": band_instrument,
            "min_snr_wav": min_snr_wav,
            "no_use_bins": no_use_bins,
        }
        self.add_to_h5(
            bin_number_2d,
            "bin_maps",
            "voronoi",
            setattr_gal="voronoi_map",
            meta=meta_dict,
            overwrite=overwrite,
        )

        if plot:
            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
            mappable = ax.imshow(
                bin_number_2d,
                origin="lower",
                interpolation="none",
                cmap="nipy_spectral_r",
            )
            fig.colorbar(mappable, ax=ax)
            ax.set_title(f"Voronoi Map (SNR > {SNR_reqs}, ref: {ref_band})")
            plt.show()

    def _calculate_min_wav_band(
        self, redshift, min_snr_wav=1216 * u.AA, only_snr_instrument=None
    ):
        self.get_filter_wavs()
        min_snr_wav_obs = min_snr_wav * (1 + redshift)
        delta_wav = 1e10 * u.AA
        no_SNR_requirement = []
        for band in self.bands:
            instrument = self.filter_instruments[band]
            if (
                only_snr_instrument is not None
                and instrument != only_snr_instrument
            ):
                # print(f'Skipping {band} as it is not in {only_snr_instrument}')
                no_SNR_requirement.append(band)

            if self.filter_ranges[band][1] < min_snr_wav_obs:
                no_SNR_requirement.append(band)
            """
            if self.filter_ranges[band][1] > min_snr_wav_obs and self.filter_ranges[band][1] - min_snr_wav_obs < delta_wav:
                min_band = band
                delta_wav = self.filter_ranges[band][1] - min_snr_wav_obs 
            """
        return no_SNR_requirement

    def pixedfit_binning(
        self,
        SNR_reqs=7,
        ref_band="F277W",
        min_band="auto",
        Dmin_bin=7,
        redc_chi2_limit=9.0,
        del_r=2.0,
        overwrite=False,
        min_snr_wav=1216 * u.AA,
        only_snr_instrument="NIRCam",
        save_out=True,
        remove_files=True,
        redshift="self",
        animate=False,
        name_out="pixedfit",  # Allows you to override the name of the output maps
        animation_save_path=None,
    ):
        """
        : SNR_reqs: list of SNR requirements for each band
        : ref_band: reference band for pixel binning
        : Dmin_bin: minimum diameter between pixels in binning (should be ~ FWHM of PSF)
        """
        from piXedfit.piXedfit_bin import pixel_binning

        if not hasattr(self, "img_process"):
            raise ValueError(
                "No image processing done. Run pixedfit_processing() first"
            )

        if (
            getattr(self, f"{name_out}_map", None) is not None
            and not overwrite
        ):
            print(
                f"{name_out} map already exists. Set overwrite=True to re-run"
            )
            return

        # ref_band_pos = np.argwhere(np.array([band == ref_band for band in self.bands])).flatten()[0]

        # Should calculate SNR requirements intelligently based on redshift
        if min_band is not None:
            if min_band == "auto":
                if redshift == "self":
                    redshift = self.redshift
                elif type(redshift) in [int, float]:
                    redshift = redshift
                else:
                    raise ValueError(
                        f"Redshift must be a float or int, not {type(redshift)}"
                    )

                no_SNR_requirement = self._calculate_min_wav_band(
                    min_snr_wav=min_snr_wav,
                    only_snr_instrument=only_snr_instrument,
                    redshift=redshift,
                )

        name_out_fits = f"{self.dir_images}/{self.survey}_{self.galaxy_id}_{name_out}_binned.fits"

        header = fits.open(self.flux_map_path)[0].header
        header_order = [
            header[f"FIL{pos}"].split("_")[-1].upper()
            for pos in range(0, len(self.bands))
        ]
        ref_band_pos = header_order.index(ref_band)

        SNR_reqs = {
            band: SNR_reqs
            for band in self.bands
            if band not in no_SNR_requirement
        }
        for band in no_SNR_requirement:
            SNR_reqs[band] = 0

        SNR = [SNR_reqs[band] for band in header_order]
        """
        for pos, band in enumerate(header_order):
            print(band, SNR_reqs[band])
            if pos == ref_band_pos:
                print('Reference band.')
        """

        out = pixel_binning(
            self.flux_map_path,
            ref_band=ref_band_pos,
            Dmin_bin=Dmin_bin,
            SNR=SNR,
            redc_chi2_limit=redc_chi2_limit,
            del_r=del_r,
            name_out_fits=name_out_fits,
            animate=animate,
            save_path=animation_save_path,
        )

        if not out:
            print("Pixedfit binning failed")
            return False

        meta_dict = {
            "name_out": name_out,
            "min_snr_wav": min_snr_wav,
            "only_snr_instrument": only_snr_instrument,
            "ref_band": ref_band,
            "SNR_reqs": SNR_reqs,
            "Dmin_bin": Dmin_bin,
            "redc_chi2_limit": redc_chi2_limit,
            "del_r": del_r,
        }

        if save_out:
            if name_out not in self.maps:
                self.maps.append(name_out)

            self.pixedfit_binmap_path = name_out_fits
            self.add_to_h5(
                name_out_fits,
                "bin_maps",
                name_out,
                ext="BIN_MAP",
                setattr_gal=f"{name_out}_map",
                overwrite=overwrite,
                meta=meta_dict,
            )
            self.add_to_h5(
                name_out_fits,
                "bin_fluxes",
                name_out,
                ext="BIN_FLUX",
                setattr_gal=f"{name_out}_binned_flux_map",
                overwrite=overwrite,
            )
            self.add_to_h5(
                name_out_fits,
                "bin_flux_err",
                name_out,
                ext="BIN_FLUXERR",
                setattr_gal=f"{name_out}_binned_flux_err_map",
                overwrite=overwrite,
            )

        if remove_files:
            os.remove(self.flux_map_path)
            os.remove(name_out_fits)

        return name_out_fits

    def plot_kron_ellipse(
        self, ax, center, band="detection", color="red", return_params=False
    ):
        if band == "detection":
            if self.total_photometry is None:
                if return_params:
                    return 0, 0, 0
                else:
                    return False
            # kron = self.total_photometry[self.detection_band][f'KRON_RADIUS_{self.detection_band}']
            a = self.total_photometry[self.detection_band][
                f"a_{self.detection_band}"
            ]  # already scaled by kron
            b = self.total_photometry[self.detection_band][
                f"b_{self.detection_band}"
            ]  # already scaled by kron
            theta = self.total_photometry[self.detection_band][
                f"theta_{self.detection_band}"
            ]
        else:
            kron_radius = self.auto_photometry[band]["KRON_RADIUS"]
            a = self.auto_photometry[band]["A_IMAGE"] * kron_radius
            b = self.auto_photometry[band]["B_IMAGE"] * kron_radius
            theta = self.auto_photometry[band]["THETA_IMAGE"]

        if return_params:
            return a, b, theta

        # center = np.shape(data)[0]/2
        e = Ellipse(
            (center, center),
            a,
            b,
            angle=theta,
            edgecolor=color,
            facecolor="none",
        )

        ax.add_artist(e)
        e.set_clip_box(ax.bbox)
        e.set_alpha(0.5)

    def plot_image_stamp(
        self,
        band,
        scale="log10",
        save=False,
        save_path=None,
        show=False,
        facecolor="white",
        sex_factor=6,
    ):
        fig, ax = plt.subplots(1, 1, figsize=(6, 6), facecolor=facecolor)
        if scale == "log10":
            data = np.log10(self.phot_imgs[band])
        elif scale == "linear":
            data = self.phot_imgs[band]
        else:
            raise ValueError("Scale must be log10 or linear")
        ax.imshow(data, origin="lower", interpolation="none")
        if band in self.auto_photometry.keys():
            self.plot_kron_ellipse(ax, center=self.cutout_size / 2, band=band)

        re = 15  # pixels
        d_A = cosmo.angular_diameter_distance(self.redshift)
        pix_scal = u.pixel_scale(0.03 * u.arcsec / u.pixel)
        re_as = (re * u.pixel).to(u.arcsec, pix_scal)
        re_kpc = (re_as * d_A).to(u.kpc, u.dimensionless_angles())

        # First scalebar
        scalebar = AnchoredSizeBar(
            ax.transData,
            0.5 / self.im_pixel_scales[band].value,
            '0.5"',
            "lower right",
            pad=0.3,
            color="black",
            frameon=False,
            size_vertical=1,
            fontproperties=FontProperties(size=18),
        )
        ax.add_artist(scalebar)
        # Plot scalebar with physical size
        scalebar = AnchoredSizeBar(
            ax.transData,
            re,
            f"{re_kpc:.1f}",
            "upper left",
            pad=0.3,
            color="black",
            frameon=False,
            size_vertical=1,
            fontproperties=FontProperties(size=18),
        )
        scalebar.set(
            path_effects=[
                PathEffects.withStroke(linewidth=3, foreground="white")
            ]
        )
        ax.add_artist(scalebar)

        # Add scalebar

        ax.set_title(f"{band} Image")
        if save:
            plt.savefig(save_path)
        if show:
            plt.show()

    def plot_image_stamps(self, show=False):
        nrows = len(self.bands) // 6 + 1
        fig, axes = plt.subplots(nrows, 6, figsize=(24, 4 * nrows))
        axes = axes.flatten()
        # Remove empty axes
        for i in range(len(self.bands), len(axes)):
            fig.delaxes(axes[i])

        for i, band in enumerate(self.bands):
            axes[i].imshow(
                np.log10(self.phot_imgs[band]),
                origin="lower",
                interpolation="none",
            )
            axes[i].set_title(f"{band} Image")
        plt.subplots_adjust(
            left=0.05,
            right=0.95,
            bottom=0.05,
            top=0.95,
            hspace=0.1,
            wspace=0.15,
        )
        # return fig
        if show:
            plt.show()
        else:
            return fig

    def plot_gal_region(
        self, bin_type="pixedfit", facecolor="white", show=False
    ):
        if self.gal_region is None:
            raise ValueError(
                "No gal_region region found. Run pixedfit_processing() first"
            )
        else:
            if bin_type not in self.gal_region.keys():
                raise ValueError(
                    f"gal_region not found for {bin_type}. Run pixedfit_processing() first"
                )
        gal_region = self.gal_region[bin_type]
        nrows = len(self.bands) // 6 + 1
        fig, axes = plt.subplots(
            nrows, 6, figsize=(18, 4 * nrows), facecolor=facecolor
        )
        axes = axes.flatten()
        # Remove empty axes
        for i in range(len(self.bands), len(axes)):
            fig.delaxes(axes[i])

        for i, band in enumerate(self.bands):
            # rows, cols = np.where(gal_region==0)
            # gal_region[rows,cols] = float('nan')
            axes[i].imshow(
                np.log10(self.phot_imgs[band]),
                origin="lower",
                interpolation="none",
            )
            axes[i].set_title(f"{band} Image")
            axes[i].imshow(
                gal_region,
                origin="lower",
                interpolation="none",
                alpha=0.5,
                cmap="copper",
            )
        plt.subplots_adjust(
            left=0.05,
            right=0.95,
            bottom=0.05,
            top=0.95,
            hspace=0.1,
            wspace=0.15,
        )
        if show:
            plt.show()
        else:
            return fig
        # return fig

    def add_detection_data(
        self,
        detection_instrument="NIRCam",
        galfind_work_dir="/raid/scratch/work/austind/GALFIND_WORK",
        overwrite=False,
    ):
        if self.det_data is not None and not overwrite:
            print(f"Detection data already loaded for {self.galaxy_id}.")
            return

        im_path = f"{galfind_work_dir}/Stacked_Images/{self.galfind_version}/{detection_instrument}/{self.survey}/{self.survey}_{self.detection_band}_{self.galfind_version}_stack_new.fits"
        seg_path = f"{galfind_work_dir}/SExtractor/{detection_instrument}/{self.galfind_version}/{self.survey}/{self.survey}_{self.detection_band}_{self.detection_band}_sel_cat_{self.galfind_version}_seg.fits"
        hdu_data = fits.open(im_path)
        hdu_seg = fits.open(seg_path)
        det_data = {}

        # print(im_path)
        # print(seg_path)
        output_flux_unit = self.phot_pix_unit["F444W"]
        hd = fits.getheader(self.im_paths["F444W"], ext=1)
        wcs_test = WCS(hd)
        # print(self.sky_coord)
        pix_scale = self.im_pixel_scales["F444W"].to(u.arcsec)
        for ext, hdu, name, flux_unit in zip(
            ["SCI", "ERR", 0],
            [hdu_data, hdu_data, hdu_seg],
            ["phot", "rms_err", "seg"],
            [output_flux_unit, output_flux_unit, None],
        ):
            data = hdu[ext].data
            header = hdu[ext].header
            zeropoint = (
                header["ZEROPNT"] if "ZEROPNT" in header.keys() else None
            )
            unit = header["BUNIT"] if "BUNIT" in header.keys() else None
            try:
                unit = u.Unit(unit)
            except ValueError:
                unit = None

            wcs = WCS(header)
            # print(wcs.world_to_pixel(self.sky_coord))
            # print(wcs_test.world_to_pixel(self.sky_coord))
            skycoord = SkyCoord(
                self.meta_properties["ALPHA_J2000"] * u.deg,
                self.meta_properties["DELTA_J2000"] * u.deg,
                frame="icrs",
            )

            cutout = Cutout2D(
                data,
                position=self.sky_coord,
                size=(self.cutout_size, self.cutout_size),
                wcs=wcs,
            )

            data = cutout.data

            assert (
                np.shape(data) == (self.cutout_size, self.cutout_size)
            ), f"Cutout shape {np.shape(data)} not same as cutout size {self.cutout_size}"
            if flux_unit is not None:
                if unit or zeropoint:
                    if unit == u.Unit("MJy/sr"):
                        if output_flux_unit == u.Unit("MJy/sr"):
                            data = data * u.MJy / u.sr

                        else:
                            data = data * unit * pix_scale**2

                            if output_flux_unit in [u.Jy, u.mJy, u.uJy, u.nJy]:
                                data = data.to(output_flux_unit)
                                unit = output_flux_unit
                            elif (
                                output_flux_unit
                                == u.erg / u.s / u.cm**2 / u.AA
                            ):
                                data = data.to(
                                    output_flux_unit,
                                    equivalencies=u.spectral_density(
                                        wave[band]
                                    ),
                                )
                                unit = output_flux_unit
                            else:
                                raise Exception(
                                    "Output flux unit not recognised"
                                )
                    elif zeropoint is not None:
                        outzp = output_flux_unit.to(u.ABmag)
                        data = data * 10 ** ((outzp - zeropoint) / 2.5)

                    else:
                        data = data * unit
                        data = data.to(output_flux_unit)

            self.add_to_h5(data, "det_data", f"{name}", overwrite=overwrite)
            det_data[name] = data

        hdu_data.close()
        hdu_seg.close()
        self.det_data = det_data

    def add_original_data(
        self,
        instruments=["ACS_WFC", "NIRCam"],
        excl_bands=[],
        aper_diams=[0.32] * u.arcsec,
        templates_arr=["fsps_larson"],
        overwrite=False,
        lowz_zmax_arr=[[4.0, 6.0, None]],
        cat=None,
        return_cat=False,
        crop_by="ID",
    ):
        if (
            self.unmatched_data is not None
            and self.unmatched_rms_err is not None
            and self.unmatched_seg is not None
            and not overwrite
        ):
            print(f"Unmatched data already loaded for {self.galaxy_id}.")
            return

        if crop_by == "ID":
            crop_by = f"ID={int(self.galaxy_id)}"

        if cat is None:
            from galfind import EAZY, Catalogue
            from galfind.Catalogue_Creator import GALFIND_Catalogue_Creator

            SED_code_arr = [EAZY()]
            SED_fit_params_arr = make_EAZY_SED_fit_params_arr(
                SED_code_arr, templates_arr, lowz_zmax_arr
            )
            # Make cat creator
            cat_creator = GALFIND_Catalogue_Creator(
                "loc_depth", aper_diams[0], 10
            )
            # Load catalogue and populate galaxies
            cat = Catalogue.from_pipeline(
                survey=self.survey.replace("_psfmatched", ""),
                version=self.galfind_version,
                instruments=instruments,
                aper_diams=aper_diams,
                cat_creator=cat_creator,
                SED_fit_params_arr=SED_fit_params_arr,
                forced_phot_band=self.detection_band.split("+"),
                excl_bands=excl_bands,
                loc_depth_min_flux_pc_errs=[10],
                crop_by=crop_by,
            )
            # Make cutouts - this may not work currently as data.wht_types doesn't appear to be defined.

        # print(len(cat))
        # print([gal.ID for gal in cat.gals])

        cat.make_cutouts(
            [int(self.galaxy_id)],
            cutout_size=self.cutout_size
            * self.im_pixel_scales["F444W"].to(u.arcsec),
        )

        # Obtain galaxy object
        galaxy = [
            gal for gal in cat.gals if str(gal.ID) == str(self.galaxy_id)
        ]
        if len(galaxy) == 0:
            raise ValueError(f"Galaxy with ID {self.galaxy_id} not found")

        assert (
            len(galaxy) == 1
        ), f"{len(galaxy)} galaxies found with ID {self.galaxy_id}"
        galaxy = galaxy[0]

        cutout_paths = galaxy.cutout_paths

        bands = (
            galaxy.phot.instrument.band_names
        )  # should be bands just for galaxy!
        galaxy_skycoord = galaxy.sky_coord
        # Some checks
        assert (
            [
                np.round(galaxy_skycoord.ra.degree, 4),
                np.round(galaxy_skycoord.dec.degree, 4),
            ]
            == [
                np.round(self.sky_coord.ra.degree, 4),
                np.round(self.sky_coord.dec.degree, 4),
            ]
        ), f"Galaxy skycoord {galaxy_skycoord} does not match input skycoord {self.sky_coord}"

        bands_mask = galaxy.phot.flux_Jy.mask
        bands = bands[~bands_mask]

        # bands =

        assert (
            len(bands) >= len(self.bands)
        ), f"Bands {bands} ({len(bands)} do not match input bands {self.bands} ({len(self.bands)})"

        phot_imgs = {}
        phot_pix_unit = {}
        rms_err_imgs = {}
        seg_imgs = {}
        phot_img_headers = {}

        hfile = h5.File(self.h5_path, "a")
        if "unmatched_data" not in hfile.keys():
            hfile.create_group("unmatched_data")
        output_flux_unit = self.phot_pix_unit["F444W"]

        for band in bands:
            cutout_path = cutout_paths[band]
            hdu = fits.open(cutout_path)
            assert (
                hdu["SCI"].header["NAXIS1"]
                == hdu["SCI"].header["NAXIS2"]
                == self.cutout_size
            )  # Check cutout size
            data = hdu["SCI"].data
            try:
                rms_data = hdu["RMS_ERR"].data
            except KeyError:
                weight_data = hdu["WHT"].data
                rms_data = np.where(
                    weight_data == 0, 0, 1 / np.sqrt(weight_data)
                )

            unit = (
                hdu["SCI"].header["BUNIT"]
                if "BUNIT" in hdu["SCI"].header.keys()
                else None
            )
            zeropoint = (
                hdu["SCI"].header["ZEROPNT"]
                if "ZEROPNT" in hdu["SCI"].header.keys()
                else None
            )
            if unit:
                try:
                    unit = u.Unit(unit)
                except ValueError:
                    unit = None

            pix_scale = self.im_pixel_scales[band]
            # convert to flux_unit
            if unit == u.Unit("MJy/sr"):
                if output_flux_unit == u.Unit("MJy/sr"):
                    data = data * u.MJy / u.sr
                    rms_data = rms_data * u.MJy / u.sr
                    unit = u.MJy / u.sr
                else:
                    data = data * unit * pix_scale**2
                    rms_data = rms_data * unit * pix_scale**2

                    if output_flux_unit in [u.Jy, u.mJy, u.uJy, u.nJy]:
                        data = data.to(output_flux_unit)
                        rms_data = rms_data.to(output_flux_unit)
                        unit = output_flux_unit
                    elif output_flux_unit == u.erg / u.s / u.cm**2 / u.AA:
                        data = data.to(
                            output_flux_unit,
                            equivalencies=u.spectral_density(wave[band]),
                        )
                        rms_data = rms_data.to(
                            output_flux_unit,
                            equivalencies=u.spectral_density(wave[band]),
                        )
                        unit = output_flux_unit
                    else:
                        raise Exception("Output flux unit not recognised")
            elif zeropoint is not None:
                outzp = output_flux_unit.to(u.ABmag)
                data = data * 10 ** ((outzp - zeropoint) / 2.5)
                rms_data = rms_data * 10 ** ((outzp - zeropoint) / 2.5)
            else:
                data = data * unit
                rms_data = rms_data * unit
                rms_data = rms_data.to(output_flux_unit)
                data = data.to(output_flux_unit)

            phot_imgs[band] = copy.copy(data)
            phot_pix_unit[band] = unit
            rms_err_imgs[band] = copy.copy(rms_data)
            seg_imgs[band] = copy.copy(hdu["SEG"].data)
            # phot_img_headers[band] = str(hdu['SCI'].header)

            # Remove all references to the fits file so it can be closed

            if f"phot_{band}" in hfile["unmatched_data"].keys():
                hfile["unmatched_data"][f"phot_{band}"][()] = phot_imgs[band]
                hfile["unmatched_data"][f"rms_err_{band}"][()] = rms_err_imgs[
                    band
                ]
                hfile["unmatched_data"][f"seg_{band}"][()] = seg_imgs[band]
            else:
                hfile["unmatched_data"].create_dataset(
                    f"phot_{band}", data=phot_imgs[band]
                )
                hfile["unmatched_data"].create_dataset(
                    f"rms_err_{band}", data=rms_err_imgs[band]
                )
                hfile["unmatched_data"].create_dataset(
                    f"seg_{band}", data=seg_imgs[band]
                )

            hdu.close()
            del data, rms_data, hdu

        hfile.close()

        self.unmatched_data = phot_imgs
        self.unmatched_rms_err = rms_err_imgs
        self.unmatched_seg = seg_imgs

        if return_cat:
            return cat

        """
        for band in self.bands:
            if err_folder is not None and im_folder is not None:
                im_path = glob.glob(f'{im_folder}/*{band}*.fits')[0]
                err_path = glob.glob(f'{err_folder}/*{band}*.fits')[1]
            else:
                im_path = self.im_paths[band].replace(self.survey, self.survey.replace('_psfmatched', ''))
                err_path = self.rms_err_paths[band].replace(self.survey, self.survey.replace('_psfmatched', ''))

            # Get img
            hdu = fits.open(im_path)
            data = hdu[self.im_exts[band]].data
            header = hdu[self.im_exts[band]].header
            hdu.close()
            wcs = WCS(header)
            cutout = Cutout2D(fits.open(path)[self.im_exts[band]].data, self.sky_coord, size=self.cutout_size*u.pixel, wcs=wcs)
            cutout_data = cutout.data
            cutout_header = cutout.wcs.to_header()

            # Get err
            hdu = fits.open(err_path)
            err_data = hdu[self.rms_err_exts[band]].data
            err_header = hdu[self.rms_err_exts[band]].header
            hdu.close()
            wcs = WCS(err_header)
            cutout = Cutout2D(fits.open(path)[self.rms_err_exts[band]].data, self.sky_coord, size=self.cutout_size*u.pixel, wcs=wcs)
            cutout_err_data = cutout.data
            cutout_err_header = cutout.wcs.to_header()
        """

    def add_to_h5(
        self,
        original_data,
        group,
        name,
        ext=0,
        setattr_gal=None,
        overwrite=False,
        meta=None,
        setattr_gal_meta=None,
        force=False,
    ):
        if not self.save_out and not force:
            print(
                "Skipping writing to .h5 as save_out is False. Set force=True to do this anyway."
            )
            return

        if type(original_data) in [
            u.Quantity,
            u.Magnitude,
            Masked(u.Quantity),
        ]:
            data = original_data.value
        else:
            data = original_data

        if type(data) in [dict]:
            data = str(data)

        if type(data) is str:
            if data.endswith(".fits"):
                data = fits.open(data)[ext].data
                original_data = data

        if not os.path.exists(os.path.dirname(self.h5_path)):
            os.makedirs(os.path.dirname(self.h5_path))

        hfile = h5.File(self.h5_path, "a")
        if group not in hfile.keys():
            hfile.create_group(group)
        if name in hfile[group].keys():
            if overwrite:
                del hfile[group][name]
            else:
                print(
                    f"{name} already exists in {group} group and overwrite is set to False"
                )
                return

        if type(data) == np.ndarray:
            compression = "gzip" if data.nbytes > 1e6 else None
        else:
            compression = None
        hfile[group].create_dataset(name, data=data, compression=compression)
        if meta is not None:
            for key in meta.keys():
                sys.stdout.write(f"Setting meta, {key}, {str(meta[key])}")
                sys.stdout.flush()
                hfile[group][name].attrs[key] = str(meta[key])
            if setattr_gal_meta is not None:
                setattr(self, setattr_gal_meta, meta)

        sys.stdout.write(f"\r added to {hfile}, {group}, {name}")
        sys.stdout.flush()

        hfile.close()

        if setattr_gal is not None:
            print(f"Setting {setattr_gal} attribute.")
            setattr(self, setattr_gal, original_data)

    def plot_err_stamps(self):
        fig, axes = plt.subplots(
            1, len(self.bands), figsize=(4 * len(self.bands), 4)
        )
        for i, band in enumerate(self.bands):
            axes[i].imshow(
                np.log10(self.rms_err_imgs[band]),
                origin="lower",
                interpolation="none",
            )
            axes[i].set_title(f"{band} Error")

    def plot_seg_stamps(self, show_pixedfit=False):
        # Split ax over rows - max 6 per row
        nrows = len(self.bands) // 6 + 1
        fig, axes = plt.subplots(nrows, 6, figsize=(24, 4 * nrows))
        axes = axes.flatten()
        # Remove empty axes
        for i in range(len(self.bands), len(axes)):
            fig.delaxes(axes[i])

        for i, band in enumerate(self.bands):
            if show_pixedfit:
                mappable = axes[i].imshow(
                    self.img_process.segm_maps[i],
                    origin="lower",
                    interpolation="none",
                )

            mappable = axes[i].imshow(
                self.seg_imgs[band], origin="lower", interpolation="none"
            )
            fig.colorbar(mappable, ax=axes[i])
            axes[i].set_title(f"{band} Segmentation")
        return fig

    def pixedfit_plot_map_fluxes(self):
        from piXedfit.piXedfit_images import plot_maps_fluxes

        if not hasattr(self, "flux_map_path"):
            raise ValueError(
                "No flux map found. Run pixedfit_processing() first"
            )

        plot_maps_fluxes(self.flux_map_path, ncols=8, savefig=False)

    def pixedfit_plot_radial_SNR(self):
        from piXedfit.piXedfit_images import plot_SNR_radial_profile

        if not hasattr(self, "flux_map_path"):
            raise ValueError(
                "No flux map found. Run pixedfit_processing() first"
            )
        plot_SNR_radial_profile(self.flux_map_path, savefig=False)

    def pixedfit_plot_image_stamps(self):
        if not hasattr(self, "img_process"):
            raise Exception("Need to run pixedfit_processing first")
        fig = self.img_process.plot_image_stamps(savefig=False)
        return fig

    def pixedfit_plot_galaxy_region(self):
        if not hasattr(self, "gal_region"):
            raise Exception("Need to run pixedfit_processing first")
        self.img_process.plot_gal_region(self.gal_region, savefig=False)

    def pixedfit_plot_segm_maps(self):
        if not hasattr(self, "img_process"):
            raise Exception("Need to run pixedfit_processing first")
        self.img_process.plot_segm_maps(savefig=False)

    def pixedfit_plot_binmap(self):
        if not hasattr(self, "pixedfit_binmap_path"):
            raise Exception("Need to run pixedfit_binning first")
        from piXedfit.piXedfit_bin import plot_binmap

        plot_binmap(
            self.pixedfit_binmap_path, plot_binmap_spec=False, savefig=False
        )

    def pixel_by_pixel_galaxy_region(
        self,
        snr_req=5,
        band_req="F444W",
        region_name="auto",
        overwrite=False,
        override_psf_type=None,
        mask=None,
    ):
        # Make a boolean galaxy region based on pixels in band_req that have SNR > snr_req

        # Use PSF matched data
        if not hasattr(self, "psf_matched_data"):
            raise ValueError("No PSF matched data found.")

        if not hasattr(self, "psf_matched_rms_err"):
            raise ValueError("No PSF matched rms_err found.")

        if hasattr(self, "use_psf_type") and override_psf_type is None:
            psf_type = self.use_psf_type
        galaxy_region = np.zeros_like(
            self.psf_matched_data[psf_type][band_req]
        )

        snr_map = (
            self.psf_matched_data[psf_type][band_req]
            / self.psf_matched_rms_err[psf_type][band_req]
        )
        galaxy_region[snr_map > snr_req] = 1

        if mask is not None:
            assert np.shape(mask) == np.shape(galaxy_region)
            galaxy_region[mask == 0] = 0

        if region_name == "auto":
            region_name = f"SNR_{snr_req}_{band_req}"

        self.add_to_h5(
            galaxy_region, "galaxy_region", region_name, overwrite=overwrite
        )

        if not hasattr(self, "gal_region"):
            self.gal_region = {}

        self.gal_region[region_name] = galaxy_region

    def pixel_by_pixel_binmap(self, galaxy_region=None, overwrite=False):
        if galaxy_region is None:
            if not hasattr(self, "gal_region"):
                raise ValueError(
                    "No galaxy region found. Run pixedfit_processing() first"
                )
            if len(self.gal_region.keys()) == 1:
                galaxy_region = self.gal_region[
                    list(self.gal_region.keys())[0]
                ]

        elif type(galaxy_region) is str:
            galaxy_region = self.gal_region[galaxy_region]

        # Make a binmap where each pixel wihin the galaxy region is a seperate bin

        # Count number of pixels in galaxy region
        number_of_bins = np.sum(galaxy_region)
        binmap = np.zeros_like(galaxy_region)

        # Loop over each pixel in the galaxy region and assign a bin number
        bin_number = 1

        for i in range(np.shape(galaxy_region)[0]):
            for j in range(np.shape(galaxy_region)[1]):
                if galaxy_region[i, j] == 1:
                    binmap[i, j] = bin_number
                    bin_number += 1

        self.add_to_h5(
            binmap,
            "bin_maps",
            "pixel_by_pixel",
            ext="BIN_MAP",
            setattr_gal="pixel_by_pixel_map",
            overwrite=overwrite,
        )

    def measure_flux_in_bins(
        self, override_psf_type=None, binmap_type="pixedfit", overwrite=False
    ):
        if hasattr(self, "use_psf_type") and override_psf_type is None:
            psf_type = self.use_psf_type
        else:
            psf_type = override_psf_type

        if not hasattr(self, f"{binmap_type}_map"):
            raise Exception(f"Need to run {binmap_type}_binning first")
        # Sum fluxes in each bins and produce a table

        if (
            hasattr(self, "photometry_table")
            and self.photometry_table is not None
        ):
            if (
                psf_type in self.photometry_table.keys()
                and binmap_type in self.photometry_table[psf_type].keys()
                and not overwrite
            ):
                print(
                    f"Photometry table already exists for {psf_type} and {binmap_type}"
                )
                return self.photometry_table[psf_type][binmap_type]

        binmap = getattr(self, f"{binmap_type}_map")
        table = QTable()
        table["ID"] = [str(i) for i in range(1, int(np.max(binmap)) + 1)]
        table["type"] = "bin"
        for pos, band in enumerate(self.bands):
            fluxes = []
            flux_errs = []
            for i in range(1, int(np.max(binmap)) + 1):
                flux = np.sum(
                    self.psf_matched_data[psf_type][band][binmap == i]
                )
                flux_err = np.sqrt(
                    np.sum(
                        self.psf_matched_rms_err[psf_type][band][binmap == i]
                        ** 2
                    )
                )
                if type(flux) in [u.Quantity, u.Magnitude, Masked(u.Quantity)]:
                    assert flux.unit == self.phot_pix_unit[band]
                    flux = flux.value

                if type(flux_err) in [
                    u.Quantity,
                    u.Magnitude,
                    Masked(u.Quantity),
                ]:
                    assert flux_err.unit == self.phot_pix_unit[band]
                    flux_err = flux_err.value

                fluxes.append(flux)
                flux_errs.append(flux_err)

            table[band] = u.Quantity(
                np.array(fluxes), unit=self.phot_pix_unit[band]
            )
            table[f"{band}_err"] = u.Quantity(
                np.array(flux_errs), unit=self.phot_pix_unit[band]
            )

        # Add sum of all rows
        row = ["TOTAL_BIN", "TOTAL_BIN"]
        for pos, band in enumerate(self.bands):
            row.append(np.sum(table[band]))
            row.append(np.sqrt(np.sum(table[f"{band}_err"] ** 2)))
        table.add_row(row)

        if self.auto_photometry not in [None, {}]:
            # Add MAG_AUTO and MAGERR_AUTO
            for i in ["MAG_AUTO"]:
                row = [i, i]
                self.get_filter_wavs()
                for pos, band in enumerate(self.bands):
                    try:
                        mag = self.auto_photometry[band][i] * u.ABmag
                        try:
                            mag_err = self.auto_photometry[band][
                                i.replace("_", "ERR_")
                            ]
                        except:
                            mag_err = 0.1
                            print(
                                f"No {i} error found for {band}, setting to 0.1 mag"
                            )
                        flux = mag.to(
                            u.uJy,
                            equivalencies=u.spectral_density(
                                self.filter_wavs[band]
                            ),
                        )
                        flux_err = 0.4 * np.log(10) * flux * mag_err

                    except (TypeError, KeyError) as e:
                        print(e)
                        print(
                            f"WARNING! No {i} found for {band}, falling back to aperture photometry! Probably no detection. "
                        )
                        flux = (
                            self.aperture_dict[str(0.32 * u.arcsec)]["flux"][
                                pos
                            ]
                            * u.Jy
                        )
                        flux_err = (
                            self.aperture_dict[str(0.32 * u.arcsec)][
                                "flux_err"
                            ][pos]
                            * u.Jy
                        )

                    flux = flux.to(u.uJy)
                    flux_err = flux_err.to(u.uJy)
                    row.append(flux)
                    row.append(flux_err)
                table.add_row(row)
        # Add MAG_APER
        for aper in self.aperture_dict.keys():
            if type(aper) is u.Quantity:
                aper = aper.value
            row = [f"MAG_APER_{aper}", f"MAG_APER_{aper}"]
            for pos, band in enumerate(self.bands):
                flux = self.aperture_dict[aper]["flux"][pos]
                flux_err = self.aperture_dict[aper]["flux_err"][pos]
                if type(flux) == Masked(u.Quantity):
                    flux = flux.unmasked

                if type(flux_err) == Masked(u.Quantity):
                    flux_err = flux_err.unmasked

                if type(flux) not in [
                    u.Quantity,
                    u.Magnitude,
                    Masked(u.Quantity),
                ]:
                    flux *= u.Jy
                if type(flux_err) not in [
                    u.Quantity,
                    u.Magnitude,
                    Masked(u.Quantity),
                ]:
                    flux_err *= u.Jy

                flux = flux.to(u.uJy)
                flux_err = flux_err.to(u.uJy)
                row.append(flux)
                row.append(flux_err)
            table.add_row(row)

        # Add total aperture fluxes
        if getattr(self, "total_photometry", None) is not None:
            print("Adding total aperture fluxes.")
            row = ["MAG_APER_TOTAL", "MAG_APER_TOTAL"]

            for pos, band in enumerate(self.bands):
                flux = self.total_photometry[band]["flux"]
                flux_err = self.total_photometry[band]["flux_err"]
                if type(flux) not in [
                    u.Quantity,
                    u.Magnitude,
                    Masked(u.Quantity),
                ]:
                    flux_unit = u.Unit(
                        self.total_photometry[band]["flux_unit"]
                    )
                    flux *= flux_unit
                    flux_err *= flux_unit
                row.append(flux)
                row.append(flux_err)

            table.add_row(row)

        if (
            not hasattr(self, "photometry_table")
            or self.photometry_table is None
        ):
            self.photometry_table = {psf_type: {binmap_type: table}}
        elif psf_type not in self.photometry_table.keys():
            self.photometry_table[psf_type] = {binmap_type: table}
        else:
            self.photometry_table[psf_type][binmap_type] = table

        # print(table)
        # Write table to our existing h5 file
        write_table_hdf5(
            table,
            self.h5_path,
            f"binned_photometry_table/{psf_type}/{binmap_type}",
            serialize_meta=True,
            overwrite=True,
            append=True,
        )

        return table

    def provide_bagpipes_phot(
        self, gal_id, return_bands=False, exclude_bands=[]
    ):
        """Provide the fluxes in the correct format for bagpipes"""
        gal_id = str(gal_id)
        if not hasattr(self, "photometry_table"):
            raise Exception("Need to run measure_flux_in_bins first")
        if hasattr(self, "use_psf_type"):
            psf_type = self.use_psf_type
        else:
            psf_type = "webbpsf"

        if hasattr(self, "use_binmap_type"):
            binmap_type = self.use_binmap_type
        else:
            binmap_type = "pixedfit"

        if hasattr(self, "sed_min_percentage_err"):
            min_percentage_error = self.sed_min_percentage_err
        else:
            min_percentage_error = 5

        if gal_id == 1:
            print(
                f"Using {psf_type} PSF and {binmap_type} binning derived fluxes"
            )
        flux_table = self.photometry_table[psf_type][binmap_type]

        if flux_table[flux_table.colnames[1]].unit != u.uJy:
            for col in flux_table.colnames[2:]:
                flux_table[col] = flux_table[col].to(u.uJy)

        bands = [band for band in self.bands if band not in exclude_bands]

        for band in bands:
            flux_col_name = band
            fluxerr_col_name = f"{band}_err"
            # Where the error is less than 10% of the flux, set the error to 10% of the flux, if the flux is greater than 0
            mask = (
                flux_table[fluxerr_col_name] / flux_table[flux_col_name]
                < min_percentage_error / 100
            ) & (flux_table[flux_col_name] > 0)
            flux_table[fluxerr_col_name][mask] = (
                min_percentage_error / 100 * flux_table[flux_col_name][mask]
            )

        row = flux_table[flux_table["ID"] == gal_id]

        if len(row) == 0:
            raise Exception(f"ID {gal_id} not found in flux table")
        elif len(row) > 1:
            raise Exception(f"More than one ID {gal_id} found in flux table")

        row = Table(row)

        order = list(
            np.ndarray.flatten(
                np.array(
                    [
                        [f"{band}", f"{band}_err"]
                        for pos, band in enumerate(bands)
                    ]
                )
            )
        )

        table_order = row[order]

        flux, err = [], []
        for pos, item in enumerate(table_order[0]):
            if pos % 2 == 0:
                flux.append(item)
            else:
                err.append(item)
        final = np.vstack((np.array(flux), np.array(err))).T

        # Loop over rows and check for any where flux and error are 0
        mask = ~np.all(final == 0, axis=1)
        final = final[mask]

        if return_bands:
            return np.array(self.bands)[mask]

        return final

    def plot_photometry_bins(
        self,
        binmap_type="pixedfit",
        bins_to_show="all",
        fig=None,
        ax=None,
        wav_unit=u.um,
        flux_unit=u.uJy,
        label_individual=False,
        save=False,
        save_path=None,
        min_flux=None,
        show=True,
        legend=True,
        facecolor="white",
        cmap_bins="nipy_spectral_r",
    ):
        self.get_filter_wavs()
        # from galfind import Filter

        if min_flux is not None:
            min_flux = min_flux.to(flux_unit).value

        # Get photometry table
        if not hasattr(self, "photometry_table"):
            raise Exception("Need to run measure_flux_in_bins first")
        if hasattr(self, "use_psf_type"):
            psf_type = self.use_psf_type
        else:
            psf_type = "webbpsf"

        table = self.photometry_table[psf_type][binmap_type]
        if fig is None:
            fig = plt.figure(
                figsize=(8, 6), facecolor=facecolor, constrained_layout=True
            )
        if ax is None:
            ax = fig.add_subplot(111)

        if bins_to_show == "all":
            bins_to_show = np.unique(table["ID"])

        # Set up color map
        map = getattr(self, f"{binmap_type}_map")
        cmap = plt.get_cmap(cmap_bins)
        norm = Normalize(vmin=np.nanmin(map), vmax=np.nanmax(map))
        colors = {i: cmap(norm(i)) for i in np.unique(map)}

        colorss = []
        for i, rbin in enumerate(bins_to_show):
            mask = table["ID"] == rbin

            name = table["type"][mask]
            if len(name) == 0:
                raise Exception(f"Bin {rbin} not found in table")
            name = name[0]
            if name == "TOTAL_BIN":
                color = "black"
            if name == "MAG_AUTO":
                color = "blue"
            if name == "MAG_ISO":
                color = "green"
            if name == "MAG_BEST":
                color = "orange"
            if name == "MAG_APER_TOTAL":
                color = "violet"
            if name.startswith("MAG_APER"):
                color = "purple"
            if name == "bin":
                color = colors[int(rbin)]

            for j, band in enumerate(self.bands):
                flux = table[mask][band]
                flux_err = table[mask][f"{band}_err"]

                wav = self.filter_wavs[band]
                # print(wav, flux, flux_err)
                if flux_unit == u.ABmag:
                    fnu_jy = flux.to(
                        u.uJy, equivalencies=u.spectral_density(wav)
                    )
                    fnu_jy_err = flux_err.to(
                        u.uJy, equivalencies=u.spectral_density(wav)
                    )
                    inner = fnu_jy / (fnu_jy - fnu_jy_err)
                    err_up = 2.5 * np.log10(inner)
                    err_low = 2.5 * np.log10(1 + (fnu_jy_err / fnu_jy))
                    err_low = (
                        err_low.value
                        if type(err_low) is u.Quantity
                        else err_low
                    )
                    err_up = (
                        err_up.value if type(err_up) is u.Quantity else err_up
                    )

                    yerr = [
                        np.atleast_1d(np.abs(err_low)),
                        np.atleast_1d(np.abs(err_up)),
                    ]
                    flux = flux.to(
                        flux_unit, equivalencies=u.spectral_density(wav)
                    ).value
                else:
                    yerr = flux_err.to(
                        flux_unit, equivalencies=u.spectral_density(wav)
                    ).value
                    flux = flux.to(
                        flux_unit, equivalencies=u.spectral_density(wav)
                    ).value

                if label_individual:
                    if name == "bin":
                        label = f"bin {rbin}"
                    else:
                        label = f"{rbin}"
                else:
                    label = name

                label = label if color not in colorss else ""

                # If error low or high is nan, plot point as upper limit

                alpha = 1
                ec = "black"

                if np.isnan(yerr[0]) or np.isnan(yerr[1]):
                    dy = 0.20
                    # calculate dy as 5% of the plot range
                    flux = flux.item()
                    if min_flux is not None:
                        if flux_unit == u.ABmag:
                            condition = flux + dy > min_flux
                        else:
                            condition = flux - dy < min_flux

                        if condition or np.isnan(flux):
                            flux = (
                                min_flux - dy
                                if flux_unit == u.ABmag
                                else min_flux + dy
                            )
                            alpha = 0.5
                            ec = "mediumslateblue"

                    patch = FancyArrow(
                        x=wav.to(wav_unit).value,
                        y=flux,
                        dx=0,
                        dy=dy,
                        width=0.02,
                        fc=color,
                        ec=ec,
                        transform=ax.transData,
                        alpha=alpha,
                        length_includes_head=True,
                    )
                    ax.add_patch(patch)
                else:
                    ax.errorbar(
                        wav.to(wav_unit),
                        flux,
                        yerr=yerr,
                        fmt="o",
                        color=color,
                        label=label,
                        markeredgecolor="black",
                    )
                    colorss.append(color)
            # Plot the Bagpipes input
            # tab = self.provide_bagpipes_phot(rbin)
            # for row, band in zip(tab, self.bands):
            #    flux, err = row * u.uJy
            #    wav = self.filter_wavs[band]
            # ax.errorbar(wav.to(wav_unit)+0.05*u.um, flux.to(flux_unit), yerr=err.to(flux_unit), fmt='x', color=color)

        ax.set_xlabel(f"Wavelength ({wav_unit})")
        ax.set_ylabel(f"Flux ({flux_unit})")
        if flux_unit == u.ABmag:
            ax.invert_yaxis()

        if min_flux is not None:
            ax.set_ylim(min_flux, ax.get_ylim()[1])

        if legend:
            ax.legend(loc="lower right", frameon=False)

    def run_dense_basis(
        self,
        db_atlas_path,
        fit_photometry="all",
        overwrite=False,
        use_emcee=False,
        emcee_samples=10000,
        plot=False,
        n_jobs=4,
        binmap_type=None,
        min_flux_err=0.1,
        priors=None,  # Allow passing in of priors for speed
        save_outputs=True,
    ):
        # import dense_basis as db

        atlas_path = os.path.dirname(db_atlas_path)
        # get filename
        db_atlas_name = "_".join(
            os.path.basename(db_atlas_path).split("_")[:-3]
        )

        print(f"Running dense_basis for {db_atlas_name}")
        print(f"Using atlas {db_atlas_path}")

        if not hasattr(self, "photometry_table"):
            raise Exception("Need to run measure_flux_in_bins first")
        if hasattr(self, "use_psf_type"):
            psf_type = self.use_psf_type
        else:
            psf_type = "webbpsf"

        if hasattr(self, "use_binmap_type") and binmap_type is None:
            binmap_type = self.use_binmap_type
        elif binmap_type is None:
            print("Defaulting to pixedfit binning")
            binmap_type = "pixedfit"

        print(f"Using {psf_type} PSF and {binmap_type} binning.")
        if hasattr(self, "sed_fitting_table"):
            if "dense_basis" in self.sed_fitting_table.keys():
                if (
                    db_atlas_name
                    in self.sed_fitting_table["dense_basis"].keys()
                ):
                    if not overwrite:
                        print(f"Run {db_atlas_name} already exists")
                        return

        flux_table = copy.deepcopy(
            self.photometry_table[psf_type][binmap_type]
        )

        if fit_photometry == "all":
            mask = np.ones(len(flux_table), dtype=bool)
        elif fit_photometry == "bin":
            mask = flux_table["type"] == "bin"
        elif fit_photometry == "all_total":
            mask = flux_table["type"] != "bin"
        elif fit_photometry == "MAG_AUTO":
            mask = flux_table["type"] == "MAG_AUTO"
        elif fit_photometry == "MAG_ISO":
            mask = flux_table["type"] == "MAG_ISO"
        elif fit_photometry == "MAG_BEST":
            mask = flux_table["type"] == "MAG_BEST"
        elif fit_photometry == "TOTAL_BIN":
            mask = flux_table["type"] == "TOTAL_BIN"
        elif fit_photometry == "MAG_APER_TOTAL":
            mask = flux_table["type"] == "MAG_APER_TOTAL"
        elif fit_photometry == "MAG":
            mask = (
                (flux_table["type"] == "MAG_AUTO")
                & (flux_table["type"] == "MAG_ISO")
                & (flux_table["type"] == "MAG_BEST")
            )
        elif fit_photometry.startswith("MAG_APER"):
            mask = flux_table["type"] == "MAG_APER"
        elif fit_photometry == "TOTAL_BIN+MAG_APER_TOTAL":
            mask = (flux_table["type"] == "MAG_APER_TOTAL") | (
                flux_table["type"] == "TOTAL_BIN"
            )
        else:
            raise ValueError(
                "fit_photometry must be one of: all, bin, MAG_AUTO, MAG_ISO, MAG_BEST, TOTAL_BIN, MAG or MAG_APER"
            )

        flux_table = flux_table[mask]

        print(
            f"Fitting only {fit_photometry} fluxes, which is {len(flux_table)} sources"
        )

        ids = list(flux_table["ID"])

        if flux_table[flux_table.colnames[1]].unit != u.uJy:
            for col in flux_table.colnames[3:]:
                flux_table[col] = flux_table[col].to(u.uJy)

        from .dense_basis import run_db_fit_parallel, get_priors

        fluxes = []
        errors = []
        for row in flux_table:
            flux = []
            error = []
            for i, band in enumerate(self.bands):
                flux.append(row[band].value)
                error.append(row[f"{band}_err"].value)
            fluxes.append(flux)
            errors.append(error)

        fluxes = np.array(fluxes)
        errors = np.array(errors)
        if priors is None:
            priors = get_priors(db_atlas_path)

        fit_results = Parallel(n_jobs=n_jobs)(
            delayed(run_db_fit_parallel)(
                flux,
                error,
                db_atlas_name,
                atlas_path,
                self.bands,
                use_emcee,
                emcee_samples,
                min_flux_err,
            )
            for flux, error in zip(fluxes, errors)
        )

        self.get_filter_wavs()
        filter_wavs = [
            self.filter_wavs[band].to(u.Angstrom) for band in self.bands
        ]

        return fit_results

    def _save_db_fit_results(
        self,
        fit_results,
        ids,
        db_atlas_name,
        add_to_h5=True,
        save_full=False,
        overwrite=False,
        priors=None,
        parameters_to_save=["mstar", "sfr", "Av", "Z", "z"],
        posterior_values=[50.0, 16.0, 84.0],
    ):
        """
        Save the dense_basis fit results to the h5 file

        Parameters
        ----------
        fit_results : list
        ids: list
        db_atlas_name : str
        add_to_h5 : bool
        save_full : bool

        """
        from dense_basis import makespec_atlas

        if prior is None:
            priors = get_priors(db_atlas_path)

        assert (
            len(fit_results) == len(ids)
        ), f"Length of fit_results and ids must be the same. {len(fit_results)} != {len(ids)}"
        if not hasattr(self, "sed_fitting_table"):
            self.sed_fitting_table = {}

        if "dense_basis" not in self.sed_fitting_table.keys():
            self.sed_fitting_table["dense_basis"] = {}

        if (
            db_atlas_name in self.sed_fitting_table["dense_basis"].keys()
            and not overwrite
        ):
            print(
                f"Run {db_atlas_name} already exists and overwrite is set to False. Skipping."
            )
            return

        def ujy_to_flam(data, lam):
            flam = ((3e-5) * data) / ((lam**2.0) * (1e6))
            return flam / 1e-19

        # start assembling the table

        table = QTable()

        save_seds = {}
        save_sfhs = {}
        save_pdfs = {}
        for i, fit_result in tqdm(enumerate(fit_results)):
            table_row = {}
            table_row["#ID"] = ids[i]
            try:
                mstar_map = fit_result.evaluate_MAP_mstar(
                    bw_dex=0.001,
                    smooth="kde",
                    lowess_frac=0.3,
                    bw_method="scott",
                    vb=False,
                )
                sfr_map = fit_result.evaluate_MAP_sfr(
                    bw_dex=0.001,
                    smooth="kde",
                    lowess_frac=0.3,
                    bw_method="scott",
                    vb=False,
                )
                table_row["mstar_map"] = mstar_map
                table_row["sfr_map"] = sfr_map

            except ValueError:
                pass

            fit_result.evaluate_posterior_percentiles(
                bw_dex=0.001, percentile_values=posterior_values, vb=False
            )
            for parameter in parameters_to_save:
                val = getattr(fit_result, parameter)
                for i, quantile in enumerate(posterior_values):
                    table_row[f"{parameter}_{quantile}"] = val[i]

            # sace

            zval = fit_result.z[1]
            sfh_50, sfh_16, sfh_84, time = fit_result.evaluate_posterior_SFH(
                zval, ngals=100
            )
            # Calculate posterior best-fit SED
            out_sfh = np.vstack((time, sfh_16, sfh_50, sfh_84))
            save_sfhs[ids[i]] = out_sfh

            lam_all = []
            spec_all = []
            z_all = []

            bestn_gals = np.argsort(fit_result.likelihood)
            for i in range(ngals):
                lam_gen, spec_gen = makespec_atlas(
                    fit_result.atlas,
                    bestn_gals[-(i + 1)],
                    priors,
                    mocksp,
                    cosmo,
                    filter_list=[],
                    filt_dir=[],
                    return_spec=True,
                )
                z = fit_result.atlas["zval"][bestn_gals[-(i + 1)]]
                lam_gen_obs = lam_gen * (1 + z)
                spec_flam = ujy_to_flam(
                    spec_gen * fit_result.norm_fac, lam_gen_obs
                )
                lam_all.append(lam_gen_obs)
                spec_all.append(spec_flam)
                z_all.append(z)

            # find 16th, 50th, 84th percentiles of spectrum
            spec_all = np.array(spec_all)
            lam_all = np.array(lam_all)
            z_all = np.array(z_all)

            z = np.median(z_all)
            spec_16, spec_50, spec_84 = np.percentile(
                spec_all, [16, 50, 84], axis=0
            )
            lam = np.median(lam_all, axis=0)
            out = np.vstack((lam, spec_16, spec_50, spec_84))

            save_seds[ids[i]] = out

            if save_full:
                # Just save norm_fac and chi2 array. Other parameters can be calculated from these.
                save_pdfs[ids[i]] = {}
                save_pdfs[ids[i]]["chi2_array"] = fit_result.chi2_array
                save_pdfs[ids[i]]["norm_fac"] = fit_result.norm_fac

        self.sed_fitting_table["dense_basis"][db_atlas_name] = table

        # sed_fitting_pdfs/sed_tool/name
        # sed_fitting_seds/sed_tool/name
        # sed_fitting_sfh/sed_tool/name

        # Then Bagpipes can always work the same way

        if add_to_h5:
            write_table_hdf5(
                self.sed_fitting_table["dense_basis"][db_atlas_name],
                self.h5_path,
                f"sed_fitting_table/dense_basis/{db_atlas_name}",
                serialize_meta=True,
                overwrite=True,
                append=True,
            )

            # Save SED
            for i in save_seds.keys():
                self.add_to_h5(
                    save_seds[i],
                    f"sed_fitting_seds/dense_basis/{db_atlas_name}/",
                    i,
                    overwrite=overwrite,
                    setattr_gal=None,
                    force=True,
                )

            # Save SFH
            for i in save_sfhs.keys():
                self.add_to_h5(
                    save_sfhs[i],
                    f"sed_fitting_sfhs/dense_basis/{db_atlas_name}/",
                    i,
                    overwrite=overwrite,
                    setattr_gal=None,
                    force=True,
                )

            # Save PDFs
            if save_full:
                for i in save_pdfs.keys():
                    for key in save_pdfs[i].keys():
                        self.add_to_h5(
                            save_pdfs[i][key],
                            f"sed_fitting_pdfs/dense_basis/{db_atlas_name}/{i}/",
                            key,
                            overwrite=overwrite,
                            setattr_gal=None,
                            force=True,
                        )

    def plot_filter_transmission(
        self,
        bands="all",
        fig=None,
        ax=None,
        label=True,
        wav_unit=u.um,
        cmap="RdYlBu_r",
    ):
        self.get_filter_wavs()

        if fig is None:
            fig = plt.figure(figsize=(8, 6))
        if ax is None:
            ax = fig.add_subplot(111)

        if bands == "all":
            bands = self.bands

        cmap = plt.get_cmap(cmap)
        norm = Normalize(vmin=0, vmax=len(bands))
        colors = {band: cmap(norm(i)) for i, band in enumerate(bands)}

        from astroquery.svo_fps import SvoFps

        for band in bands:
            try:
                filter_name = self.band_codes[band]
                filter_profile = SvoFps.get_transmission_data(filter_name)
                wav = np.array(filter_profile["Wavelength"]) * u.Angstrom
                trans = np.array(filter_profile["Transmission"])
                ax.plot(
                    wav.to(wav_unit),
                    trans,
                    label=band if label else "",
                    color=colors[band],
                )
            except FileNotFoundError:
                pass

        ax.set_xlabel(f"Wavelength ({wav_unit:latex})")
        ax.set_ylabel(r"T$_{\lambda}$")

    def _plot_radial_profile(
        self,
        map_name,
        center="auto",
        nrad=20,
        minrad=1,
        maxrad=45,
        profile_type="cumulative",
        log=False,
        map_units=None,
        output_radial_units="pix",
        return_map=False,
        z_for_scale=None,
    ):
        """Plot the radial profile of the map

        Allowed profile types:
        - cumulative: sum of pixel values within radius
        - differential: difference between sum of pixel values within radius and sum of pixel values within radius - 1
        - mean: mean of pixel values within radius
        - median: median of pixel values within radius
        """

        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5), dpi=100)

        if output_radial_units == "pix":
            pix_scale = 1 * u.pix
        elif output_radial_units == "arcsec":
            pix_scale = self.im_pixel_scales[self.bands[-1]]
        elif output_radial_units == "kpc":
            if z_for_scale is None:
                print(
                    f"Output radial units set to kpc, but no redshift provided. Using self.redshift {self.redshift}"
                )
                z = self.redshift
            else:
                z = z_for_scale

            d_A = cosmo.angular_diameter_distance(z).to(u.kpc)
            pixel_size = self.im_pixel_scales[self.bands[-1]].to(u.arcsec)
            pix_scale = (d_A * pixel_size).to(u.kpc)
        else:
            raise ValueError(
                f"Output radial units {output_radial_units} not recognised"
            )

        axs[1].set_xlabel("Radius (pixels)")
        y_label = "Radial Profile"

        if center == "auto":
            axs[0].set_title("Centre of Image")
            center = (cutout_size / 2, cutout_size / 2)
        elif center == "peak":
            axs[0].set_title("Peak Center")
            center = np.unravel_index(np.argmax(map), map.shape)
            # Reverse x and y
            center = (center[1], center[0])

        elif center == "com":
            axs[0].set_title("COM Center")
            # Calculate center of mass of all pixels
            x, y = np.meshgrid(np.arange(cutout_size), np.arange(cutout_size))
            x = x.flatten()
            y = y.flatten()
            m = map.flatten()
            x_com = np.sum(x * m) / np.sum(m)
            y_com = np.sum(y * m) / np.sum(m)
            center = (y_com, x_com)

        radii, value = measure_cog(
            map, center, nrad, maxrad=maxrad, minrad=minrad
        )
        plot_radii = copy(radii)

        if map_units is not None or (
            type(map) is unyt_array and map.units != unyt.dimensionless
        ):
            y_label = f"Radial Profile ({map_units})"

        if pix_scale is not None:
            radii *= pix_scale
            axs[1].set_xlabel(f"Radius ({pix_scale.units})")

        if profile_type == "cumulative":
            value = np.cumsum(value)

        elif profile_type == "differential":
            value = np.diff(value)
            radii = radii[:-1]

        elif profile_type == "differential_density":
            # Calculate the differential density profile, normalised by the area of the annulus
            value = np.diff(value)
            radii = radii[:-1]
            area = np.pi * (radii[1:] ** 2 - radii[:-1] ** 2)
            area = np.append(np.pi * radii[0] ** 2, area)
            str_unit = r"pix$^{-2}$"

            if pix_scale is not None:
                # Convert area from pixels^2 to pix_scale^2
                area *= pix_scale**2
                str_unit = rf"{pix_scale.units}$^{{-2}}$"

            value /= area

            if map_units is not None or (
                type(map) is unyt_array and map.units != unyt.dimensionless
            ):
                y_label = y_label[:-1] + f" {str_unit})"

        else:
            raise ValueError(f"Profile type {profile_type} not recognised")

    def plot_overview(
        self,
        figsize=(12, 8),
        bands_to_show=["F814W", "F115W", "F200W", "F335M", "F444W"],
        bins_to_show=["TOTAL_BIN", "MAG_APER_TOTAL", "1"],
        show=True,
        flux_unit=u.ABmag,
        save=False,
        rgb_stretch=0.001,
        rgb_q=0.001,
        binmap_type="pixedfit",
        legend=True,
        min_sed_flux=31 * u.ABmag,
    ):
        # GridSpec

        fig = plt.figure(
            figsize=figsize,
            dpi=200,
            facecolor="white",
            constrained_layout=True,
        )
        gs = fig.add_gridspec(
            3, 2, width_ratios=[1.9, 1], height_ratios=[1, 1, 1.5]
        )

        fig_sed = fig.add_subfigure(gs[0:2, 0])
        fig_bins = fig.add_subfigure(gs[0:2, 1])
        fig_cutouts = fig.add_subfigure(gs[2, :])

        # SED - single ax
        gs = fig_sed.add_gridspec(2, 1, height_ratios=[3, 1], hspace=-0.1)

        ax_sed = fig_sed.add_subplot(gs[0])

        # Plot SED
        self.plot_photometry_bins(
            ax=ax_sed,
            fig=fig_sed,
            show=False,
            bins_to_show=bins_to_show,
            flux_unit=flux_unit,
            legend=legend,
            label_individual=True,
            binmap_type=binmap_type,
            min_flux=min_sed_flux,
        )

        ax_filt = fig_sed.add_subplot(gs[1])
        # remove xticks and xticklabels from ax_sed
        ax_sed.set_xticks([])
        ax_sed.set_xticklabels([])
        ax_sed.set_xlabel("")

        self.plot_filter_transmission(
            bands="all", fig=fig_sed, ax=ax_filt, label=False, wav_unit=u.um
        )

        ax_filt.spines["top"].set_visible(False)
        ax_sed.spines["bottom"].set_alpha(0.2)
        #

        # Set ax_filt xlim to match ax_sed
        ax_filt.set_xlim(ax_sed.get_xlim())

        ax_filt.set_ylim(0.01, None)
        ax_filt.set_yticklabels("")
        # ax_filt remove upper xticks
        ax_filt.tick_params(
            axis="x", which="both", bottom=True, top=False, labelbottom=True
        )
        ax_filt.set_yticks([])

        # Plot cutouts
        self.plot_cutouts(
            fig=fig_cutouts, show=False, bands=bands_to_show + ["F444W"]
        )

        second_last_ax = fig_cutouts.axes[-2]
        # Plot a scalebar on the second last ax

        scalebar = AnchoredSizeBar(
            second_last_ax.transData,
            0.5 / self.im_pixel_scales[bands_to_show[-1]].value,
            '0.5"',
            "lower right",
            pad=0.3,
            color="white",
            frameon=False,
            size_vertical=1,
            fontproperties=FontProperties(size=18),
            path_effects=[
                PathEffects.withStroke(linewidth=2, foreground="black")
            ],
        )
        second_last_ax.add_artist(scalebar)
        # Plot scalebar with physical size

        # Calculate size and resolution of the cutouts
        d_A = cosmo.angular_diameter_distance(self.redshift).to(u.kpc)

        scalebar_as = 0.5 * u.arcsec

        re = scalebar_as.to(u.rad).value * d_A.value

        scalebar = AnchoredSizeBar(
            second_last_ax.transData,
            scalebar_as.value / self.im_pixel_scales[bands_to_show[-1]].value,
            f"{re:.1f} kpc",
            "upper left",
            pad=0.3,
            color="white",
            frameon=False,
            size_vertical=1,
            fontproperties=FontProperties(size=18),
            path_effects=[
                PathEffects.withStroke(linewidth=2, foreground="black")
            ],
        )
        second_last_ax.add_artist(scalebar)

        last_cutout_ax = fig_cutouts.axes[-1]
        # Plot detection segmap
        if self.det_data is not None:
            arr = self.det_data["seg"]
        else:
            arr = self.seg_imgs["F444W"]

        # Renormalize arr so unique values are 0, 1, 2, 3, ...
        unique_vals = np.unique(arr)
        for i, val in enumerate(unique_vals):
            arr[arr == val] = i
        arr[arr == 0] == np.nan

        if type(arr) is u.Quantity:
            arr = arr.value

        # remove everything from last_cutout_ax
        last_cutout_ax.clear()
        norm = last_cutout_ax.imshow(
            arr, origin="lower", interpolation="none", cmap="magma"
        )
        # write 'Det Seg' or 'F444W Seg' on the plot

        last_cutout_ax.text(
            0.9,
            0.9,
            "Det Seg" if self.det_data is not None else "F444W Seg",
            color="w",
            transform=last_cutout_ax.transAxes,
            ha="right",
            va="top",
            fontsize=12,
            fontweight="bold",
            path_effects=[
                PathEffects.Stroke(linewidth=2, foreground="black"),
                PathEffects.Normal(),
            ],
        )
        last_cutout_ax.axis("off")

        # Cente
        # Plot RGB and pixedfit in fig_bins - 2 rows
        ax_rgb = fig_bins.add_subplot(211)
        ax_pixedfit = fig_bins.add_subplot(212)

        self.plot_lupton_rgb(
            ax=ax_rgb,
            fig=fig_bins,
            show=False,
            red=["F444W"],
            green=["F356W"],
            blue=["F200W"],
            return_array=False,
            stretch=rgb_stretch,
            q=rgb_q,
            label_bands=True,
        )

        # plot pixedfit
        norm = ax_pixedfit.imshow(
            getattr(self, f"{binmap_type}_map"),
            origin="lower",
            interpolation="none",
            cmap="nipy_spectral_r",
        )
        # steal space from ax_pixedfit for colorbar
        ax_divider = make_axes_locatable(ax_pixedfit)

        cax = ax_divider.append_axes("right", size="5%", pad="2%")
        fig.colorbar(norm, cax=cax, orientation="vertical", label="Bin Number")
        # move colorbar ticks to top
        cax.yaxis.set_ticks_position("right")
        cax.yaxis.set_label_position("right")

        ax_pixedfit.axis("off")

        # fig.suptitle(f'Galaxy {self.galaxy_id} Overview')

        # Add a table with galaxy name, redshift (eazy), number of bins
        props = dict(boxstyle="square", facecolor="white", alpha=0.5)
        try:
            z50 = self.meta_properties["zbest_fsps_larson_zfree"]
            z16 = self.meta_properties["zbest_16_fsps_larson_zfree"]
            z84 = self.meta_properties["zbest_84_fsps_larson_zfree"]
            if type(z84) in [list, np.ndarray]:
                z84 = z84[0]
            if type(z16) in [list, np.ndarray]:
                z16 = z16[0]
            error = f"$^{{+{np.abs(z84-z50):.2f}}}_{{-{np.abs(z50-z16):.2f}}}$"
        except:
            z50 = self.redshift
            error = ""

        textstr = f"Galaxy {self.galaxy_id}\nRedshift: {z50:.2f}{error}\nBinmap Type: {binmap_type.replace('_', ' ')}\nNumber of bins: {len(np.unique(getattr(self, f'{binmap_type}_map')))-1}"
        fig.text(
            0.01,
            0.45,
            textstr,
            transform=fig.transFigure,
            fontsize=13,
            verticalalignment="top",
            bbox=props,
        )
        if self.sky_coord is not None:
            textstr = f"RA: {self.sky_coord.ra.degree:.4f}\nDec: {self.sky_coord.dec.degree:.4f}\nCutout size: {self.cutout_size} pix"
            fig.text(
                0.5,
                0.45,
                textstr,
                transform=fig.transFigure,
                fontsize=13,
                verticalalignment="top",
                bbox=props,
            )

        if save:
            if not os.path.exists(f"{resolved_galaxy_dir}/diagnostic_plots"):
                os.makedirs(f"{resolved_galaxy_dir}/diagnostic_plots")

            fig.savefig(
                f"{resolved_galaxy_dir}/diagnostic_plots/{self.survey}_{self.galaxy_id}_overview.png",
                dpi=200,
            )

        return fig

    def run_bagpipes(
        self,
        bagpipes_config,
        filt_dir=bagpipes_filter_dir,
        fit_photometry="all",
        run_dir="pipes/",
        overwrite=False,
        overwrite_internal=False,
        mpi_serial=False,
        use_mpi=True,
        only_run=False,  # Only run the bagpipes fit, don't do any other processing - good for MPI
        time_calls=False,
        return_run_args=False,
        override_binmap_type=None,
        skip_single_bin=True,  # Skip fitting single bins
    ):
        # meta - run_name, use_bpass, redshift (override)
        assert (
            type(bagpipes_config) is dict
        ), (
            "Bagpipes config must be a dictionary"
        )  # Could load from a file as well
        meta = bagpipes_config.get("meta", {})
        # Override fit_photometry if it is in the meta
        if "fit_photometry" in meta.keys():
            print(
                f'Overriding fit_photometry: {fit_photometry} to "{meta["fit_photometry"]}"'
            )
            fit_photometry = meta["fit_photometry"]

        print(
            f"Fitting photometry: {fit_photometry} with run_name {meta.get('run_name', 'default')}"
        )

        fit_instructions = bagpipes_config.get("fit_instructions", {})

        use_bpass = meta.get("use_bpass", False)
        os.environ["use_bpass"] = str(int(use_bpass))
        run_name = meta.get("run_name", "default")

        redshift = meta.get(
            "redshift", self.redshift
        )  # PLACEHOLDER for self.redshift
        if type(redshift) is str:
            # logic to choose a redshift to fit at
            if redshift == "eazy":
                print("Using EAZY redshift.")
                redshift = self.meta_properties["zbest_fsps_larson_zfree"]
            elif redshift == "self":
                print("Using self redshift.")
                redshift = self.redshift
            elif redshift in self.sed_fitting_table["bagpipes"].keys():
                table = self.sed_fitting_table["bagpipes"][redshift]
                redshift_id = meta.get("redshift_id", table["#ID"][0])
                row_index = table["#ID"] == redshift_id
                table = table[row_index]
                if len(table) == 0:
                    raise Exception(
                        f"No ID {redshift_id} found in {redshift} table when attempting to fetch redshift for bagpipes"
                    )

                if "redshift_50" in table.colnames:
                    redshift = float(table["redshift_50"][0])
                elif "input_redshift" in table.colnames:
                    redshift = float(table["input_redshift"][0])
        try:
            from mpi4py import MPI

            rank = MPI.COMM_WORLD.Get_rank()
            size = MPI.COMM_WORLD.Get_size()
            from mpi4py.futures import MPIPoolExecutor

            if size > 1:
                print("Running with mpirun/mpiexec detected.")
                save_out = False  # Avoids locking issues

        except ImportError:
            rank = 0
            size = 1

        if return_run_args:
            rank = 10  # Skips making folders etc.

        redshift_sigma = meta.get("redshift_sigma", 0)
        min_redshift_sigma = meta.get("min_redshift_sigma", 0)
        sampler = meta.get("sampler", "multinest")

        if redshift_sigma == "eazy":
            z50 = self.meta_properties["zbest_fsps_larson_zfree"]
            z16 = self.meta_properties["zbest_16_fsps_larson_zfree"]
            z84 = self.meta_properties["zbest_84_fsps_larson_zfree"]
            if type(z84) in [list, np.ndarray]:
                z84 = z84[0]
            if type(z16) in [list, np.ndarray]:
                z16 = z16[0]
            redshift_sigma = np.mean([z84 - z50, z50 - z16])
        elif redshift_sigma == "min":
            # Ensures
            redshift_sigma = min_redshift_sigma

        if redshift_sigma != 0:
            redshift_sigma = max(redshift_sigma, min_redshift_sigma)
        else:
            redshift_sigma = None

        if (
            type(redshift) in [float, np.float64, np.float32]
            and redshift_sigma is not None
        ):
            print(
                f"Fitting Redshift = {redshift}, Redshift sigma = {redshift_sigma}, allowed range = ({redshift - 3*redshift_sigma}, {redshift + 3*redshift_sigma})"
            )
        elif redshift_sigma is None and type(redshift) in [float, np.float64]:
            print(f"Fitting fixed redshift = {redshift}.")
        elif type(redshift) in [list, np.ndarray, tuple]:
            assert (
                len(redshift) == 2
            ), "Redshift must be a float or a list of length 2"
            print(f"Allowing free redshift: {redshift[0]} < z < {redshift[1]}")
            fit_instructions["redshift"] = (redshift[0], redshift[1])
            redshift = None
        else:
            print(
                redshift, type(redshift), redshift_sigma, type(redshift_sigma)
            )
            raise ValueError("I don't understand the redshift input.")

        if not hasattr(self, "photometry_table"):
            raise Exception("Need to run measure_flux_in_bins first")
        if hasattr(self, "use_psf_type"):
            psf_type = self.use_psf_type
        else:
            psf_type = "webbpsf"

        if override_binmap_type:
            print(f"Overriding binmap type to {override_binmap_type}")
            binmap_type = override_binmap_type
            self.use_binmap_type = binmap_type
        elif hasattr(self, "use_binmap_type"):
            binmap_type = self.use_binmap_type
        else:
            binmap_type = "pixedfit"

        if hasattr(self, "sed_fitting_table"):
            if "bagpipes" in self.sed_fitting_table.keys():
                if run_name in self.sed_fitting_table["bagpipes"].keys():
                    if overwrite or overwrite_internal:
                        print(f"Overwriting run {run_name}")
                    else:
                        print(f"Run {run_name} already exists")
                        if not return_run_args:
                            return

        # print(psf_type, binmap_type)
        # print(self.photometry_table.keys())
        flux_table = self.photometry_table[psf_type][binmap_type]
        if fit_photometry == "all":
            mask = np.ones(len(flux_table), dtype=bool)
        elif fit_photometry == "bin":
            mask = flux_table["type"] == "bin"
        elif fit_photometry == "all_total":
            mask = flux_table["type"] != "bin"
        elif fit_photometry == "MAG_AUTO":
            mask = flux_table["type"] == "MAG_AUTO"
        elif fit_photometry == "MAG_ISO":
            mask = flux_table["type"] == "MAG_ISO"
        elif fit_photometry == "MAG_BEST":
            mask = flux_table["type"] == "MAG_BEST"
        elif fit_photometry == "TOTAL_BIN":
            mask = flux_table["type"] == "TOTAL_BIN"
        elif fit_photometry == "MAG_APER_TOTAL":
            mask = flux_table["type"] == "MAG_APER_TOTAL"
        elif fit_photometry == "MAG":
            mask = (
                (flux_table["type"] == "MAG_AUTO")
                & (flux_table["type"] == "MAG_ISO")
                & (flux_table["type"] == "MAG_BEST")
            )
        elif fit_photometry.startswith("MAG_APER"):
            mask = flux_table["type"] == "MAG_APER"
        elif fit_photometry == "TOTAL_BIN+MAG_APER_TOTAL":
            mask = (flux_table["type"] == "MAG_APER_TOTAL") | (
                flux_table["type"] == "TOTAL_BIN"
            )
        else:
            raise ValueError(
                "fit_photometry must be one of: all, bin, MAG_AUTO, MAG_ISO, MAG_BEST, TOTAL_BIN, MAG or MAG_APER"
            )

        flux_table = flux_table[mask]
        print(
            f"Fitting only {fit_photometry} fluxes, which is {len(flux_table)} sources"
        )

        if (
            fit_photometry == "bin"
            and len(flux_table) == 1
            and skip_single_bin
        ):
            print(f'Only one bin found, skipping fitting for "{run_name}"')
            return

        ids = list(flux_table["ID"])

        vary_filter_list = False
        nircam_filts = [
            f"{filt_dir}/{band}_LePhare.txt" for band in self.bands
        ]

        # Check flux_table, see if any fluxes and errors are perfect 0s
        ids_bands = {band: [] for band in self.bands}

        for band in self.bands:
            flux_col_name = band
            fluxerr_col_name = f"{band}_err"
            mask = (flux_table[flux_col_name] == 0) & (
                flux_table[fluxerr_col_name] == 0
            )
            if np.any(mask):
                vary_filter_list = True
                print(
                    f"Warning: {np.sum(mask)} sources have 0 flux and 0 error in {band} band"
                )
                # Get IDs
                ids_bands[band] = list(flux_table["ID"][mask])

        # Make list of actual bands for each source

        if vary_filter_list:
            print("Varying filter list")
            nircam_filts = []
            for id in ids:
                bands = []
                for band in self.bands:
                    if id not in ids_bands[band]:
                        bands.append(f"{filt_dir}/{band}_LePhare.txt")
                nircam_filts.append(bands)

        if redshift is None:
            print("Allowing free redshift.")
            redshifts = None
        else:
            redshifts = np.ones(len(flux_table)) * redshift

        import bagpipes as pipes
        # out_subdir (moving files)

        out_subdir = f"{run_name}/{self.survey}/{self.galaxy_id}"
        # make a temp directory
        # hash out_subdir to get a unique shorter name
        import hashlib

        out_subdir_encoded = hashlib.md5(out_subdir.encode()).hexdigest()

        existing_files = []
        for i in [out_subdir, out_subdir_encoded]:
            for j in ["posterior"]:
                path = f"{run_dir}/{j}/{i}"
                if rank == 0:
                    os.makedirs(path, exist_ok=True)
                    os.chmod(path, 0o777)
                existing_files += glob.glob(f"{path}/*")

        # os.makedirs(f"{run_dir}/cats/{out_subdir_encoded}", exist_ok=True)

        existing_filenames = [
            os.path.basename(file) for file in existing_files
        ]

        # path_fits = f'{run_dir}/cats/{run_name}/{self.survey}/' # Filename is galaxy ID rather than being a folder

        # for path in [path_post, path_plots, path_sed, path_fits]:
        #    os.makedirs(path, exist_ok=True)
        #    os.chmod(path, 0o777)

        # existing_files = glob.glob(f'{path_post}/*') + glob.glob(f'{path_plots}/*') + glob.glob(f'{path_sed}/*') + glob.glob(f'{path_fits}/*')
        # Copy any .h5 in out_subdir_encoded to out_subdir
        if rank == 0:
            for file in existing_files:
                if out_subdir_encoded in file:
                    new_file = file.replace(out_subdir_encoded, out_subdir)
                    # If encoded file path, check if the new file exists and if not, rename
                    if not os.path.exists(new_file):
                        os.rename(file, new_file)
                    else:
                        # If it does exist, remove the file if we are overwriting
                        if overwrite:
                            os.remove(new_file)
                            os.rename(file, new_file)

        exist_already = False
        if overwrite:
            if rank == 0:
                for file in existing_files:
                    print(f"Removing {file}")
                    os.remove(file)
        else:  # Check if already run
            mask = np.zeros(len(ids))
            for pos, gal_id in enumerate(ids):
                if f"{gal_id}.h5" in existing_filenames:
                    print(f"{gal_id}.h5 already exists")
                    mask[pos] = 1

            # Check if catalogue already exists
            path = f"{run_dir}/cats/{run_name}/{self.survey}/{self.galaxy_id}.fits"
            if os.path.exists(path):
                cat_exists = True
            else:
                cat_exists = False

            if np.all(mask == 1) and cat_exists:
                print("All files already exist")
                exist_already = True
            elif np.all(mask == 1) and not cat_exists:
                print(
                    "All files exist except catalogue, rerunning to build catalogue."
                )

        if np.any(mask == 1):
            if rank == 0:
                # Rename the out_subdir to out_subdir_encoded
                for i in ["posterior"]:
                    path = f"{run_dir}/{i}/{out_subdir}"
                    new_path = f"{run_dir}/{i}/{out_subdir_encoded}"
                    os.rename(path, new_path)
                # Move catalogue

            path = f"{run_dir}/cats/{run_name}/{self.survey}/{self.galaxy_id}.fits"
            nrun_name = f"{run_dir}/cats/{out_subdir_encoded}.fits"
            if rank == 0:
                if os.path.exists(path) and not os.path.exists(nrun_name):
                    os.rename(path, nrun_name)

        if "continuity" in fit_instructions.keys() and meta.get(
            "update_cont_bins", False
        ):
            # Update the continuity bins
            print("Updating continuity bins")
            from .bagpipes.plotpipes import calculate_bins

            fit_instructions["continuity"]["bin_edges"] = list(
                calculate_bins(
                    redshift=redshifts[0],
                    num_bins=meta.get("cont_nbins", 6),
                    first_bin=meta.get("cont_first_bin", 10 * u.Myr),
                    second_bin=None,
                    return_flat=True,
                    output_unit="Myr",
                    log_time=False,
                )
            )

        meta["binmap_type"] = binmap_type

        if return_run_args:
            # print(type(redshifts), type(redshift_sigma), type(nircam_filts), type(self.provide_bagpipes_phot('1')))
            # dog
            return {
                "ids": ids,
                "fit_instructions": fit_instructions,
                "meta": meta,
                "run_name": run_name,
                "galaxy_id": self.galaxy_id,
                "already_run": exist_already,
                "phot": [self.provide_bagpipes_phot(i).tolist() for i in ids],
                "cat_filt_list": nircam_filts,
                "redshifts": list(redshifts),
                "redshift_sigma": redshift_sigma,
                "out_dir": f"{run_dir}/posterior/{out_subdir}",
            }
        if not exist_already or return_run_args:
            fit_cat = pipes.fit_catalogue(
                ids,
                fit_instructions,
                self.provide_bagpipes_phot,
                spectrum_exists=False,
                photometry_exists=True,
                run=out_subdir_encoded,
                make_plots=False,
                cat_filt_list=nircam_filts,
                vary_filt_list=vary_filter_list,
                redshifts=redshifts,
                redshift_sigma=redshift_sigma,
                save_pdf_txts=False,
                full_catalogue=True,
                time_calls=time_calls,
            )  # analysis_function=custom_plotting,
            print("Beginning fit")
            print(fit_instructions)
            # Run this with MPI
            if mpi_serial and size > 1:
                print("Running with MPI, one galaxy per core. ")
            fit_cat.fit(
                verbose=False,
                mpi_serial=mpi_serial,
                sampler=sampler,
                use_mpi=use_mpi,
            )

        if rank == 0:
            # Move files to the correct location
            for i in ["posterior", "plots", "pdfs", "seds", "sfr"]:
                new_path = f"{run_dir}/{i}/{out_subdir}"
                current_path = f"{run_dir}/{i}/{out_subdir_encoded}"
                # If the current path is empty or doesn't exist, continue
                if (
                    not os.path.exists(current_path)
                    or len(glob.glob(f"{current_path}/*")) == 0
                ):
                    continue
                # Make parent directory
                os.makedirs(os.path.dirname(new_path), exist_ok=True)
                if os.path.exists(new_path):
                    # Just move the files
                    for file in glob.glob(f"{current_path}/*"):
                        new_file = file.replace(current_path, new_path)
                        os.rename(file, new_file)
                    # Remove the empty directory
                    os.rmdir(current_path)
                else:
                    os.rename(current_path, new_path)

            # Move catalogue
            path = f"{run_dir}/cats/{run_name}/{self.survey}/{self.galaxy_id}.fits"
            os.makedirs(
                f"{run_dir}/cats/{run_name}/{self.survey}", exist_ok=True
            )
            old_name = f"{run_dir}/cats/{out_subdir_encoded}.fits"
            # os.makedirs(f'{run_dir}/cats/{out_subdir_encoded}', exist_ok=True)
            # Check if all folders in path exist
            os.makedirs(os.path.dirname(path), exist_ok=True)
            os.rename(old_name, path)

        """
        json_file = json.dumps(fit_instructions)
        f = open(f'{path_overall}/posterior/{out_subdir}/config.json',"w")
        f.write(json_file)
        f.close()
        """
        if not only_run and rank == 0:
            meta = {"binmap_type": binmap_type}

            self.load_bagpipes_results(run_name, meta=meta)

    def load_bagpipes_results(self, run_name, run_dir="pipes/", meta=None):
        catalog_path = (
            f"{run_dir}/cats/{run_name}/{self.survey}/{self.galaxy_id}.fits"
        )
        try:
            table = Table.read(catalog_path)
        except:
            raise Exception(f"Catalog {catalog_path} not found")

        if meta is not None:
            table.meta = meta

        # SFR map
        # Av map (dust)
        # Stellar Mass map
        # Metallicity map
        # Age map
        if getattr(self, "sed_fitting_table", None) is None:
            self.sed_fitting_table = {"bagpipes": {run_name: table}}
        else:
            if "bagpipes" not in self.sed_fitting_table.keys():
                self.sed_fitting_table["bagpipes"] = {run_name: table}
            else:
                self.sed_fitting_table["bagpipes"][run_name] = table

        write_table_hdf5(
            self.sed_fitting_table["bagpipes"][run_name],
            self.h5_path,
            f"sed_fitting_table/bagpipes/{run_name}",
            serialize_meta=True,
            overwrite=True,
            append=True,
        )

    def convert_table_to_map(
        self, table, id_col, value_col, map, remove_log10=False
    ):
        changed = np.zeros(map.shape)
        return_map = copy.deepcopy(map)
        for row in table:
            gal_id = row[id_col]
            try:
                gal_id = float(gal_id)
            except:
                continue
            value = row[value_col]
            return_map[map == float(gal_id)] = (
                value if not remove_log10 else 10**value
            )
            changed[map == float(gal_id)] = 1
        # print(f'changed {np.sum(changed)} pixels out of {len(map.flatten())} pixels')
        return_map[changed == 0] = np.nan
        return return_map

    def param_unit(self, param):
        zsun = u.def_unit(
            "Zsun", format={"latex": r"Z_{\odot}", "html": "Z<sub>sun</sub>"}
        )
        param_dict = {
            "stellar_mass": u.Msun,
            "stellar_mass_density": u.Msun / u.kpc**2,
            "formed_mass": u.Msun,
            "sfr": u.Msun / u.yr,
            "sfr_10myr": u.Msun / u.yr,
            "sfr_density": u.Msun / u.yr / u.kpc**2,
            "dust:Av": u.mag,
            "UV_colour": u.mag,
            "VJ_colour": u.mag,
            "ssfr": u.Gyr**-1,
            "ssfr_10myr": u.Gyr**-1,
            "nsfr_10myr": u.dimensionless_unscaled,
            "nsfr": u.dimensionless_unscaled,
            "mass_weighted_zmet": zsun,
            "m_UV": u.mag,
            "M_UV": u.mag,
            "Ha_EWrest": u.AA,
            "Ha_flux": u.erg / u.s / u.cm**2,
            "Halpha_flux": u.erg / u.s / u.cm**2,
            "Halpha_EWrest": u.AA,
            "OIII_flux": u.erg / u.s / u.cm**2,
            "OIII_EWrest": u.AA,
            "Hb_flux": u.erg / u.s / u.cm**2,
            "Hb_EWrest": u.AA,
            "xi_ion_caseB": u.Hz / u.erg,
            "beta_C94": u.dimensionless_unscaled,
            "tform": u.Gyr,
            "tquench": u.Gyr,
            "age_min": u.Gyr,
            "age_max": u.Gyr,
            "massformed": u.Msun,
            "mass_weighted_age": u.Gyr,
            "chisq_phot-": u.dimensionless_unscaled,
            "metallicity": zsun,
            "Av": u.mag,
        }
        return param_dict.get(param, u.dimensionless_unscaled)

    def plot_bagpipes_corner(
        self,
        run_name=None,
        bins_to_show="all",
        save=False,
        corner_bins=25,
        facecolor="white",
        colors="black",
        cache=None,
        fig=None,
        plotpipes_dir="pipes_scripts/",
        run_dir="pipes/",
    ):
        if run_name is None:
            run_name = list(self.sed_fitting_table["bagpipes"].keys())
            if len(run_name) > 1:
                raise Exception("Multiple runs found, please specify run_name")
            else:
                run_name = run_name[0]

        table = self.sed_fitting_table["bagpipes"][run_name]

        if bins_to_show == "all":
            bins_to_show = np.unique(table["#ID"])

        if type(colors) is str:
            colors = [colors for i in range(len(bins_to_show))]

        if cache is None:
            cache = {}

        x_lims = []
        y_lims = []

        for (
            rbin,
            color,
        ) in zip(bins_to_show, colors):
            h5_path = f"{run_dir}/posterior/{run_name}/{self.survey}/{self.galaxy_id}/{rbin}.h5"

            if rbin == "RESOLVED":
                continue
            try:
                pipes_obj = self.load_pipes_object(
                    run_name,
                    rbin,
                    run_dir=run_dir,
                    cache=cache,
                    plotpipes_dir=plotpipes_dir,
                )
            except FileNotFoundError:
                print(f"File not found for {run_name} {rbin} (corner)")
                continue

            fig_xlim, fig_ylim = [], []
            fig = pipes_obj.plot_corner_plot(
                show=False,
                save=save,
                bins=corner_bins,
                type="fit_params",
                fig=fig,
                color=color,
                facecolor=facecolor,
            )
            for ax in fig.get_axes():
                fig_xlim.append(ax.get_xlim())
                fig_ylim.append(ax.get_ylim())
            x_lims.append(fig_xlim)
            y_lims.append(fig_ylim)

        if fig is not None:
            for pos, ax in enumerate(fig.get_axes()):
                if len(x_lims) > 0:
                    all_xlim = [x_lims[i][pos] for i in range(len(x_lims))]
                    all_ylim = [y_lims[i][pos] for i in range(len(y_lims))]
                    ax.set_xlim(np.min(all_xlim), np.max(all_xlim))
                    ax.set_ylim(np.min(all_ylim), np.max(all_ylim))
                # print(ax.get_xlabel(), np.min(all_xlim), np.max(all_xlim))
                # print(ax.get_ylabel(), np.min(all_ylim), np.max(all_ylim))
                if len(x_lims) > 1:
                    ax.set_title("")

        return fig, cache

    def plot_bagpipes_fit(
        self,
        run_name=None,
        axes=None,
        fig=None,
        bins_to_show="all",
        save=False,
        facecolor="white",
        marker_colors="black",
        wav_units=u.um,
        plotpipes_dir="pipes_scripts/",
        flux_units=u.ABmag,
        lw=1,
        fill_uncertainty=False,
        zorder=5,
        show_photometry=False,
        run_dir="pipes/",
        cache=None,
    ):
        if run_name is None:
            run_name = list(self.sed_fitting_table["bagpipes"].keys())
            if len(run_name) > 1:
                raise Exception("Multiple runs found, please specify run_name")
            else:
                run_name = run_name[0]
        if (
            not hasattr(self, "sed_fitting_table")
            or "bagpipes" not in self.sed_fitting_table.keys()
            or run_name not in self.sed_fitting_table["bagpipes"].keys()
        ):
            self.load_bagpipes_results(run_name)
        table = self.sed_fitting_table["bagpipes"][run_name]

        if axes is None:
            fig, axes = plt.subplots(
                1,
                1,
                figsize=(6, 3),
                constrained_layout=True,
                facecolor=facecolor,
            )
        if bins_to_show == "all":
            bins_to_show = np.unique(table["bin"])

        if type(marker_colors) is str:
            marker_colors = [marker_colors for i in range(len(bins_to_show))]

        if cache is None:
            cache = {}
        for (
            rbin,
            color,
        ) in zip(bins_to_show, marker_colors):
            h5_path = f"{run_dir}/posterior/{run_name}/{self.survey}/{self.galaxy_id}/{rbin}.h5"

            if rbin == "RESOLVED":
                total_flux, total_wav = self.get_resolved_bagpipes_sed(
                    run_name, run_dir=run_dir, plotpipes_dir=plotpipes_dir
                )
                total_flux *= u.uJy
                total_wav *= u.um

                axes.plot(
                    total_wav.to(wav_units),
                    total_flux.to(
                        flux_units, equivalencies=u.spectral_density(total_wav)
                    ),
                    label="RESOLVED",
                    color="tomato",
                )
                continue

            try:
                pipes_obj = self.load_pipes_object(
                    run_name,
                    rbin,
                    run_dir=run_dir,
                    cache=cache,
                    plotpipes_dir=plotpipes_dir,
                )
            except FileNotFoundError:
                print(f"File not found for {run_name} {rbin} (fit)")
                continue

            pipes_obj.plot_best_fit(
                axes,
                color,
                wav_units=wav_units,
                flux_units=flux_units,
                lw=lw,
                fill_uncertainty=fill_uncertainty,
                zorder=zorder,
            )

            if show_photometry:
                print("show photometry")
                pipes_obj.plot_best_photometry(
                    axes,
                    colour=color,
                    zorder=6,
                    wav_units=wav_units,
                    flux_units=flux_units,
                )

        # cbar.set_label('Age (Gyr)', labelpad=10)
        # cbar.ax.xaxis.set_ticks_position('top')
        # cbar.ax.xaxis.set_label_position('top')
        # cbar.ax.tick_params(labelsize=8)
        # cbar.ax.xaxis.set_major_formatter(ScalarFormatter())
        return fig, cache

    def get_photometry_from_region(
        self,
        region,
        override_psf_type=None,
        debug=True,
        return_array=False,
        debug_fig=None,
        save_debug_plot=True,
    ):
        """
        Take an regions object and return the photometry from the region for all bands

        """
        regmask = region.to_mask(mode="subpixels", subpixels=10)
        flux = {}
        flux_err = {}
        if override_psf_type is not None:
            psf_type = override_psf_type
        else:
            psf_type = self.use_psf_type

        for band in self.bands:
            data = self.psf_matched_data[psf_type][band]
            err = self.psf_matched_rms_err[psf_type][band]
            band_flux = regmask.multiply(data)
            flux[band] = np.sum(band_flux) * self.phot_pix_unit[band]
            band_err = regmask.multiply(err)
            flux_err[band] = (
                np.sqrt(np.sum(band_err**2)) * self.phot_pix_unit[band]
            )

        if return_array:
            flux_arr = (
                np.array([flux[band].value for band in self.bands])
                * flux[band].unit
            )
            flux_err_arr = (
                np.array([flux_err[band].value for band in self.bands])
                * flux[band].unit
            )
            return flux_arr, flux_err_arr

        if debug:
            # Plot the region on the image in a band
            if debug_fig is None:
                fig, ax = plt.subplots()
            else:
                if len(debug_fig.get_axes()) == 0:
                    ax = debug_fig.add_subplot(111)
                else:
                    ax = debug_fig.get_axes()[0]
                fig = debug_fig

            ax.imshow(data, origin="lower", cmap="viridis", norm=LogNorm())

            region.plot(
                ax=ax,
                edgecolor=region.visual["color"]
                if region.visual["color"] != 0
                else "black",
            )

            """

            re = 15  # pixels
            d_A = cosmo.angular_diameter_distance(self.redshift)
            pix_scal = u.pixel_scale(0.03 * u.arcsec / u.pixel)
            re_as = (re * u.pixel).to(u.arcsec, pix_scal)
            re_kpc = (re_as * d_A).to(u.kpc, u.dimensionless_angles())

            # First scalebar
            scalebar = AnchoredSizeBar(
                ax.transData,
                0.5 / self.im_pixel_scales[band].value,
                '0.5"',
                "lower right",
                pad=0.3,
                color="black",
                frameon=False,
                size_vertical=1,
                fontproperties=FontProperties(size=18),
            )
            ax.add_artist(scalebar)
            # Plot scalebar with physical size
            scalebar = AnchoredSizeBar(
                ax.transData,
                re,
                f"{re_kpc:.1f}",
                "upper left",
                pad=0.3,
                color="black",
                frameon=False,
                size_vertical=1,
                fontproperties=FontProperties(size=18),
            )
            scalebar.set(
                path_effects=[
                    PathEffects.withStroke(linewidth=3, foreground="white")
                ]
            )
            ax.add_artist(scalebar)
            """

            # Add a scalebar

            # fig.suptitle(f"Flux in region {band}: {flux[band].to(u.uJy)}")

            if save_debug_plot:
                path = f"{resolved_galaxy_dir}/diagnostic_plots/{self.survey}_{self.galaxy_id}_region_photometry.png"
                print("Saving region photometry plot to", path)
                plt.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.1)

        return flux, flux_err

    def plot_photometry_from_region(
        self,
        region,
        ax=None,
        fig=None,
        show=True,
        override_psf_type=None,
        facecolor="white",
        flux_unit=u.ABmag,
        wav_unit=u.micron,
        color="region",
        label=None,
        debug=False,
        debug_fig=None,
    ):
        from regions import (
            PixCoord,
            RectanglePixelRegion,
            PolygonPixelRegion,
            CirclePixelRegion,
        )

        if fig is None:
            fig = plt.figure(
                figsize=(6, 3), constrained_layout=True, facecolor=facecolor
            )

        if ax is None:
            ax = fig.add_subplot(111)

        if color == "region":
            color = region.visual["color"]

        flux, flux_err = self.get_photometry_from_region(
            region,
            override_psf_type=override_psf_type,
            debug=debug,
            debug_fig=debug_fig,
        )

        if type(color) == int:
            color = "black"
        if type(region) is PolygonPixelRegion:
            marker = "x"
        elif type(region) is CirclePixelRegion:
            marker = "o"
        elif type(region) is RectanglePixelRegion:
            marker = "s"

        self.get_filter_wavs()

        for pos, band in enumerate(self.bands):
            wav = self.filter_wavs[band]
            if flux_unit != u.ABmag:
                yerr = flux_err[band].to(
                    flux_unit,
                    equivalencies=u.spectral_density(self.filter_wavs[band]),
                )
            else:
                fnu_jy = flux[band].to(
                    u.uJy, equivalencies=u.spectral_density(wav)
                )
                fnu_jy_err = flux_err[band].to(
                    u.uJy, equivalencies=u.spectral_density(wav)
                )
                inner = fnu_jy / (fnu_jy - fnu_jy_err)
                err_up = 2.5 * np.log10(inner)
                err_low = 2.5 * np.log10(1 + (fnu_jy_err / fnu_jy))
                err_low = (
                    err_low.value if type(err_low) is u.Quantity else err_low
                )
                err_up = err_up.value if type(err_up) is u.Quantity else err_up

                yerr = [
                    np.atleast_1d(np.abs(err_low)),
                    np.atleast_1d(np.abs(err_up)),
                ]

            ax.errorbar(
                wav.to(wav_unit),
                flux[band]
                .to(flux_unit, equivalencies=u.spectral_density(wav))
                .value,
                yerr=yerr,
                color=color,
                marker=marker,
                markersize=10,
                label=label
                if (label is not None and pos == len(self.bands) - 1)
                else "",
            )

    def plot_bagpipes_sfh(
        self,
        run_name=None,
        bins_to_show="RESOLVED",
        save=False,
        facecolor="white",
        marker_colors="black",
        time_unit="Gyr",
        cmap="viridis",
        plotpipes_dir="pipes_scripts/",
        run_dir="pipes/",
        plottype="lookback",
        cache=None,
        fig=None,
        axes=None,
        plot=True,
    ):
        if run_name is None:
            run_name = list(self.sed_fitting_table["bagpipes"].keys())
            if len(run_name) > 1:
                raise Exception("Multiple runs found, please specify run_name")
            else:
                run_name = run_name[0]
        if (
            not hasattr(self, "sed_fitting_table")
            or "bagpipes" not in self.sed_fitting_table.keys()
            or run_name not in self.sed_fitting_table["bagpipes"].keys()
        ):
            self.load_bagpipes_results(run_name)
        table = self.sed_fitting_table["bagpipes"][run_name]
        if fig is None and plot:
            fig = plt.figure(
                figsize=(6, 3),
                constrained_layout=True,
                facecolor=facecolor,
                dpi=200,
            )
        if axes is None and plot:
            axes = fig.add_subplot(111)

        if type(bins_to_show) not in [list, np.ndarray]:
            bins_to_show = [bins_to_show]

        if type(marker_colors) is str and len(bins_to_show) > 1:
            cmap = plt.get_cmap(cmap)
            marker_colors = cmap(np.linspace(0, 1, len(bins_to_show)))
            # marker_colors = [marker_colors for i in range(len(bins_to_show))]

        if plottype == "lookback":
            save_name = run_name
        elif plottype == "absolute":
            save_name = f"{run_name}_absolute"

        if cache is None:
            cache = {}
        for (
            rbin,
            color,
        ) in zip(bins_to_show, marker_colors):
            h5_path = f"{run_dir}/posterior/{run_name}/{self.survey}/{self.galaxy_id}/{rbin}.h5"
            if rbin == "RESOLVED":
                """ Special case where we sum the SFH of all the bins"""
                dummy_fig, dummy_ax = plt.subplots(1, 1)
                found = False
                if self.resolved_sfh is not None:
                    if save_name in self.resolved_sfh.keys():
                        resolved_sfh = self.resolved_sfh[save_name]
                        found = True
                        x_all = resolved_sfh[:, 0] * u.Gyr
                        x_all = x_all.to(time_unit).value
                        if plot:
                            axes.plot(
                                x_all,
                                resolved_sfh[:, 2],
                                color="tomato",
                                label="RESOLVED",
                                lw=2,
                            )
                            axes.fill_between(
                                x_all,
                                resolved_sfh[:, 1],
                                resolved_sfh[:, 3],
                                color="tomato",
                                alpha=0.5,
                            )
                        continue

                if not found:
                    set = False
                    for pos, tbin in enumerate(np.unique(table["#ID"])):
                        try:
                            float(tbin)
                            pipes_obj = self.load_pipes_object(
                                run_name,
                                tbin,
                                run_dir=run_dir,
                                cache=cache,
                                plotpipes_dir=plotpipes_dir,
                            )
                            tx, tsfh = pipes_obj.plot_sfh(
                                dummy_ax,
                                color,
                                timescale=time_unit,
                                plottype=plottype,
                                logify=False,
                                cosmo=None,
                                label=rbin,
                                return_sfh=True,
                            )
                            if pos == 0:
                                x_all = tx
                                y_all = tsfh
                            else:
                                assert np.all(
                                    x_all == tx
                                ), "Time scales do not match"
                                y_all = np.sum([y_all, tsfh], axis=0)

                            set = True

                        except (ValueError, FileNotFoundError) as e:
                            print("error!")
                            print(run_name, tbin)
                            print(e)
                            # print(traceback.format_exc())
                            continue

                    if set:
                        if plot:
                            axes.plot(
                                x_all,
                                y_all[:, 1],
                                color="tomato",
                                label="RESOLVED",
                                lw=2,
                            )
                            axes.fill_between(
                                x_all,
                                y_all[:, 0],
                                y_all[:, 2],
                                color="tomato",
                                alpha=0.5,
                            )

                        # Save resolved SFH
                        x_all *= u.Unit(time_unit)
                        x_all = x_all.to(u.Gyr)
                        if type(x_all) is u.Quantity:
                            x_all = x_all.value

                        save_array = np.vstack(
                            (x_all, y_all[:, 0], y_all[:, 1], y_all[:, 2])
                        ).T

                        self.add_to_h5(
                            save_array,
                            "resolved_sfh",
                            save_name,
                            overwrite=True,
                            meta={
                                "time_unit": time_unit,
                                "plottype": plottype,
                            },
                        )
                        if self.resolved_sfh is None:
                            self.resolved_sfh = {}
                        self.resolved_sfh[save_name] = save_array

                plt.close(dummy_fig)
            else:
                try:
                    pipes_obj = self.load_pipes_object(
                        run_name,
                        rbin,
                        run_dir=run_dir,
                        cache=cache,
                        plotpipes_dir=plotpipes_dir,
                    )
                except FileNotFoundError:
                    print(f"File not found for {run_name} {rbin} (sfh)")
                    continue
                if plot:
                    pipes_obj.plot_sfh(
                        axes,
                        color,
                        modify_ax=True,
                        add_zaxis=True,
                        timescale=time_unit,
                        plottype=plottype,
                        logify=False,
                        cosmo=None,
                        label=rbin,
                    )
        if plot:
            axes.legend(fontsize=8)

        # cbar.set_label('Age (Gyr)', labelpad=10)
        # cbar.ax.xaxis.set_ticks_position('top')
        # cbar.ax.xaxis.set_label_position('top')
        # cbar.ax.tick_params(labelsize=8)
        # cbar.ax.xaxis.set_major_formatter(ScalarFormatter())
        if plot:
            return fig, cache

    def add_flux_aper_total(
        self,
        catalogue_path,
        id_column="NUMBER",
        overwrite=False,
        columns_to_add=[
            "KRON_RADIUS_",
            "a_",
            "b_",
            "theta_",
            "area_",
            "flux_ratio_",
            "EE_correction_",
            "total_correction_",
            "FLUX_APER_band_TOTAL_Jy",
            "FLUXERR_APER_band_TOTAL_Jy",
        ],
        stacked_band="F277W+F356W+F444W",
    ):
        if (
            getattr(self, "total_photometry", None) is not None
            and not overwrite
        ):
            print(f"Total photometry already loaded for {self.galaxy_id}.")
            return

        table = Table.read(catalogue_path)
        mask = [str(i) == str(self.galaxy_id) for i in table[id_column]]

        # fix mask so it isn't single boolean

        if type(mask) is bool:
            raise Exception("mask not working properly")
        table = table[mask]
        if len(table) == 0:
            raise Exception(f"ID {self.galaxy_id} not found in {table}.")
        if len(table) > 1:
            print(len(table))
            raise Exception(f"Multiple IDs found for {self.galaxy_id}.")

        total_photometry = {}
        band_cols = [col for col in columns_to_add if "band" in col]
        other_cols = [col for col in columns_to_add if col.endswith("_")]

        for band in self.bands:
            zp = self.im_zps[band]
            total_photometry[band] = {}
            for col in band_cols:
                if "band" in col:
                    col = col.replace("band", band)
                    val = table[col]

                    val = (
                        val[0]
                        if type(val) in [np.ndarray, list, Column]
                        else val
                    )
                    # total_photometry[band][col] = val
                    if "FLUX_" in col:
                        # Convert to uJy
                        if getattr(val, "unit", None) is None:
                            flux = val * u.Jy
                        else:
                            flux = val

                        if type(flux) in [np.ndarray, list]:
                            flux = flux[0]

                        # 0**(-zp/2.5) * 3631 * u.Jy
                        flux = flux.to(u.uJy)
                        total_photometry[band]["flux"] = flux.value
                        total_photometry[band]["flux_unit"] = str(flux.unit)

                    elif "FLUXERR_" in col:
                        if getattr(val, "unit", None) is None:
                            err = val * u.Jy
                        else:
                            err = val

                        # ensure flux err is not a single length array
                        if type(err) in [np.ndarray, list]:
                            err = err[0]

                        if type(err) in [np.ndarray, list]:
                            err = err[0]

                        # if err has __len__ attribute, it is a quantity
                        if hasattr(err, "__len__"):
                            err = err[0]
                        if getattr(err, "unit", None) is None:
                            err = err * u.Jy

                        err = err.to(u.uJy).value

                        total_photometry[band]["flux_err"] = err

        total_photometry[stacked_band] = {}
        for col in other_cols:
            data = table[f"{col}{stacked_band}"]
            data = (
                data[0] if type(data) in [np.ndarray, list, Column] else data
            )
            if type(data) in [u.Quantity, u.Magnitude, Masked(u.Quantity)]:
                data = data.value

            total_photometry[stacked_band][f"{col}{stacked_band}"] = data

        self.add_to_h5(
            total_photometry,
            "total_photometry",
            "total_photometry",
            overwrite=True,
            setattr_gal="total_photometry",
        )

    def plot_bagpipes_map_gif(
        self,
        parameter="dust:Av",
        run_name=None,
        lazy=False,
        facecolor="white",
        cache=None,
        n_samples=100,
        weight_mass_sfr=True,
        logmap=False,
        savetype="temp",
        path="temp",
        cmap="magma",
    ):
        if (
            not hasattr(self, "sed_fitting_table")
            or "bagpipes" not in self.sed_fitting_table.keys()
        ):
            print("No bagpipes results found.")
            return None

        if run_name is None:
            run_name = list(self.sed_fitting_table["bagpipes"].keys())
            if len(run_name) > 1:
                raise Exception("Multiple runs found, please specify run_name")
            else:
                run_name = run_name[0]

        map = self.pixedfit_map

        values = list(np.unique(map[~np.isnan(map)]))
        values = [int(val) for val in values if val != 0.0]

        map_3d = np.zeros((n_samples, map.shape[0], map.shape[1]))

        objects = self.load_pipes_object(
            run_name, values, cache=cache, get_advanced_quantities=False
        )
        if not lazy:
            for pos, rbin in enumerate(values):
                obj = objects[pos]
                samples = obj.plot_pdf(None, parameter, return_samples=True)
                draws = np.random.choice(samples, n_samples)
                # For all pixels in map that are equal to bin, set the 1st dimension of map_3d to the draw
                mask = map == rbin
                mask_x, mask_y = np.where(mask)
                # Generate random draws
                draws = np.random.choice(samples, n_samples)
                for pos, draw in enumerate(draws):
                    map_3d[pos, mask_x, mask_y] = draw

        map_3d[:, map == 0] = np.nan
        map_3d[:, map == 0.0] = np.nan

        if parameter in ["stellar_mass", "sfr"]:
            density_map = True
            density = "$kpc^{-2}$"
            ref_band = {"stellar_mass": "F444W", "sfr": "1500A"}
            if weight_mass_sfr:
                weight_by_band = ref_band[parameter]
            else:
                weight_by_band = False
            if parameter == "stellar_mass":
                map_3d = 10**map_3d
                # print(map_3d[0, 32, 32])
        else:
            density_map = False
            density = ""
            weight_by_band = False

        if logmap:
            log = "$log_{10}$"
        else:
            log = ""
        unit = self.param_unit(parameter.split(":")[-1])
        if unit != u.dimensionless_unscaled:
            unit = f"[{log}{unit:latex}{density}]"
        else:
            unit = ""

        cbar_label = f'{parameter.replace("_", " ")} {unit}'
        # print('density_map', density_map)

        gif = self.make_animation(
            map_3d,
            draw_random=False,
            facecolor=facecolor,
            n_draws=n_samples,
            html=True if savetype == "html" else False,
            save=True if savetype == "gif" else False,
            cbar_label=cbar_label,
            density_map=density_map,
            logmap=logmap,
            weight_by_band=weight_by_band,
            cmap=cmap,
        )

        if path == "temp":
            import tempfile

            path = tempfile.mktemp(suffix=".gif")
            gif.save(path, writer="pillow", fps=60)
            return path

        return gif

    def eazy_fit_measured_photometry(
        self,
        phot_name,
        override_psf_type=None,
        override_binmap_type=None,
        overwrite=False,
        n_proc=4,
        min_percentage_err=0.05,
        template_name="fsps_larson",
        template_dir="/nvme/scratch/work/tharvey/EAZY/inputs/scripts/",
        meta_details=None,
        save_tempfilt=True,
        load_tempfilt=True,
        save_tempfilt_path="internal",
        tempfilt=None,
        update_meta_properties=False,
        exclude_bands=[],
    ):
        """
        Wrapper function for fit_eazy_photometry. Provides either
        total fluxes, aperture fluxes or fluxes from photometry_table
        """
        if override_psf_type is not None:
            self.use_psf_type = override_psf_type
        if override_binmap_type is not None:
            self.use_binmap_type = override_binmap_type

        arr = self.provide_bagpipes_phot(
            phot_name, exclude_bands=exclude_bands
        )
        fluxes = arr[:, 0] * u.uJy
        flux_errs = arr[:, 1] * u.uJy

        name = f"{phot_name}_{template_name}_{min_percentage_err}"

        if len(exclude_bands) > 0:
            name += f"_exclude_{exclude_bands}"

        if (
            self.interactive_outputs is not None
            and name.replace(".", "_") in list(self.interactive_outputs.keys())
            and not overwrite
        ):
            print(
                f"Output {name} already exists. Use overwrite=True to recompute."
            )
            return

        ez = self.fit_eazy_photometry(
            fluxes,
            flux_errs,
            n_proc=n_proc,
            template_name=template_name,
            template_dir=template_dir,
            meta_details=meta_details,
            save_tempfilt=save_tempfilt,
            load_tempfilt=load_tempfilt,
            save_tempfilt_path=save_tempfilt_path,
            tempfilt=tempfilt,
            exclude_bands=exclude_bands,
        )

        meta_dict = self.save_eazy_outputs(
            ez, [name], [fluxes], [flux_errs], save_txt=False
        )

        if update_meta_properties:
            self.meta_properties[f"zbest_{template_name}_zfree"] = meta_dict[
                "z_best"
            ]
            self.meta_properties[f"zbest_16_{template_name}_zfree"] = (
                meta_dict["z16"]
            )
            self.meta_properties[f"zbest_84_{template_name}_zfree"] = (
                meta_dict["z84"]
            )
            self.meta_properties[f"chi2_best_{template_name}_zfree"] = (
                meta_dict["chi2"]
            )
            self.dump_to_h5()

        return ez

    def fit_eazy_photometry(
        self,
        fluxes,
        flux_errs,
        n_proc=4,
        template_name="fsps_larson",
        template_dir="/nvme/scratch/work/tharvey/EAZY/inputs/scripts/",
        meta_details=None,
        save_tempfilt=True,
        load_tempfilt=True,
        save_tempfilt_path="internal",
        tempfilt=None,
        exclude_bands=[],
        z_min=0.01,
        z_max=20.0,
        z_step=0.01,
    ):
        import eazy
        from .eazy.eazy_config import make_params, filter_codes

        bands = [band for band in self.bands if band not in exclude_bands]
        eazy_path = os.getenv("EAZYCODE", "")
        if not os.path.exists(eazy_path):
            raise ValueError(
                "EAZYCODE environment variable not set or invalid"
            )

        if save_tempfilt_path == "internal":
            save_tempfilt_path = os.path.join(
                os.path.dirname(__file__), "eazy/tempfilt/"
            )

        if type(fluxes) == list:
            fluxes = np.array(fluxes)
        if type(flux_errs) == list:
            flux_errs = np.array(flux_errs)

        # Check that last dimension of fluxes and flux_errs is the same and equal to the number of bands
        assert (
            fluxes.shape[-1] == flux_errs.shape[-1]
        ), "Fluxes and flux errors must have the same number of bands"
        assert (
            fluxes.shape[-1] == len(self.bands) - len(exclude_bands)
        ), f"Fluxes and flux errors must have the same number of bands as the number of bands in the survey. Expected {len(self.bands)} bands, got {fluxes.shape[-1]} bands"

        if type(fluxes) is u.Quantity:
            unit = fluxes.unit
        elif type(fluxes[0]) is u.Quantity:
            unit = fluxes[0].unit
        elif type(fluxes[0][0]) is u.Quantity:
            unit = fluxes[0][0].unit
        else:
            print(fluxes, type(fluxes))
            raise ValueError(
                "Fluxes must be a Quantity object or a list of Quantity objects"
            )

        flux = np.atleast_2d(fluxes)
        flux_err = np.atleast_2d(flux_errs)

        # Create a temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create EAZY files
            # Make input catalogue - IDs - numer of bands - then Flux Error flux Error for each band. Column names should be 'F{filter_codes[band]', 'E{filter_codes[band]}' etc.

            # Create a temporary catalogue
            cat = Table()
            cat["id"] = np.arange(len(flux))
            for i, band in enumerate(bands):
                cat[f"F{filter_codes[band]}"] = flux[:, i]
                cat[f"E{filter_codes[band]}"] = flux_err[:, i]

            cat_path = f"{tmpdir}/eazy_input.fits"

            # make empty fake translate file
            translate_file = f"{tmpdir}/translate.dat"
            with open(translate_file, "w") as f:
                f.write("")

            cat.write(cat_path, overwrite=True)

            params = make_params(
                cat_path,
                tmpdir,
                template_name="fsps_larson",
                template_file=None,
                z_step=z_step,
                z_min=z_min,
                z_max=z_max,
                fix_zspec=False,
                cat_flux_unit=unit,
                template_dir=template_dir,
            )

            print(params)

            if load_tempfilt:
                if tempfilt is None:
                    import pickle

                    tempfilt_path = f"{save_tempfilt_path}/{self.survey}_{template_name}_{z_min}_{z_max}_{z_step}"

                    if exclude_bands:
                        tempfilt_path += f"_exclude_{'_'.join(exclude_bands)}"
                    tempfilt_path += "_tempfilt.pkl"

                    if os.path.exists(tempfilt_path):
                        with open(tempfilt_path, "rb") as f:
                            tempfilt = pickle.load(f)
                        print("Loading tempfilt from file:", tempfilt_path)
                    print(tempfilt_path)
                else:
                    print("Loading tempfilt from data")

            ez = eazy.photoz.PhotoZ(
                param_file=None,
                translate_file=translate_file,
                zeropoint_file=None,
                params=params,
                load_prior=False,
                load_products=False,
                n_proc=n_proc,
                tempfilt=tempfilt,
            )

            if save_tempfilt:
                print(f"Saving tempfilt to {save_tempfilt_path}")
                tempfilt = ez.tempfilt.tempfilt
                if not os.path.exists(save_tempfilt_path):
                    os.makedirs(save_tempfilt_path)

                # pickle tempfilt
                import pickle

                with open(
                    tempfilt_path,
                    "wb",
                ) as f:
                    pickle.dump(ez.tempfilt, f)

            ez.fit_catalog(
                n_proc=n_proc, get_best_fit=True, prior=False, beta_prior=False
            )

            return ez

    def _recalculate_bagpipes_wavelength_array(
        self,
        bands=None,
        bagpipes_filter_dir=bagpipes_filter_dir,
        use_bpass=False,
    ):
        if bands is None:
            bands = self.bands

        paths = [
            glob.glob(f"{bagpipes_filter_dir}/*{band}*")[0] for band in bands
        ]
        if use_bpass:
            from bagpipes import config_bpass as config
        else:
            from bagpipes import config

        from bagpipes.filters import filter_set

        ft = filter_set(paths)
        min_wav = ft.min_phot_wav
        max_wav = ft.max_phot_wav
        max_z = config.max_redshift

        max_wavs = [(min_wav / (1.0 + max_z)), 1.01 * max_wav, 10**8]

        x = [1.0]

        R = [config.R_other, config.R_phot, config.R_other]

        for i in range(len(R)):
            if i == len(R) - 1 or R[i] > R[i + 1]:
                while x[-1] < max_wavs[i]:
                    x.append(x[-1] * (1.0 + 0.5 / R[i]))

            else:
                while x[-1] * (1.0 + 0.5 / R[i]) < max_wavs[i]:
                    x.append(x[-1] * (1.0 + 0.5 / R[i]))

        wav = np.array(x)

        return wav

    def update_internal_redshift(self, z_override):
        if type(z_override) in [int, float]:
            self.redshift = z_override

        if type(z_override) is str:
            # Could refer to Bagpipes result.
            # Or EAZY result.
            raise NotImplementedError

    def save_eazy_outputs(
        self, ez, regions, fluxes, flux_errs, save_txt=False
    ):
        from regions import PolygonPixelRegion

        for pos, (region, flux, flux_err) in enumerate(
            zip(regions, fluxes, flux_errs)
        ):
            data = ez.show_fit(
                pos,
                id_is_idx=True,
                show_components=False,
                show_prior=False,
                logpz=False,
                get_spec=True,
                show_fnu=1,
            )

            z16, z50, z84 = ez.pz_percentiles([16, 50, 84])[0]

            meta_dict = {
                "id_phot": data["id"],
                "z_best": data["z"],
                "chi2": data["chi2"],
                "z16": z16,
                "z50": z50,
                "z84": z84,
                "flux_unit": data["flux_unit"],
                "wave_unit": data["wave_unit"],
                "templates_file": ez.param["TEMPLATES_FILE"],
                "z_min": ez.param["Z_MIN"],
                "z_max": ez.param["Z_MAX"],
                "z_step": ez.param["Z_STEP"],
                "region": region.serialize(format="ds9")
                if type(region) is not str
                else region,
            }

            model_flux = data["templf"]
            model_lam = data["templz"]

            model_flux_unit = u.Unit(data["flux_unit"])
            model_lam_unit = u.Unit(data["wave_unit"])

            # Stack the model fluxes and wavelengths
            output = np.vstack((model_lam, model_flux)).T

            model_flux *= model_flux_unit
            model_lam *= model_lam_unit

            p_z = 10 ** ez.lnp[pos]
            z = ez.zgrid

            output_z = np.vstack((z, p_z)).T
            # generate unique ID based on region coordinates
            if type(region) is str:
                region_id = region
            elif type(region) is not PolygonPixelRegion:
                region_id = f"{region.center.x}_{region.center.y}"
            else:
                vertices = region.vertices.xy
                region_id = f"{np.mean(vertices[0])}_{np.mean(vertices[1])}"

            region_id = region_id.replace(".", "_")

            self.add_to_h5(
                output,
                f"interactive_outputs/{region_id}/",
                "eazy_fit",
                overwrite=True,
                meta=meta_dict,
            )
            self.add_to_h5(
                output_z,
                f"interactive_outputs/{region_id}/",
                "p_z",
                overwrite=True,
            )
            self.add_to_h5(
                flux,
                f"interactive_outputs/{region_id}/",
                "input_flux",
                overwrite=True,
            )
            self.add_to_h5(
                flux_err,
                f"interactive_outputs/{region_id}/",
                "input_flux_err",
                overwrite=True,
            )

            if (
                not hasattr(self, "interactive_outputs")
                or self.interactive_outputs is None
            ):
                self.interactive_outputs = {}

            self.interactive_outputs[region_id] = {
                "eazy_fit": output,
                "p_z": output_z,
                "input_flux": flux,
                "input_flux_err": flux_err,
                "meta": meta_dict,
            }

            if save_txt:
                if not os.path.exists(f"{resolved_galaxy_dir}/eazy_outputs"):
                    os.makedirs(f"{resolved_galaxy_dir}/eazy_outputs")

                print(f"{resolved_galaxy_dir}/eazy_outputs/")

                # Make an output in um and mJy
                output = np.vstack(
                    (model_lam.to(u.um).value, model_flux.to(u.mJy).value)
                ).T

                np.savetxt(
                    f"{resolved_galaxy_dir}/eazy_outputs/{self.survey}_{self.galaxy_id}_{region_id}_eazy_fit.txt",
                    output,
                )
                np.savetxt(
                    f"{resolved_galaxy_dir}/eazy_outputs/{self.survey}_{self.galaxy_id}_{region_id}_p_z.txt",
                    output_z,
                )
                np.savetxt(
                    f"{resolved_galaxy_dir}/eazy_outputs/{self.survey}_{self.galaxy_id}_{region_id}_input_flux.txt",
                    flux.to(u.uJy).value,
                )
                np.savetxt(
                    f"{resolved_galaxy_dir}/eazy_outputs/{self.survey}_{self.galaxy_id}_{region_id}_input_flux_err.txt",
                    flux_err.to(u.uJy).value,
                )
                # save .reg
                region.write(
                    f"{resolved_galaxy_dir}/eazy_outputs/{self.survey}_{self.galaxy_id}_{region_id}.reg",
                    format="ds9",
                )

        return meta_dict
        # For each region, save the input fluxes and flux errors, best fit redshift and SED/p(z) in the .h5 file

    def plot_eazy_fit(
        self,
        ez,
        id,
        ax_sed=None,
        ax_pz=None,
        fig=None,
        show=True,
        color="black",
        lw=1,
        zorder=5,
        wav_units=u.um,
        flux_units=u.ABmag,
        label=None,
    ):
        if fig is None:
            fig = plt.figure(
                figsize=(6, 3), constrained_layout=True, facecolor="white"
            )
        if ax_sed is None:
            ax = fig.add_subplot(111)

        idx = np.where(ez.OBJID == id)[0][0]
        if ax_pz is not None:
            p_z = 10 ** ez.lnp[idx]
            z = ez.zgrid
            ax_pz.plot(z, p_z / np.max(p_z), color=color, lw=lw, zorder=zorder)

            if ax_pz.get_xlim()[0] < 0.1 and ax_pz.get_xlim()[1] > 18:
                # x-axis range to 5, 95 percentile of p(z)
                percentiles = np.percentile(p_z, [5, 95])
                lower = z[np.where(p_z > percentiles[0])[0][0]]
                upper = z[np.where(p_z > percentiles[1])[0][0]]
                if abs(upper - lower) < 1.5:
                    upper += 0.75
                    lower -= 0.75

                print("setting pz limits", lower, upper)
                ax_pz.set_xlim(lower, upper)

        data = ez.show_fit(
            id,
            id_is_idx=False,
            show_components=False,
            show_prior=False,
            logpz=False,
            get_spec=True,
            show_fnu=1,
        )

        id_phot = data["id"]
        z_best = data["z"]
        chi2 = data["chi2"]
        flux_unit = data["flux_unit"]
        wav_unit = data["wave_unit"]
        model_lam = data["templz"] * wav_unit
        model_flux = data["templf"] * flux_unit

        model_flux = model_flux.to(
            flux_units, equivalencies=u.spectral_density(model_lam)
        )

        if label is True:
            label = (
                rf"Region ${id_phot} \ (z={z_best:.2f} \ \chi^2={chi2:.2f})$"
            )
        else:
            label = ""

        ax_sed.plot(
            model_lam.to(wav_units),
            model_flux,
            color=color,
            lw=lw,
            zorder=zorder,
            label=label,
        )

        return data

    def plot_bagpipes_results(
        self,
        run_name=None,
        parameters=[
            "bin_map",
            "stellar_mass",
            "sfr",
            "dust:Av",
            "chisq_phot-",
            "UV_colour",
        ],
        reload_from_cat=False,
        save=False,
        facecolor="white",
        max_on_row=4,
        weight_mass_sfr=True,
        norm="linear",
        total_params=[],
    ):
        if (
            not hasattr(self, "sed_fitting_table")
            or "bagpipes" not in self.sed_fitting_table.keys()
        ):
            print("No bagpipes results found.")
            return None

        if run_name is None:
            run_name = list(self.sed_fitting_table["bagpipes"].keys())
            if len(run_name) > 1:
                raise Exception("Multiple runs found, please specify run_name")
            else:
                run_name = run_name[0]
        cmaps = [
            "magma",
            "RdYlBu",
            "cmr.ember",
            "cmr.cosmic",
            "cmr.lilac",
            "cmr.eclipse",
            "cmr.sapphire",
            "cmr.dusk",
            "cmr.emerald",
        ]
        if (
            not hasattr(self, "sed_fitting_table")
            or "bagpipes" not in self.sed_fitting_table.keys()
            or run_name not in self.sed_fitting_table["bagpipes"].keys()
            or reload_from_cat
        ):
            self.load_bagpipes_results(run_name)

        if not hasattr(self, "pixedfit_map"):
            raise Exception("Need to run pixedfit_binning first")
        # If it still isn't there, return None
        if (
            not hasattr(self, "sed_fitting_table")
            or "bagpipes" not in self.sed_fitting_table.keys()
            or run_name not in self.sed_fitting_table["bagpipes"].keys()
        ):
            return None

        table = self.sed_fitting_table["bagpipes"][run_name]

        # fig, axes = plt.subplots(1, len(parameters), figsize=(4*len(parameters), 4),
        #  constrained_layout=True, facecolor=facecolor)
        fig, axes = plt.subplots(
            len(parameters) // max_on_row + 1,
            max_on_row,
            figsize=(
                2.5 * max_on_row,
                2.5 * (len(parameters) // max_on_row + 1),
            ),
            constrained_layout=True,
            facecolor=facecolor,
            sharex=True,
            sharey=True,
            dpi=200,
        )
        # add gap between rows using get_layout_engine
        fig.get_layout_engine().set(h_pad=4 / 72, hspace=0.1)

        axes = axes.flatten()
        # Remove empty axes
        for i in range(len(parameters), len(axes)):
            fig.delaxes(axes[i])

        redshift = self.sed_fitting_table["bagpipes"][run_name][
            "input_redshift"
        ][0]

        for i, param in enumerate(parameters):
            ax_divider = make_axes_locatable(axes[i])
            cax = ax_divider.append_axes("top", size="5%", pad="2%")

            if param == "bin_map":
                map = copy.copy(self.pixedfit_map)
                map[map == 0] = np.nan
                log = ""
            else:
                param_name = (
                    f"{param[:-1]}" if param.endswith("-") else f"{param}_50"
                )
                map = self.convert_table_to_map(
                    table,
                    "#ID",
                    param_name,
                    self.pixedfit_map,
                    remove_log10=param.startswith("stellar_mass"),
                )
                log = ""

            if param in [
                "stellar_mass",
                "sfr",
                "stellar_mass_10myr",
                "sfr_10myr",
            ]:
                ref_band = {
                    "stellar_mass": "F444W",
                    "sfr": "1500A",
                    "stellar_mass_10myr": "F444W",
                    "sfr_10myr": "1500A",
                }
                # print(param, np.nanmin(map), np.nanmax(map))

                if weight_mass_sfr:
                    weight = ref_band[param]
                else:
                    weight = False

                map = self.map_to_density_map(
                    map, redshift=redshift, weight_by_band=weight, logmap=True
                )
                # map = self.map_to_density_map(map, redshift = redshift, logmap = True)
                log = r"$\log_{10}$ "
                param = f"{param}_density"

            if norm == "log":
                norm = LogNorm(vmin=np.nanmin(map), vmax=np.nanmax(map))
            else:
                norm = Normalize(vmin=np.nanmin(map), vmax=np.nanmax(map))

            gunit = self.param_unit(param.split(":")[-1])
            unit = (
                f" ({log}{gunit:latex})"
                if gunit != u.dimensionless_unscaled
                else ""
            )
            param_str = param.replace("_", r"\ ")

            for pos, total_param in enumerate(total_params):
                if total_param in table["#ID"]:
                    value = table[table["#ID"] == total_param][param_name]
                    if param_name.endswith("_50"):
                        err_up = (
                            table[table["#ID"] == total_param][
                                param_name.replace("50", "84")
                            ]
                            - value
                        )
                        err_down = (
                            value
                            - table[table["#ID"] == total_param][
                                param_name.replace("50", "16")
                            ]
                        )

                        berr = (
                            f"$^{{+{err_up[0]:.2f}}}_{{-{err_down[0]:.2f}}}$"
                        )
                    else:
                        berr = ""

                    if 0 < norm(value) < 1:
                        c_map = plt.cm.get_cmap(cmaps[i])
                        color = c_map(norm(value))
                    else:
                        color = "black"

                    axes[i].text(
                        0.5,
                        0.03 + 0.1 * pos,
                        f"{total_param}:{value[0]:.2f}{berr} {unit}",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                        transform=axes[i].transAxes,
                        color=color,
                        path_effects=[
                            PathEffects.withStroke(
                                linewidth=0.2, foreground="black"
                            )
                        ],
                    )

            # Create actual normalisation

            mappable = axes[i].imshow(
                map,
                origin="lower",
                interpolation="none",
                cmap=cmaps[i],
                norm=norm,
            )
            cbar = fig.colorbar(mappable, cax=cax, orientation="horizontal")

            # ensure cbar is using ScalarFormatter
            cbar.ax.xaxis.set_major_formatter(ScalarFormatter())
            cbar.ax.xaxis.set_minor_formatter(ScalarFormatter())

            cbar.set_label(
                rf"$\rm{{{param_str}}}${unit}", labelpad=10, fontsize=8
            )
            # Change cbar scale number size

            cbar.ax.xaxis.get_offset_text().set_fontsize(0)
            cbar.ax.xaxis.set_ticks_position("top")
            cbar.ax.xaxis.set_tick_params(labelsize=6, which="both")
            cbar.ax.xaxis.set_label_position("top")
            # disable xtick labels
            # cax.set_xticklabels([])
            # Fix colorbar tick size and positioning if log
            if norm == "log":
                # Generate reasonable ticks
                ticks = np.logspace(
                    np.log10(np.nanmin(map)), np.log10(np.nanmax(map)), num=5
                )
                cbar.set_ticks(ticks)
                cbar.ax.xaxis.set_major_formatter(ScalarFormatter())
                cbar.ax.xaxis.set_minor_formatter(ScalarFormatter())
                cbar.ax.xaxis.set_tick_params(labelsize=6, which="both")
                cbar.ax.xaxis.set_label_position("top")
                cbar.ax.xaxis.set_ticks_position("top")
                cbar.update_ticks()

        if save:
            fig.savefig(
                f"{resolved_galaxy_dir}/{run_name}_maps.png",
                dpi=300,
                bbox_inches="tight",
            )
        return fig

    def map_to_density_map(
        self,
        map,
        cosmo=FlatLambdaCDM(H0=70, Om0=0.3),
        redshift=None,
        logmap=False,
        weight_by_band=False,
        psf_type="star_stack",
        binmap_type="pixedfit",
    ):
        pixel_scale = self.im_pixel_scales[self.bands[0]]
        density_map = copy.copy(map)
        if binmap_type == "pixedfit":
            pixel_map = self.pixedfit_map
        else:
            raise Exception("Only pixedfit binning is supported for now")

        for gal_id in np.unique(pixel_map):
            # print(id)
            if gal_id == 0:
                continue
            mask = pixel_map == gal_id

            if redshift is None:
                z = self.redshift
            else:
                z = redshift

            re_as = pixel_scale
            d_A = cosmo.angular_diameter_distance(
                z
            )  # Angular diameter distance in Mpc
            pix_kpc = (re_as * d_A).to(
                u.kpc, u.dimensionless_angles()
            )  # re of galaxy in kpc
            pix_area = (
                np.sum(mask) * pix_kpc**2
            )  # Area of pixels with id in kpc^2
            # Check if all values in map[mask] are equal
            if len(np.unique(map[mask])) > 1:
                raise Exception(f"This was supposed to be equal {gal_id}")

            if weight_by_band:
                if weight_by_band.endswith("A"):
                    # Calculate which band is closest to the wavelength
                    wav = float(weight_by_band[:-1]) * u.AA
                    # Convert to observed frame
                    obs_wav = wav * (1 + z)

                    self.get_filter_wavs()

                    fmask = [
                        self.filter_wavs[band].to(u.AA).value > obs_wav.value
                        for band in self.bands
                    ]
                    fmask = np.array(fmask, dtype=bool)

                    pos = np.argwhere(fmask)[0][0]
                    band = self.bands[pos]
                    # print(f'Using band {band} at wavelength
                    # {self.filter_wavs[band].to(u.AA)} for weighting at {obs_wav} (rest frame {wav})')
                else:
                    band = weight_by_band
                if self.use_psf_type:
                    psf_type = self.use_psf_type

                data_band = self.psf_matched_data[psf_type][band]
                norm = np.sum(data_band[mask])

                x, y = np.where(mask)
                for xi, yi in zip(x, y):
                    value = map[xi, yi]
                    # print(mask[xi, yi], data_band[xi, yi] / norm)
                    weighted_val = value * data_band[xi, yi] / norm
                    weighted_val_area = weighted_val / pix_kpc**2

                    if logmap:
                        weighted_val_area = np.log10(weighted_val_area.value)
                    if type(weighted_val_area) in [
                        u.Quantity,
                        u.Magnitude,
                        Masked(u.Quantity),
                    ]:
                        weighted_val_area = weighted_val_area.value
                    density_map[xi, yi] = weighted_val_area
                # assert np.sum(density_map[mask]) == value,
                # f'overall sum should still correspond, {np.sum(density_map[mask])} != {value}'
            else:
                value = np.unique(map[mask]) / pix_area
                if type(value) in [
                    u.Quantity,
                    u.Magnitude,
                    Masked(u.Quantity),
                ]:
                    value = value.value

                if logmap:
                    value = np.log10(value)

                density_map[mask] = value

        return density_map

    def load_pipes_object(
        self,
        run_name,
        rbin,
        run_dir="pipes/",
        bagpipes_filter_dir=bagpipes_filter_dir,
        cache=None,
        plotpipes_dir="bagpipes/",
        get_advanced_quantities=True,
        n_cores=1,
        psf_type="star_stack",
        binmap_type="pixedfit",
    ):
        plotpipes_dir = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), plotpipes_dir
        )

        run_dir = os.path.join(
            os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
            ),
            run_dir,
        )

        if type(rbin) not in [list, np.ndarray]:
            single = True
            rbin = [rbin]
        else:
            single = False

        # If name can be a number, only allow integers
        for i in range(len(rbin)):
            try:
                rbin[i] = int(rbin[i])
            except ValueError:
                pass

        output = {}
        from .bagpipes.plotpipes import PipesFit, PipesFitNoLoad

        found = False
        if cache is not None:
            for b in rbin:
                if b in cache.keys():
                    output[b] = cache[b]

        if run_name in self.internal_bagpipes_cache.keys():
            for b in rbin:
                if b in self.internal_bagpipes_cache[run_name].keys():
                    output[b] = self.internal_bagpipes_cache[run_name][b]
        else:
            self.internal_bagpipes_cache[run_name] = {}

        still_to_load = [b for b in rbin if b not in output.keys()]

        if len(still_to_load) > 0:
            params_dict = {
                "galaxy_id": None,
                "field": self.survey,
                "h5_path": None,
                "pipes_path": run_dir,
                "catalog": None,
                "overall_field": None,
                "load_spectrum": False,
                "filter_path": bagpipes_filter_dir,
                "ID_col": "NUMBER",
                "field_col": "field",
                "catalogue_flux_unit": u.MJy / u.sr,
                "bands": self.bands,
                "data_func": self.provide_bagpipes_phot,
                "get_advanced_quantities": get_advanced_quantities,
            }

            params_b = []
            for load_p in still_to_load:
                print(f"Loading {run_name} {load_p}")
                param = copy.copy(params_dict)
                param["galaxy_id"] = load_p
                param["h5_path"] = (
                    f"{run_dir}/posterior/{run_name}/{self.survey}/{self.galaxy_id}/{load_p}.h5"
                )
                # Need to only use bands that were run with.
                #
                table = self.photometry_table[psf_type][binmap_type]
                mask = table["ID"] == load_p

                bands_gal = self.provide_bagpipes_phot(
                    str(load_p), return_bands=True
                )
                param["bands"] = bands_gal
                params_b.append(param)

            pipes_objs = Parallel(n_jobs=n_cores)(
                delayed(PipesFitNoLoad)(**param) for param in params_b
            )

            for obj, b in zip(pipes_objs, still_to_load):
                output[b] = obj
                if cache is not None:
                    cache[b] = obj

                self.internal_bagpipes_cache[run_name][b] = obj

        # Set output as list in order of input
        pipes_obj = [output[b] for b in rbin]
        if get_advanced_quantities:
            for obj in pipes_obj:
                if not obj.has_advanced_quantities:
                    print("Adding advanced quantities")
                    obj.add_advanced_quantities()

        # Double check that all objects
        assert all(
            [pipesi.galaxy_id == rbin[i] for i, pipesi in enumerate(pipes_obj)]
        )

        if single:
            pipes_obj = pipes_obj[0]

        return pipes_obj

    def plot_bagpipes_component_comparison(
        self,
        parameter="stellar_mass",
        run_name=None,
        bins_to_show="all",
        save=False,
        run_dir="pipes/",
        facecolor="white",
        plotpipes_dir="pipes_scripts/",
        bagpipes_filter_dir=bagpipes_filter_dir,
        n_draws=10000,
        cache=None,
        colors=None,
        fig=None,
        axes=None,
    ):
        plotpipes_dir = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), plotpipes_dir
        )

        from bagpipes.plotting import hist1d

        if run_name is None:
            run_name = list(self.sed_fitting_table["bagpipes"].keys())
            if len(run_name) > 1:
                raise Exception("Multiple runs found, please specify run_name")
            else:
                run_name = run_name[0]

        if (
            not hasattr(self, "sed_fitting_table")
            or "bagpipes" not in self.sed_fitting_table.keys()
            or run_name not in self.sed_fitting_table["bagpipes"].keys()
        ):
            self.load_bagpipes_results(run_name)

        table = self.sed_fitting_table["bagpipes"][run_name]

        bins = []
        if bins_to_show == "all":
            show_all = True
            bins_to_show = table["#ID"]
            for rbin in bins_to_show:
                try:
                    rbin = float(rbin)
                except:
                    bins.append(rbin)

        else:
            show_all = False
            bins = bins_to_show
        if fig is None:
            fig, ax_samples = plt.subplots(
                1,
                1,
                figsize=(3, 3),
                constrained_layout=True,
                facecolor=facecolor,
            )
        else:
            ax_samples = axes

        if cache is None:
            cache = {}

        all_samples = []
        if colors is None:
            colors = mcm.get_cmap("cmr.guppy", len(bins))
            colors = {rbin: colors(i) for i, rbin in enumerate(bins)}
        if type(colors) is list:
            assert len(colors) == len(
                bins
            ), "Need to provide a color for each bin"
            colors = {rbin: colors[i] for i, rbin in enumerate(bins)}

        for rbin in bins_to_show:
            if rbin == "RESOLVED":
                continue
            try:
                pipes_obj = self.load_pipes_object(
                    run_name,
                    rbin,
                    run_dir=run_dir,
                    cache=cache,
                    plotpipes_dir=plotpipes_dir,
                )
            except FileNotFoundError:
                print(f"File not found for {run_name} {rbin} (comp_corr)")
                continue

            bin_number = True
            try:
                rbin = float(rbin)
            except:
                bin_number = False
            # This will need to change huh.
            if show_all:
                datta = ax_samples if not bin_number else None
                collors = colors[rbin] if not bin_number else "black"
                labbels = str(rbin) if not bin_number else None
                ret_samples = True
            else:
                datta = ax_samples
                collors = colors[rbin]
                labbels = str(rbin)
                ret_samples = False
            # print('parameter', parameter)
            samples = pipes_obj.plot_pdf(
                datta,
                parameter,
                return_samples=ret_samples,
                linelabel=labbels,
                colour=collors,
                norm_height=True,
            )

            if ret_samples:
                all_samples.append(samples)
        # Sum all samples

        if "all" in bins_to_show:
            all_samples = np.array(all_samples, dtype=object)

            # all_samples = all_samples[~np.isnan(all_samples)]
            print(f"Combining {len(all_samples)} samples for {parameter}")
            # all_samples = all_samples.T
            new_samples = np.zeros((n_draws, len(all_samples)))
            for i, samples in enumerate(all_samples):
                # samples = samples[~np.isnan(samples)]
                new_samples[:, i] = np.random.choice(samples, size=n_draws)

            sum_samples = np.log10(np.sum(10**new_samples, axis=1))
            # print(len(sum_samples))
            # Normalize height of histogram

            hist1d(
                sum_samples,
                ax_samples,
                smooth=True,
                color="black",
                percentiles=False,
                lw=1,
                alpha=1,
                fill_between=False,
                norm_height=True,
                label=r"$\Sigma$ Resolved",
            )

        # Fix xticks
        # ax_samples.set_xticks(np.arange(ax_samples.get_xlim()[0], ax_samples.get_xlim()[1], 0.5))
        ax_samples.legend(fontsize=6)

        gunit = self.param_unit(parameter.split(":")[-1])
        unit = f" ({gunit:latex})" if gunit != u.dimensionless_unscaled else ""
        param_str = parameter.replace("_", r" ")

        ax_samples.set_xlabel(f"{param_str} {unit}")
        # print(f"{param_str} {unit}")
        return fig, cache

    def get_total_resolved_property(
        self,
        run_name,
        property="stellar_mass",
        combine_type="sum",  # or "flatten"
        log=True,  # for mass
        sed_fitting_tool="bagpipes",
        n_draws=10000,
        return_quantiles=True,
        run_dir="pipes/",
        pipes_dir="pipes_scripts/",
        overwrite=False,
        n_cores=1,
        open_h5=True,
    ):
        table = self.sed_fitting_table[sed_fitting_tool][run_name]
        bins = []
        bins_to_show = table["#ID"]

        if return_quantiles:
            if property == "stellar_mass" and not overwrite:
                if (
                    self.resolved_mass is not None
                    and run_name in self.resolved_mass.keys()
                    and not overwrite
                ):
                    return self.resolved_mass[run_name]

            if property == "sfr" or property == "sfr_100myr" and not overwrite:
                if (
                    self.resolved_sfr_100myr is not None
                    and run_name in self.resolved_sfr_100myr.keys()
                    and not overwrite
                ):
                    return self.resolved_sfr_100myr[run_name]

            if property == "sfr_10myr" and not overwrite:
                if (
                    self.resolved_sfr_10myr is not None
                    and run_name in self.resolved_sfr_10myr.keys()
                    and not overwrite
                ):
                    return self.resolved_sfr_10myr[run_name]

        for rbin in bins_to_show:
            try:
                float(rbin)
                bins.append(rbin)
            except:
                pass

        all_samples = []

        if not open_h5:
            try:
                #

                pipes_objs = self.load_pipes_object(
                    run_name,
                    bins,
                    get_advanced_quantities=True,
                    run_dir=run_dir,
                    plotpipes_dir=pipes_dir,
                    n_cores=n_cores,
                )
            except FileNotFoundError:
                print(f"Files not found for {run_name} {bins}")
                return None

            # print(pipes_objs)

            for obj in pipes_objs:
                samples = obj.plot_pdf(None, property, return_samples=True)

                all_samples.append(samples)
        else:
            # Manually open .h5 and read in samples. Should be faster.
            for rbin in bins:
                h5_path = f"{run_dir}/posterior/{run_name}/{self.survey}/{self.galaxy_id}/{rbin}.h5"
                with h5.File(h5_path, "r") as f:
                    if property in f["basic_quantities"].keys():
                        samples = f["basic_quantities"][property][:]
                    elif property in f["advanced_quantities"].keys():
                        samples = f["advanced_quantities"][property][:]
                    else:
                        for key in f["basic_quantities"].keys():
                            if key.endswith(":massformed"):
                                sfh = key.split(":")[0]
                                break
                        new_prop_name = f"{sfh}:key"
                        if new_prop_name in f["advanced_quantities"].keys():
                            samples = f["advanced_quantities"][new_prop_name][
                                :
                            ]
                        else:
                            raise Exception(
                                f"Property {property} not found in {h5_path}. Available properties are {f['basic_quantities'].keys()} and {f['advanced_quantities'].keys()}"
                            )
                    all_samples.append(samples)

        all_samples = np.array(all_samples, dtype=object)
        new_samples = np.zeros((n_draws, len(all_samples)))

        for i, samples in enumerate(all_samples):
            # samples = samples[~np.isnan(samples)]
            new_samples[:, i] = np.random.choice(samples, size=n_draws)

        if log:
            new_samples = 10**new_samples

        if combine_type == "sum":
            # sum_samples = np.log10(np.sum(10**new_samples, axis=1))
            sum_samples = np.sum(new_samples, axis=1)
        elif combine_type == "flatten":
            sum_samples = new_samples.flatten()

        if log:
            sum_samples = np.log10(sum_samples)

        percentiles = np.percentile(sum_samples, [16, 50, 84])

        if property == "stellar_mass":
            if self.resolved_mass is None:
                self.resolved_mass = {}

            self.resolved_mass[run_name] = percentiles

            self.add_to_h5(
                percentiles, "resolved_mass", run_name, overwrite=True
            )

        elif property == "sfr" or property == "sfr_10myr":
            if property == "sfr":
                property = "sfr_100myr"
                if self.resolved_sfr_100myr is None:
                    self.resolved_sfr_100myr = {}
                self.resolved_sfr_100myr[run_name] = percentiles
            elif property == "sfr_10myr":
                if self.resolved_sfr_10myr is None:
                    self.resolved_sfr_10myr = {}
                self.resolved_sfr_10myr[run_name] = percentiles

            self.add_to_h5(
                percentiles, f"resolved_{property}", run_name, overwrite=True
            )

        if return_quantiles:
            return percentiles
        else:
            return sum_samples

    def make_animation(
        self,
        map_3d,
        draw_random=True,
        n_draws=10,
        facecolor="white",
        scale="linear",
        save=False,
        html=False,
        cbar_label=None,
        density_map=False,
        logmap=False,
        weight_by_band=False,
        path="temp",
        cmap="magma",
    ):
        map_3d = copy.copy(map_3d)

        # Check map has 3 dimensions
        if len(map_3d.shape) != 3:
            raise Exception("Map must have 3 dimensions")

        if draw_random:
            # Make n_draws maps of random indices of the
            # same shape as the 2nd and 3rd dimensions of map_3d

            N, M, _ = map_3d.shape

            # Create an array of random indices for each pixel
            random_indices = np.random.randint(0, N, size=(n_draws, M, M))

            # Create meshgrid for the M x M coordinates
            i, j = np.meshgrid(range(M), range(M), indexing="ij")

            # Use advanced indexing to select random pixels
            map = map_3d[random_indices, i, j]

            if density_map:
                # print(weight_by_band)
                map = self.map_to_density_map(
                    map,
                    redshift=self.redshift,
                    logmap=logmap,
                    weight_by_band=weight_by_band,
                )
            else:
                if logmap:
                    map = np.log10(map)

            new_map = map

        else:
            n_draws = map_3d.shape[0]
            if density_map:
                temp_map = np.zeros(
                    (n_draws, map_3d.shape[1], map_3d.shape[2])
                )
                for i in range(n_draws):
                    new_map = self.map_to_density_map(
                        map_3d[i],
                        redshift=self.redshift,
                        logmap=logmap,
                        weight_by_band=weight_by_band,
                    )
                    if type(new_map) in [
                        u.Quantity,
                        u.Magnitude,
                        Masked(u.Quantity),
                    ]:
                        new_map = new_map.value
                    temp_map[i] = new_map
                new_map = temp_map
            else:
                new_map = map_3d

        # Set all 0 values to nan
        new_map[new_map == 0] = np.nan
        new_map[new_map == 0.0] = np.nan

        if scale == "log":
            scale = LogNorm(vmin=np.nanmin(new_map), vmax=np.nanmax(new_map))
        elif scale == "linear":
            scale = Normalize(vmin=np.nanmin(new_map), vmax=np.nanmax(new_map))
        else:
            raise Exception("Scale must be log or linear")

        fig, ax = plt.subplots(
            1, 1, figsize=(4, 4), facecolor=facecolor, layout="tight"
        )

        # Create the image plot once
        im = ax.imshow(
            new_map[0],
            origin="lower",
            interpolation="none",
            cmap=cmap,
            norm=scale,
            animated=True,
        )
        cax = make_axes_locatable(ax).append_axes("top", size="5%", pad="2%")
        cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
        cbar.ax.xaxis.set_ticks_position("top")

        if cbar_label is not None:
            cbar.set_label(cbar_label, labelpad=10, fontsize=8)
            cbar.ax.xaxis.set_label_position("top")

        def animate(i):
            im.set_array(new_map[i])
            return [im]

        """
        def animate(i):
            ax.clear()
            ax.imshow(new_map[i], origin='lower', interpolation='none', cmap=cmap, norm=scale)
        """
        ani = FuncAnimation(
            fig, animate, frames=n_draws, interval=100, repeat=True, blit=True
        )

        # Stop colorbar getting cut off

        if save:
            ani.save(
                f"{resolved_galaxy_dir}/animation.gif", writer="pillow", fps=60
            )

        plt.close(fig)

        if html:
            return ani.to_jshtml()

        return ani

    def compose_bagpipes_pandas_table(
        self,
        show_individual_resolved=False,
        parameters_to_show=[
            "redshift",
            "stellar_mass",
            "sfr",
            "ssfr",
            "sfr_10myr",
            "dust:Av",
            "mass_weighted_age",
            "metallicity",
            "chisq_phot",
            "beta_C94",
            " M_UV",
        ],
        cumulative_parameters=[
            "stellar_mass",
            "sfr",
            "sfr_10myr",
            "m_UV",
            "M_UV",
        ],
        param_details={
            "stellar_mass": ["log"],
            "m_UV": ["abmag"],
            "M_UV": ["abmag"],
        },
    ):
        """
        # Create a pandas table from the bagpipes results showing key parameters
        only show for aperture and total by default
        """

        if (
            not hasattr(self, "sed_fitting_table")
            or "bagpipes" not in self.sed_fitting_table.keys()
        ):
            raise Exception("Need to run bagpipes first")

        import pandas as pd

        tables = self.sed_fitting_table["bagpipes"]

        # Create a pandas table
        df = pd.DataFrame()
        # Create each row -one for each bagpipes run
        for run_name in tables.keys():
            table = tables[run_name]
            # Create a dictionary for each row
            row = {}

            if not show_individual_resolved:
                # filter out numerical IDs
                mask = np.array([not i.isnumeric() for i in table["#ID"]])
                table_resolved = table[~mask]
                table = table[mask]

            if len(table) != 0:
                for param in parameters_to_show:
                    row["Run Name"] = [run_name] * len(table)

                    # stop ID being a byte string
                    row["ID"] = [str(i) for i in table["#ID"]]

                    if param in table.colnames:
                        row[param] = [table[param].data]
                    if f"{param}_50" in table.colnames:
                        # row[param] =
                        # For each item, format as HTML with <sup> and <sub> tags
                        err_up = (
                            table[f"{param}_84"].data
                            - table[f"{param}_50"].data
                        )
                        err_down = (
                            table[f"{param}_50"].data
                            - table[f"{param}_16"].data
                        )
                        value = table[f"{param}_50"].data

                        # row[param] = [f'{i:.2f}<sup>+{j:.2f}</sup><sub>-{k:.2f}</sub>' for i, j, k in zip(value, err_up, err_down)]

                        row[param] = [
                            f'<span>{i:.2f}<span style="display: inline-block; margin: -9em 0; vertical-align: -0.55em; line-height: 1.35em; font-size: 70%; text-align: left;">+{j:.2f}<br />-{k:.2f}</span></span>'
                            for i, j, k in zip(value, err_up, err_down)
                        ]

                        # row[f'{param}uerr'] = [err_up]
                        # row[f'{param}lerr'] = [err_down]

                    else:
                        row[param] = [np.nan] * len(table)

            row2 = {}

            if len(table_resolved) != 0:
                row2["Run Name"] = [run_name]
                row2["ID"] = ["RESOLVED"]
                # Sum up the resolved values for each parameter
                # If parameter is in cumulative_parameters, then add, otherwise take the median
                for param in parameters_to_show:
                    skip = False
                    if param in cumulative_parameters:
                        if param not in table_resolved.colnames:
                            param_name = f"{param}_50"
                        else:
                            param_name = param

                        if param_name not in table_resolved.colnames:
                            skip = True

                        if not skip:
                            if param in param_details.keys():
                                if "log" in param_details[param]:
                                    row2[param] = [
                                        f"{np.log10(np.sum(10**table_resolved[param_name].data)):.2f}"
                                    ]
                                elif "abmag" in param_details[param]:
                                    flux = (
                                        table_resolved[param_name].data
                                        * u.ABmag
                                    )
                                    sum = np.sum(flux.to(u.uJy))
                                    row2[param] = [
                                        f"{sum.to(u.ABmag).value:.2f}"
                                    ]
                            else:
                                row2[param] = [
                                    f"{np.sum(table_resolved[param_name].data):.2f}"
                                ]
                    else:
                        if param not in table_resolved.colnames:
                            param_name = f"{param}_50"
                        else:
                            param_name = param

                        if param_name not in table_resolved.colnames:
                            skip = True
                        if not skip:
                            row2[param] = [
                                f"{np.median(table_resolved[param_name].data):.2f}"
                            ]

                    if skip:
                        row2[param] = [np.nan]

                # Append the resolved row to the main table

            df = df._append(pd.DataFrame(row), ignore_index=True)

            if len(table_resolved) != 0:
                df = df._append(pd.DataFrame(row2), ignore_index=True)

        return df

    def scale_map_by_band(self, map, band):
        # Unused
        pix_map_band = self.self.psf_matched_data[binmap_type][band]

        bins = np.unique(self.pixedfit_map)
        for rbin in bins:
            mask = self.pixedfit_map == rbin
            map[mask] = map[mask] * pix_map_band[mask]

    def get_resolved_bagpipes_sed(
        self,
        run_name,
        run_dir="pipes/",
        plotpipes_dir="pipes_scripts/",
        overwrite=False,
        flux_units=u.uJy,
        load_h5=True,
    ):
        if (
            not hasattr(self, "sed_fitting_table")
            or "bagpipes" not in self.sed_fitting_table.keys()
        ):
            raise Exception("Need to run bagpipes first")

        if (
            self.resolved_sed is not None
            and run_name in self.resolved_sed.keys()
            and not overwrite
        ):
            # Check its 2D
            if len(self.resolved_sed[run_name].shape) == 2:
                return (
                    self.resolved_sed[run_name][:, 1],
                    self.resolved_sed[run_name][:, 0],
                )

        table = self.sed_fitting_table["bagpipes"][run_name]

        # Make cmap for map
        count = []
        for i, gal_id in enumerate(table["#ID"]):
            try:
                gal_id = int(gal_id)
                count.append(gal_id)
            except:
                pass

        if 0 in count:
            count.remove(0)

        if len(count) == 0:
            return (False, False)

        if not load_h5:
            try:
                # Load all
                pipes_objs = self.load_pipes_object(
                    run_name,
                    count,
                    run_dir=run_dir,
                    cache=None,
                    plotpipes_dir=plotpipes_dir,
                )
            except FileNotFoundError:
                print(f"File not found for {run_name} {self.galaxy_id}")
                return (False, False)
        else:
            fluxes = []
            wavs = []
            # Make dummy pipes_obj list to iterate over
            pipes_objs = [1] * len(count)
            # Manually open .h5 and read in samples. Should be faster.
            for rbin in count:
                h5_path = f"{run_dir}/posterior/{run_name}/{self.survey}/{self.galaxy_id}/{rbin}.h5"
                if not os.path.exists(h5_path):
                    print(
                        f"File not found for {run_name} {self.galaxy_id} {rbin}"
                    )
                    return (False, False)

                with h5.File(h5_path, "r") as f:
                    """ Fun bodging of SED into correct form"""
                    use_bpass = (
                        ast.literal_eval(f.attrs["config"])["type"] == "BPASS"
                    )
                    if "spectrum_full" in f["advanced_quantities"].keys():
                        data = f["advanced_quantities"]["spectrum_full"][:]
                        bands = self.provide_bagpipes_phot(
                            str(rbin), return_bands=True
                        )
                        wav = self._recalculate_bagpipes_wavelength_array(
                            bands=bands
                        )
                        # Data is 2D - check one of the dimensions is the same as the length of wav
                        if data.shape[1] != len(wav) and data.shape[0] != len(
                            wav
                        ):
                            raise Exception(
                                f"Data shape {data.shape} does not match wavelength shape {len(wav)}"
                            )

                        if "redshift" in f["basic_quantities"].keys():
                            redshift = np.median(
                                f["basic_quantities"]["redshift"][:]
                            )
                        else:
                            redshift = ast.literal_eval(
                                f.attrs["fit_instructions"]
                            )["redshift"]
                        wavs_aa = wav * (1.0 + redshift) * u.AA

                        spec_post = np.percentile(data, (16, 50, 84), axis=0).T

                        spec_post = spec_post.astype(
                            float
                        )  # fixes weird isfinite error

                        flux_lambda = (
                            spec_post * u.erg / (u.s * u.cm**2 * u.AA)
                        )

                        wavs_micron_3 = np.vstack(
                            [wavs_aa, wavs_aa, wavs_aa]
                        ).T

                        flux = flux_lambda.to(
                            flux_units,
                            equivalencies=u.spectral_density(wavs_micron_3),
                        ).value

                        fluxes.append(flux[:, 1])
                        wavs.append(wavs_aa.to(u.um).value)

                    else:
                        raise Exception(
                            f"Property spectrum_full not found in {h5_path}. Available properties are {f['advanced_quantities'].keys()}"
                        )

        # Get SED
        for pos, pipes_obj in enumerate(pipes_objs):
            # Check if a callabale object
            if pipes_obj != 1:
                wav, flux = pipes_obj.plot_best_fit(
                    None, wav_units=u.um, flux_units=u.uJy, return_flux=True
                )
            else:
                flux = fluxes[pos]
                wav = wavs[pos]

            # Clip the SED to the wavelength range 0 to 8 um
            flux = flux[(wav >= 0) & (wav < 8)]
            wav = wav[(wav >= 0) & (wav < 8)]

            # print(galaxy_id, np.min(wav), np.max(wav), len(wav))
            if pos == 0:
                total_flux = np.zeros_like(flux)
                total_wav = wav
            else:
                # assert all(
                #    wav == total_wav
                # ), f"Wavelengths do not match, {np.min(wav)} - {np.max(wav)} vs {np.min(total_wav)} - {np.max(total_wav)}"
                # switch to np.isclose
                assert np.allclose(
                    wav, total_wav, rtol=0, atol=0.003
                ), f"Wavelengths not within tolerance, {np.min(wav)} - {np.max(wav)} vs {np.min(total_wav)} - {np.max(total_wav)}"
            total_flux += flux

        out_array = np.zeros((len(total_flux), 2))
        out_array[:, 0] = total_wav
        out_array[:, 1] = total_flux

        self.add_to_h5(out_array, "resolved_sed", run_name, overwrite=True)

        if self.resolved_sed is None:
            self.resolved_sed = {}

        self.resolved_sed[run_name] = out_array

        return (total_flux, total_wav)

    def sep_process(
        self,
        forced_phot_band=["F277W", "F356W", "F444W"],
        detection_image=None,
        detection_rms_error=None,
        debug=False,
        phot_images="psf_matched",
        rms_err_images="psf_matched",
        conv=np.array(
            [
                [0.034673, 0.119131, 0.179633, 0.119131, 0.034673],
                [0.119131, 0.409323, 0.617200, 0.409323, 0.119131],
                [0.179633, 0.617200, 0.930649, 0.617200, 0.179633],
                [0.119131, 0.409323, 0.617200, 0.409323, 0.119131],
                [0.034673, 0.119131, 0.179633, 0.119131, 0.034673],
            ]
        ),
        return_results=False,
        update_h5=True,
        bands=None,
        override_psf_type=False,
        depth_file={
            "NIRCam": "/raid/scratch/work/austind/GALFIND_WORK/Depths/NIRCam/v11/JOF/old/0.32as/n_nearest/JOF_depths.ecsv",
            "ACS_WFC": "/raid/scratch/work/austind/GALFIND_WORK/Depths/ACS_WFC/v11/JOF/0.32as/n_nearest/JOF_depths.ecsv",
        },
        det_thresh=1.8,
        minarea=9,
        deblend_nthresh=32,
        deblend_cont=0.005,
        aper_radius=0.16 * u.arcsec,
        overwrite=False,
    ):
        to_add = []
        if not overwrite:
            if (self.aperture_dict is None) or (self.aperture_dict == {}):
                to_add.append("aperture")
            if (self.auto_photometry is None) or (self.auto_photometry == {}):
                to_add.append("auto")
            if (self.seg_imgs is None) or (self.seg_imgs == {}):
                to_add.append("seg")
            if self.det_data is None or self.det_data == {}:
                to_add.append("det_data")
            # total_photometry
            if self.total_photometry is None or self.total_photometry == {}:
                to_add.append("total_photometry")
        else:
            to_add = [
                "aperture",
                "auto",
                "seg",
                "det_data",
                "total_photometry",
            ]

        if to_add == []:
            print(
                f"All photometry already set, and overwrite = False. Skipping."
            )
            return

        print(f"Running for {to_add}")

        import sep_pjw as sep

        if override_psf_type:
            psf_type = override_psf_type
        else:
            psf_type = self.use_psf_type

        if bands is None:
            bands = copy.copy(self.bands)

        if phot_images == "psf_matched":
            phot_images = self.psf_matched_data[psf_type]
        elif phot_images == "phot":
            phot_images = self.phot_imgs
            phot_psfmatched = False

        if rms_err_images == "psf_matched":
            rms_err_images = self.psf_matched_rms_err[psf_type]
        elif rms_err_images == "phot":
            rms_err_images = self.rms_err_imgs
            phot_psfmatched = False

        for pos, key in enumerate(depth_file.keys()):
            table = Table.read(depth_file[key], format="ascii.ecsv")

            if pos == 0:
                main_table = table
            else:
                main_table = vstack([main_table, table])

        table = main_table

        # Select 'all' region column
        table = table[table["region"] == "all"]
        # band_depths:
        depths = {}
        for band in bands:
            row = table[table["band"] == band]
            assert len(row) == 1, f"Multiple rows found for {band}"
            data = row["median_depth"][0]
            depths[band] = data * u.ABmag
            depths[band] = depths[band].to(u.uJy)

        # Use sep.extract to get segmentation maps and measure equivalent fluxes.
        seg_imgs = {}

        # Detection band -

        detection_band = "+".join(forced_phot_band)
        # inverse variance weighted stack of mock_forced_phot_band

        inv_var_weights = {
            band: 1 / rms_err_images[band] ** 2 for band in forced_phot_band
        }

        # Calculate the weighted sum of images
        weighted_sum = np.sum(
            [
                phot_images[band] * inv_var_weights[band]
                for band in forced_phot_band
            ],
            axis=0,
        )

        # Calculate the sum of weights
        sum_of_weights = np.sum(
            [inv_var_weights[band] for band in forced_phot_band], axis=0
        )

        # Calculate the inverse variance weighted stack
        detection_image = weighted_sum / sum_of_weights

        # Calculate the error of the stack
        detection_rms_error = np.sqrt(1 / sum_of_weights)
        if debug:
            plt.imshow(
                detection_image / detection_rms_error,
                origin="lower",
                cmap="viridis",
            )
            plt.colorbar()
            plt.show()

            plt.imshow(
                detection_image, origin="lower", cmap="viridis", norm=LogNorm()
            )
            plt.title("Detection image")
            plt.show()

            plt.imshow(
                phot_images[forced_phot_band[0]],
                origin="lower",
                cmap="viridis",
                norm=LogNorm(),
            )
            plt.title(f"{forced_phot_band[0]}")
            plt.show()

        detection_objects, detection_segmap = sep.extract(
            detection_image,
            thresh=det_thresh,
            err=detection_rms_error,
            minarea=minarea,
            deblend_nthresh=deblend_nthresh,
            filter_kernel=conv,
            deblend_cont=deblend_cont,
            clean=True,
            clean_param=1.0,
            segmentation_map=True,
        )
        # Make measurements in other bands

        assert (
            len(detection_objects) == len(np.unique(detection_segmap)) - 1
        ), "Segmap should match detection object list."
        # Can probably do det_data here as well

        det_data = {}

        det_data["phot"] = detection_image
        det_data["rms_err"] = detection_rms_error
        det_data["seg"] = detection_segmap
        detect_seg_id = (
            detection_segmap[
                detection_segmap.shape[0] // 2,
                detection_segmap.shape[1] // 2,
            ]
            if len(np.unique(detection_segmap)) > 2
            else 1
        )  # Use center pixel if multiple else use 1
        if len(np.unique(detection_segmap)) == 1:
            detect_seg_id = 0
            raise Exception(
                f"No objects found in detection segmap for {self.galaxy_id}"
            )

        # Add detection image to phot_images

        bands.append(detection_band)

        # Plot the detection image and objects
        if debug:
            fig, ax = plt.subplots(
                nrows=1,
                ncols=2,
                figsize=(10, 5),
                constrained_layout=True,
                facecolor="white",
            )
            ax[0].imshow(
                detection_image, origin="lower", cmap="viridis", norm=LogNorm()
            )

            for i in range(len(detection_objects)):
                e = Ellipse(
                    xy=(detection_objects["x"][i], detection_objects["y"][i]),
                    width=6 * detection_objects["a"][i],
                    height=6 * detection_objects["b"][i],
                    angle=detection_objects["theta"][i] * 180.0 / np.pi,
                )
                e.set_facecolor("none")
                e.set_edgecolor("red")
                ax[0].add_artist(e)

            ax[1].imshow(detection_segmap, origin="lower", cmap="viridis")
            ax[0].set_title("Detection image")
            ax[1].set_title("Detection segmentation map")
            plt.savefig(f"{self.galaxy_id}_detection_seg.png")

        print(
            f"Found {len(detection_objects)} objects in detection band {detection_band}"
        )

        assert len(detection_objects) > 0, "No objects found in detection band"
        # assert len(detection_objects) == 1, 'More than one object found in detection band'

        self.get_filter_wavs()

        # Get aperture photometry
        auto_photometry = {}
        flux_aper, flux_err_aper, aper_depths, wave = [], [], [], []
        for band in bands:
            if band == detection_band:
                flux_im = detection_image
                err_im = detection_rms_error
                pixel_radius = (
                    aper_radius / self.im_pixel_scales[self.bands[-1]]
                )

            else:
                flux_im = phot_images[band]
                err_im = rms_err_images[band]
                wave.append(self.filter_wavs[band].to(u.AA).value)
                pixel_radius = aper_radius / self.im_pixel_scales[band]

            # Make C contigous arrays
            flux_im = np.ascontiguousarray(flux_im)
            err_im = np.ascontiguousarray(err_im)

            band_objects, segmap = sep.extract(
                flux_im,
                thresh=det_thresh,
                err=err_im,
                minarea=minarea,
                deblend_nthresh=deblend_nthresh,
                filter_kernel=conv,
                deblend_cont=deblend_cont,
                clean=True,
                clean_param=1.0,
                segmentation_map=True,
            )

            # get seg ID in center
            seg_id = (
                segmap[segmap.shape[0] // 2, segmap.shape[1] // 2]
                if len(np.unique(segmap)) > 2
                else 1
            )
            if len(np.unique(segmap)) == 1:
                seg_id = 0

            if band != detection_band:
                seg_imgs[band] = segmap

            flux, fluxerr, flag = sep.sum_circle(
                flux_im,
                detection_objects["x"],
                detection_objects["y"],
                pixel_radius,
                err=err_im,
                gain=1.0,
            )

            flux = flux[detect_seg_id - 1] * u.uJy

            if band != detection_band:
                flux_aper.append(flux.to(u.Jy).value)
                # Divide by 5 as 5 sigma depths.

                flux_err_aper_i = depths[band].to(u.Jy) / 5
                flux_err_aper.append(flux_err_aper_i.to(u.Jy).value)

                aper_depths.append(
                    -2.5 * np.log10(depths[band].to(u.uJy).value) + 23.9
                )
                auto_photometry[band] = {}

            # FLUX_AUTO

            kronrad, krflag = sep.kron_radius(
                flux_im,
                band_objects["x"],
                band_objects["y"],
                band_objects["a"],
                band_objects["b"],
                band_objects["theta"],
                6.0,
            )
            kron_flux, kron_fluxerr, kron_flag = sep.sum_ellipse(
                flux_im,
                band_objects["x"],
                band_objects["y"],
                band_objects["a"],
                band_objects["b"],
                band_objects["theta"],
                2.5 * kronrad,
                subpix=1,
            )
            kron_flag |= krflag  # combine flags into 'flag'

            # Replicate SExtractor's FLUX_AUTO with minimum radius (this current setup is equivalent to PHOT_AUTOPARAMS 2.5, 4.0)
            r_min = 2  # minimum diameter = 4
            use_circle = (
                kronrad * np.sqrt(band_objects["a"] * band_objects["b"])
                < r_min
            )
            cflux, cfluxerr, cflag = sep.sum_circle(
                flux_im,
                band_objects["x"][use_circle],
                band_objects["y"][use_circle],
                r_min,
                subpix=1,
            )
            kron_flux[use_circle] = cflux
            kron_fluxerr[use_circle] = cfluxerr
            kron_flag[use_circle] = cflag

            # Kron radius

            r, flag = sep.flux_radius(
                flux_im,
                band_objects["x"],
                band_objects["y"],
                6.0 * band_objects["a"],
                0.5,
                normflux=kron_flux,
                subpix=5,
            )

            obj_mask = seg_id - 1  #

            if seg_id != 0:
                # add dimension on last axis of band_objects

                x = band_objects["x"]
                # print('here')
                # print(x, type(x))
                x = x[obj_mask] if type(x) is np.ndarray else x
                # print(x)
                y = band_objects["y"]
                y = y[obj_mask] if type(y) is np.ndarray else y
                a = band_objects["a"]
                a = a[obj_mask] if type(a) is np.ndarray else a
                b = band_objects["b"]
                b = b[obj_mask] if type(b) is np.ndarray else b
                theta = band_objects["theta"]
                theta = theta[obj_mask] if type(theta) is np.ndarray else theta

                r = r[obj_mask] if type(r) is np.ndarray else r
                kron_flux = (
                    kron_flux[obj_mask]
                    if type(kron_flux) is np.ndarray
                    else kron_flux
                )
                kron_fluxerr = (
                    kron_fluxerr[obj_mask]
                    if type(kron_fluxerr) is np.ndarray
                    else kron_fluxerr
                )
                if band != detection_band:
                    auto_photometry[band]["X_IMAGE"] = np.atleast_1d(x)[0]
                    auto_photometry[band]["Y_IMAGE"] = np.atleast_1d(y)[0]
                    auto_photometry[band]["A_IMAGE"] = np.atleast_1d(a)[0]
                    auto_photometry[band]["B_IMAGE"] = np.atleast_1d(b)[0]
                    auto_photometry[band]["THETA_IMAGE"] = np.atleast_1d(
                        theta
                    )[0]
                    auto_photometry[band]["FLUX_RADIUS"] = np.atleast_1d(r)[0]
                    auto_photometry[band]["KRON_RADIUS"] = np.atleast_1d(
                        kronrad
                    )[0]
                    auto_photometry[band]["MAG_AUTO"] = list(
                        -2.5 * np.log10(np.atleast_1d(kron_flux)) + 23.9
                    )[0]  # 23.9 is the zero point for a uJy image
                    auto_photometry[band]["MAGERR_AUTO"] = list(
                        np.atleast_1d(kron_fluxerr)
                        / np.atleast_1d(kron_flux)
                        * 2.5
                        / np.log(10)
                    )[0]

                else:
                    det_data[f"flux_{str(aper_radius)}"] = flux.to(u.uJy).value
                    det_data[f"kron_radius"] = np.atleast_1d(kronrad)[0]
                    det_data["a"] = np.atleast_1d(a)[0]
                    det_data["b"] = np.atleast_1d(b)[0]
                    det_data["theta"] = np.atleast_1d(theta)[0]
                    det_data["flux_kron"] = np.atleast_1d(kron_flux)[0]

            else:
                # No object found
                if band != detection_band:
                    auto_photometry[band]["X_IMAGE"] = None
                    auto_photometry[band]["Y_IMAGE"] = None
                    auto_photometry[band]["A_IMAGE"] = None
                    auto_photometry[band]["B_IMAGE"] = None
                    auto_photometry[band]["THETA_IMAGE"] = None
                    auto_photometry[band]["FLUX_RADIUS"] = None
                    auto_photometry[band]["MAG_AUTO"] = None
                    auto_photometry[band]["MAGERR_AUTO"] = None
                    auto_photometry[band]["KRON_RADIUS"] = None
                else:
                    print("No object found in detection band.")
                    # Show detection segmap.

            if band != detection_band:
                auto_photometry[band]["MAG_BEST"] = None
                auto_photometry[band]["MAGERR_BEST"] = None
                auto_photometry[band]["MAG_ISO"] = None
                auto_photometry[band]["MAGERR_ISO"] = None

        aperture_dict = {
            str(0.32 * u.arcsec): {
                "flux": flux_aper,
                "flux_err": flux_err_aper,
                "depths": aper_depths,
                "wave": wave,
            }
        }
        if "auto" in to_add:
            self.auto_photometry = auto_photometry
        if "aperture" in to_add:
            self.aperture_dict = aperture_dict
        if "seg" in to_add:
            self.det_data = det_data
        if "det_data" in to_add:
            self.seg_imgs = seg_imgs

        if "total_photometry" in to_add:
            self._calc_flux_aper_total()

        if return_results:
            return auto_photometry, aperture_dict, det_data

        if update_h5:
            self.dump_to_h5(force=True)

    def aperture_correction_from_psf(
        self, band, aperture, psf_type="star_stack"
    ):
        psf = self.psfs[psf_type][band]

    def _calc_flux_aper_total(
        self,
        aper_diam=0.32 * u.arcsec,
        override_psf_type=False,
        ratio_band="F277W+F356W+F444W",
        fluxes_psfmatched=True,
        fetch_renormed_psfs=True,
        psf_band="F444W",
    ):
        """Wrapper for scale_fluxes. Scales all fluxes by the same factor to get a  total flux.
        Minimum size of 0.32 arcsec. Requires auto_photometry and aperture_dict to be set.
        """

        psf_type = self.use_psf_type
        if override_psf_type:
            psf_type = override_psf_type

        from .utils import scale_fluxes

        aperture_dict = self.aperture_dict[str(aper_diam)]
        assert len(aperture_dict["flux"]) == len(
            self.bands
        ), "Aperture dict does not match bands"

        flux_totals = []
        ratios = []

        if (
            ratio_band in self.auto_photometry.keys()
            and self.auto_photometry[ratio_band]["MAG_AUTO"] is not None
        ):
            mag_auto = self.auto_photometry[ratio_band]["MAG_AUTO"]
            flux_auto = -2.5 * np.log10(10 ** ((zp - mag_auto) / 2.5)) * u.uJy
            pos = self.bands.index(ratio_band)
            flux_aper = aperture_dict["flux"][pos] * u.Jy

            #       # Get a, b thera
            a = self.auto_photometry[ratio_band]["A_IMAGE"]
            b = self.auto_photometry[ratio_band]["B_IMAGE"]
            theta = self.auto_photometry[ratio_band]["THETA_IMAGE"]
            kron_radius = self.auto_photometry[ratio_band]["KRON_RADIUS"]
        elif "+" in ratio_band:
            print(f"Assuming {ratio_band} is detection band.")
            a = self.det_data["a"]
            b = self.det_data["b"]

            theta = self.det_data["theta"]
            kron_radius = self.det_data["kron_radius"]
            flux_auto = self.det_data["flux_kron"] * u.uJy
            flux_aper = self.det_data[f"flux_{str(aper_diam/2)}"] * u.uJy
            mag_auto = -2.5 * np.log10(flux_auto.to(u.uJy).value) + 23.9

        else:
            raise Exception(f"No auto photometry found for {band}.")

        if "+" in ratio_band:
            bands = ratio_band.split("+")
            zp = self.im_zps[bands[-1]]
            pix_scale = self.im_pixel_scales[bands[-1]]
        else:
            zp = self.im_zps[ratio_band]
            pix_scale = self.im_pixel_scales[ratio_band]

        if fetch_renormed_psfs:
            # This gets the _renorm psfs we need.
            psfs = self.get_star_stack_psf(
                match_band=None, scaled_psf=True, just_return_psfs=True
            )
        else:
            psfs = self.psfs[psf_type]

        bands = self.bands
        if fluxes_psfmatched:
            psfs = [psfs[psf_band]]
            bands = [psf_band]

        ratios = []
        kron_ratios = []
        for band, psf in zip(bands, psfs):
            total_ratio, kron_ratio = scale_fluxes(
                mag_aper=flux_aper,
                mag_auto=flux_auto,
                a=a,
                b=b,
                theta=theta,
                band=band,
                kron_radius=kron_radius,
                psf=psf,
                zero_point=zp,
                aper_diam=aper_diam,
                pix_scale=pix_scale,
                flux_type="flux",
            )
            ratios.append(total_ratio)
            kron_ratios.append(kron_ratio)

        assert (
            len(ratios) == len(self.bands) or len(ratios) == 1
        ), "Ratios do not match bands"

        ratios = np.array(ratios)
        kron_ratios = np.array(kron_ratios)

        if len(ratios) == 1:
            ratios = np.repeat(ratios, len(self.bands))
            kron_ratios = np.repeat(kron_ratios, len(self.bands))

        print(f"Ratio is {ratios[0]:.3f}")
        flux = aperture_dict["flux"] * u.Jy
        flux_err = aperture_dict["flux_err"] * u.Jy
        flux = flux.to(u.uJy).value
        flux_err = flux_err.to(u.uJy).value

        total_photometry = {}

        for pos, band in enumerate(self.bands):
            total_photometry[band] = {
                "flux": flux[pos] * ratios[pos],
                "flux_err": flux_err[pos] * kron_ratios[pos],
                "flux_unit": "uJy",
            }

        total_photometry[ratio_band] = {
            f"a_{ratio_band}": a * kron_radius,
            f"b_{ratio_band}": b * kron_radius,
            f"theta_{ratio_band}": theta,
            f"kron_radius_{ratio_band}": kron_radius,
            f"total_correction_{ratio_band}": ratios[0],
            f"flux_ratio_{ratio_band}": kron_ratios[0],
        }

        self.total_photometry = total_photometry

    def plot_bagpipes_sed(
        self,
        run_name,
        run_dir="pipes/",
        bins_to_show="all",
        plotpipes_dir="pipes_scripts/",
        flux_unit=u.ABmag,
    ):
        """WARNING! NOT USED IN APP"""
        if (
            not hasattr(self, "sed_fitting_table")
            or "bagpipes" not in self.sed_fitting_table.keys()
        ):
            raise Exception("Need to run bagpipes first")

        table = self.sed_fitting_table["bagpipes"][run_name]

        # Plot map next to SED plot

        fig = plt.figure(constrained_layout=True, figsize=(8, 4))
        gs = fig.add_gridspec(2, 3)
        ax_map = fig.add_subplot(gs[:, 0])
        ax_sed = fig.add_subplot(gs[:, 1:])

        ax_divider = make_axes_locatable(ax_map)
        cax = ax_divider.append_axes("top", size="5%", pad="2%")

        # Make cmap for map
        count = []
        other_count = []
        for i, gal_id in enumerate(table["#ID"]):
            try:
                gal_id = float(gal_id)
                count.append(i)
            except:
                other_count.append(i)
                pass

        cmap = plt.cm.get_cmap("cmr.cosmic", len(count))
        color = {table["#ID"][i]: cmap(pos) for pos, i in enumerate(count)}
        entire_cmap = plt.cm.get_cmap("cmr.pepper", len(other_count))
        color.update(
            {
                table["#ID"][i]: entire_cmap(pos)
                for pos, i in enumerate(other_count)
            }
        )
        map = copy.copy(self.pixedfit_map)
        map[map == 0] = np.nan

        mappable = ax_map.imshow(
            map, origin="lower", interpolation="none", cmap="cmr.cosmic"
        )
        cbar = fig.colorbar(mappable, cax=cax, orientation="horizontal")
        cbar.ax.xaxis.set_ticks_position("top")
        cbar.ax.xaxis.set_label_position("top")
        cbar.ax.tick_params(labelsize=8)
        cbar.ax.xaxis.set_major_formatter(ScalarFormatter())

        if bins_to_show == "all":
            bins_to_show = table["#ID"]
        else:
            bins_to_show = bins_to_show

        for pos, rbin in enumerate(bins_to_show):
            # print("rbin", rbin)
            if rbin == "RESOLVED":
                total_flux, total_wav = self.get_resolved_bagpipes_sed(
                    run_name, run_dir=run_dir, plotpipes_dir=plotpipes_dir
                )
                total_flux = total_flux * u.uJy
                total_wav = total_wav * u.um

                ax_sed.plot(
                    total_wav,
                    total_flux,
                    label="RESOLVED",
                    color="tomato",
                )
                continue

            else:
                try:
                    pipes_obj = self.load_pipes_object(
                        run_name,
                        rbin,
                        run_dir=run_dir,
                        cache=None,
                        plotpipes_dir=plotpipes_dir,
                    )
                except:
                    print(f"File not found for {run_name} {rbin} (sed)")
                    continue

                # This plots the observed SED
                # pipes_obj.plot_sed(ax=ax_sed, colour=color[bin],
                #  wav_units=u.um, flux_units=u.ABmag, x_ticks=None, zorder=4, ptsize=40,
                #                y_scale=None, lw=1., skip_no_obs=False, fcolour='blue',
                #                label=None,  marker="o", rerun_fluxes=False)
                # This plots the best fit SED
                # print(color)

                pipes_obj.plot_best_fit(
                    ax_sed,
                    colour=color[str(rbin)],
                    wav_units=u.um,
                    flux_units=flux_unit,
                    lw=1,
                    fill_uncertainty=False,
                    zorder=5,
                    linestyle="-." if pos in other_count else "solid",
                    label=rbin if pos in other_count else "",
                )
                # Plot photometry
                """pipes_obj.plot_sed(
                    ax_sed,
                    colour=color[str(rbin)],
                    wav_units=u.um,
                    flux_units=flux_unit,
                    zorder=6,
                    fcolour=color[str(rbin)],
                    ptsize=15,
                )"""
                try:
                    float(rbin)
                    # find CoM of pixels in map
                    y, x = np.where(map == rbin)
                    y = np.mean(y)
                    x = np.mean(x)
                    ax_map.text(
                        x,
                        y,
                        rbin,
                        fontsize=8,
                        color="black",
                        path_effects=[
                            PathEffects.withStroke(
                                linewidth=1, foreground="white"
                            )
                        ],
                    )
                except ValueError:
                    pass
        # Set x-axis limits
        ax_sed.set_xlim(0.5, 5)
        # Set y-axis limits
        ax_sed.set_ylim(31, 25)

        # Set x-axis label
        ax_sed.set_xlabel(r"Wavelength ($\mu$m)")
        # Set y-axis label
        ax_sed.set_ylabel("AB Mag")
        ax_sed.legend(fontsize=6, loc="upper left")

    def init_galfind_phot(
        self,
        inst="ACS_WFC+NIRCam",
        psf_type="star_stack",
        binmap_type="pixedfit",
    ):
        from galfind import Combined_Instrument, Photometry_rest

        if not hasattr(self, "photometry_table"):
            raise Exception("Need to run measure_flux_in_bins first")
        if hasattr(self, "use_psf_type"):
            psf_type = self.use_psf_type
            print(f"Using PSF type {psf_type}")
        else:
            print(f"Using PSF type from argument {psf_type}")

        if hasattr(self, "use_binmap_type"):
            binmap_type = self.use_binmap_type
            print(f"Using binmap type {binmap_type}")
        else:
            print(f"Using binmap type from argument {binmap_type}")

        if psf_type not in self.photometry_table.keys():
            raise ValueError(
                f"PSF type {psf_type} not found in photometry table"
            )
        if binmap_type not in self.photometry_table[psf_type].keys():
            raise ValueError(
                f"Binmap type {binmap_type} not found in photometry table"
            )

        table = self.photometry_table[psf_type][binmap_type]
        self.galfind_photometry_rest = {}

        start_instrument = Combined_Instrument.from_name(inst)

        for row in tqdm(table):
            flux_Jy, flux_Jy_errs = [], []
            instrument = copy.deepcopy(start_instrument)
            for ratio_band in instrument.band_names:
                if ratio_band in self.bands:
                    flux = row[ratio_band].to(u.Jy).value
                    flux_err = row[f"{ratio_band}_err"].to(u.Jy).value
                    flux_Jy.append(flux)
                    flux_Jy_errs.append(flux_err)
                else:
                    instrument.remove_band(ratio_band)

            flux_Jy = np.array(flux_Jy) * u.Jy
            flux_Jy_errs = np.array(flux_Jy_errs) * u.Jy
            self.galfind_photometry_rest[str(row["ID"])] = Photometry_rest(
                instrument,
                flux_Jy,
                flux_Jy_errs,
                depths=np.ones(len(flux_Jy)),
                z=self.redshift,
            )
        print("Finished building galfind photometry")

    def galfind_phot_property_map(
        self,
        property,
        iters=100,
        load_in=True,
        density=False,
        logmap=False,
        plot=True,
        ax=None,
        facecolor="white",
        cmap="viridis",
        binmap_type="pixedfit",
        **kwargs,
    ):
        """
        Wrapper for galfind.Photometry_rest calculations

        if density: property is converted to a density map (per kpc^2)


        Common required arguments:
        rest_UV_wav_lims = [1250., 3000.] * u.Angstrom : Wavelength limits for rest UV calculations
        conv_author_year = str, e.g. 'M99', Paper ref, what is available depends on the function
        ref_wav =  1_500. * u.AA, wavelength at which to do calculation.
        dust_author_year = str, dust attenuation reference
        kappa_UV_conv_author_year = str, K_UV conversion ref e.g. MD14
        line_names = [], use self.available_em_lines for options
        calc_wav = u.Quantity : calculate propety (only for dust attenuation) at this wavelength


        Property description: name : required kwargs

        Beta: beta_phot :  # rest_UV_wav_lims,
        Dust attenuation in UV: AUV_from_beta_phot : rest_UV_wav_lims, conv_author_year
        Apparent magnitude in UV: mUV_phot : rest_UV_wav_lims, conv_author_year
        Absolute magnitude in UV: MUV_phot : rest_UV_wav_lims, ref_wav
        Luminosity in UV: LUV_phot : rest_UV_wav_lims, ref_wav
        SFR from UV: SFR_UV_phot : rest_UV_wav_lims, ref_wav, dust_author_year, kappa_UV_conv_author_year, frame
        Continuum in rest optical: cont_rest_optical : line_names
        EW in rest optical: EW_rest_optical : line_names
        Dust attenuation: dust_atten : calc_wav
        Line flux in rest optical: line_flux_rest_optical : line_names
        Line luminosity in rest optical: line_lum_rest_optical : line_names
        Ionizing flux: xi_ion :
        """
        if not hasattr(self, "galfind_photometry_rest"):
            print(
                "Warning: galfind photometry not initialized, initializing with default values"
            )
            self.init_galfind_phot(binmap_type=binmap_type)

        from galfind import PDF

        required_kwargs = {
            "beta_phot": ["rest_UV_wav_lims", "iters"],
            "AUV_from_beta_phot": [
                "rest_UV_wav_lims",
                "dust_author_year",
                "iters",
                "ref_wav",
            ],
            "mUV_phot": ["rest_UV_wav_lims", "ref_wav", "iters"],
            "MUV_phot": ["rest_UV_wav_lims", "ref_wav", "iters"],
            "LUV_phot": ["rest_UV_wav_lims", "ref_wav", "iters", "frame"],
            "SFR_UV_phot": [
                "rest_UV_wav_lims",
                "ref_wav",
                "dust_author_year",
                "kappa_UV_conv_author_year",
                "frame",
                "iters",
            ],
            "fesc_from_beta_phot": [
                "rest_UV_wav_lims",
                "conv_author_year",
                "iters",
            ],
            "EW_rest_optical": ["strong_line_names"],
            "line_flux_rest_optical": ["strong_line_names", "iters"],
            "xi_ion": ["iters"],
        }
        if type(binmap_type) is str:
            map = getattr(
                self, binmap_type, getattr(self, f"{binmap_type}_map")
            )

        elif type(binmap_type) is np.ndarray:
            map = binmap_type

        map = copy.deepcopy(map)
        map[map == 0] = np.nan
        # PDFs = []

        kwargs["iters"] = iters

        test_id = np.unique(map)[1]
        func = getattr(
            self.galfind_photometry_rest[str(int(test_id))], f"calc_{property}"
        )

        # arguments = signature(func).parameters
        # func_args = arguments['args']
        # func_kwargs = arguments['kwargs']
        # print(func_args, func_kwargs)
        # print(func_args.default, func_kwargs.default)
        # print(func)

        # match kwargs to arguments
        # for arg in arguments:
        #    if arg not in kwargs.keys():
        #        raise ValueError(f'Missing required argument {arg} for {property}')

        used_kwargs = {
            arg: kwargs[arg]
            for arg in kwargs
            if arg in required_kwargs[property]
        }

        print(kwargs, used_kwargs)

        property_name = func(extract_property_name=True, **used_kwargs)
        num_pdfs = len(property_name)
        PDFs = [[] for i in range(num_pdfs)]

        for pos, gal_id in enumerate(tqdm(np.unique(map))):
            if str(gal_id) in ["0", "nan"]:
                value = np.nan
                continue

            phot = copy.deepcopy(
                self.galfind_photometry_rest[str(int(gal_id))]
            )
            loaded = False
            for prop_name in property_name:
                # print('checking', f'{prop_name}')

                if hasattr(self, f"{prop_name}"):
                    skip = False
                    ppdf = getattr(self, f"{prop_name}")[pos]
                    # print(ppdf.unit)

                    if not skip and load_in:
                        # print('Updating property PDFs from .h5')
                        # pdf = phot.property_PDFs[property_name][pos]
                        # pdf = getattr(self, f'{property_name}_PDFs')[pos]

                        saved_kwargs = getattr(self, f"{prop_name}_kwargs", {})

                        phot.property_PDFs[prop_name] = PDF.from_1D_arr(
                            prop_name, ppdf, kwargs=saved_kwargs
                        )

                        phot._update_properties_from_PDF(prop_name)
                        loaded = True
                        if len(ppdf) != iters:
                            # print(np.shape(pdf), [iters, len(np.unique(map))])
                            print(
                                f"PDFs for {prop_name} do not match shape of map, recalculating, {len(ppdf)} != {iters}"
                            )
                            loaded = False

            func_name = f"calc_{property}"
            if hasattr(phot, func_name):
                func = getattr(phot, func_name)
                _, param_names = phot._calc_property(
                    SED_rest_property_function=func, **used_kwargs
                )

                # if type(param_name) in [tuple, list]:
                #    param_name = param_name
            else:
                raise ValueError(
                    f"Function calc_{property} not found in galfind photometry object"
                )

            for pos_param, param_name in enumerate(param_names):
                value = phot.properties[param_name]
                # print(param_name, value)
                if type(value) in [
                    u.Quantity,
                    u.Magnitude,
                    Masked(u.Quantity),
                ]:
                    unit = value.unit
                else:
                    print("Unknown unit!")
                    print(value)

                    unit = u.dimensionless_unscaled

                if phot.property_PDFs[param_name] is not None:
                    out_kwargs = phot.property_PDFs[param_name].kwargs
                    PDFs[pos_param].append(
                        phot.property_PDFs[param_name].input_arr
                    )

                else:
                    out_kwargs = {}
                    PDFs[pos_param].append([])

                if pos_param == len(param_names) - 1:
                    pname = param_name
                    map[map == gal_id] = value
        if density:
            map = self.map_to_density_map(
                map, redshift=self.redshift, logmap=logmap
            )

            unit = unit / u.kpc**2

        label = f"{pname} ({unit:latex})"

        if not loaded:
            for pos, param_name in enumerate(param_names):
                unit = phot.properties[param_name].unit
                out_kwargs["unit"] = unit
                all_PDFs = np.array(PDFs[pos]) * unit
                out_kwargs = phot.property_PDFs[param_name].kwargs
                self.add_to_h5(
                    all_PDFs,
                    "photometry_properties",
                    param_name,
                    setattr_gal=f"{param_name}",
                    overwrite=True,
                    meta=out_kwargs,
                    setattr_gal_meta=f"{param_name}_kwargs",
                )
                if getattr(self, "photometry_properties", None) is None:
                    self.photometry_properties = {}
                self.photometry_properties[param_name] = all_PDFs
                self.photometry_property_names.append(param_name)

        if np.all(np.isnan(map)):
            print(f"Calculation not possible for {param_name}")
            return None, None

        if plot:
            if ax is not None:
                fig = ax.get_figure()
            else:
                fig, ax = plt.subplots(
                    1,
                    1,
                    figsize=(5, 5),
                    constrained_layout=True,
                    facecolor=facecolor,
                )

            ax_divider = make_axes_locatable(ax)
            cax = ax_divider.append_axes("top", size="5%", pad="2%")

            i = ax.imshow(map, origin="lower", interpolation="none", cmap=cmap)
            cbar = fig.colorbar(i, cax=cax, orientation="horizontal")
            # Label top of axis
            cbar.set_label(label, labelpad=10, fontsize=10)
            cbar.ax.xaxis.set_ticks_position("top")
            cbar.ax.xaxis.set_label_position("top")
            cbar.ax.tick_params(labelsize=8)
            cbar.ax.xaxis.set_major_formatter(ScalarFormatter())
            if property == "EW_rest_optical":
                line = param_name.split("_")[2]

                line_band = out_kwargs[f"{line}_emission_band"]
                cont_band = out_kwargs[f"{line}_cont_band"]
                ax.text(
                    0.05,
                    0.95,
                    f"{line_band} - {cont_band}",
                    transform=ax.transAxes,
                    fontsize=10,
                    horizontalalignment="left",
                    verticalalignment="top",
                    color="black",
                )

            return fig

        return map, out_kwargs

    # calc_beta_phot(
    @property
    def available_em_lines(self):
        from galfind import Emission_lines

        return (
            Emission_lines.strong_optical_lines
        )  # Emission_lines.line_diagnostics.keys()

    def plot_ew_figure(
        self, save=False, facecolor="white", max_col=5, **kwargs
    ):
        to_plot = {}
        for em_line in self.available_em_lines:
            map, out_kwargs = self.galfind_phot_property_map(
                "EW_rest_optical",
                plot=False,
                facecolor=facecolor,
                strong_line_names=[em_line],
                rest_optical_wavs=[3_700.0, 10_000.0] * u.AA,
                **kwargs,
            )
            if type(map) != type(None):
                to_plot[em_line] = (map, out_kwargs)

        num_rows = len(to_plot) // max_col + 1
        fig, axes = plt.subplots(
            num_rows,
            max_col,
            figsize=(2.5 * max_col, 2.5 * num_rows),
            constrained_layout=True,
            facecolor=facecolor,
            sharex=True,
            sharey=True,
        )
        fig.get_layout_engine().set(h_pad=4 / 72, hspace=0.2)

        axes = axes.flatten()
        for pos, line in enumerate(to_plot.keys()):
            cax = make_axes_locatable(axes[pos]).append_axes(
                "top", size="5%", pad="2%"
            )
            map, out_kwargs = to_plot[line]
            i = axes[pos].imshow(
                map, origin="lower", interpolation="none", cmap="viridis"
            )
            cbar = fig.colorbar(i, cax=cax, orientation="horizontal")
            line_band = out_kwargs[f"{line}_emission_band"]
            cont_band = out_kwargs[f"{line}_cont_band"]
            axes[pos].text(
                0.05,
                0.95,
                f"{line_band} - {cont_band}",
                transform=axes[pos].transAxes,
                fontsize=10,
                horizontalalignment="left",
                verticalalignment="top",
                color="black",
            )
            cbar.set_label(line, labelpad=10)
            cbar.ax.xaxis.set_ticks_position("top")
            cbar.ax.xaxis.set_label_position("top")
            cbar.ax.tick_params(labelsize=8)
            cbar.ax.xaxis.set_major_formatter(ScalarFormatter())
            # axes[pos].set_title(line)
        # Remove empty axes
        for i in range(len(to_plot), len(axes)):
            fig.delaxes(axes[i])
        return fig


class MockResolvedGalaxy(ResolvedGalaxy):
    def __init__(
        self,
        galaxy_id,
        mock_survey,
        mock_version,
        instruments=["ACS_WFC", "NIRCam"],
        bands=[],
        excl_bands=[],
        cutout_size=64,
        forced_phot_band=["F277W", "F356W", "F444W"],
        aper_diams=[0.32] * u.arcsec,
        output_flux_unit=u.uJy,
        h5_folder=resolved_galaxy_dir,
        dont_psf_match_bands=[],
        already_psf_matched=False,
        sky_coord=None,
        im_paths=[],
        im_exts=[],
        err_paths=[],
        rms_err_exts=[],
        seg_paths=[],
        rms_err_paths=[],
        phot_img_headers={},
        psf_matched_data=None,
        psf_matched_rms_err=None,
        phot_imgs={},
        rms_err_imgs={},
        seg_imgs={},
        aperture_dict=None,
        redshift=None,
        im_pixel_scales=None,
        im_zps=None,
        phot_pix_unit=None,
        det_data=None,
        synthesizer_galaxy=None,
        noise_images=None,
        meta_properties=None,
        property_images=None,
        auto_photometry=None,
        seds=None,
        sfh=None,
        **kwargs,
    ):
        self.synthesizer_galaxy = synthesizer_galaxy
        self.noise_images = noise_images
        self.property_images = property_images
        self.meta_properties = meta_properties
        self.seds = seds
        self.sfh = sfh

        super().__init__(
            galaxy_id=galaxy_id,
            sky_coord=sky_coord,
            survey=mock_survey,
            bands=bands,
            im_paths=im_paths,
            im_exts=im_exts,
            im_zps=im_zps,
            seg_paths=seg_paths,
            detection_band="+".join(forced_phot_band),
            galfind_version=mock_version,
            rms_err_paths=err_paths,
            rms_err_exts=rms_err_exts,
            im_pixel_scales=im_pixel_scales,
            phot_imgs=phot_imgs,
            phot_pix_unit=phot_pix_unit,
            phot_img_headers=phot_img_headers,
            rms_err_imgs=rms_err_imgs,
            seg_imgs=seg_imgs,
            aperture_dict=aperture_dict,
            redshift=redshift,
            cutout_size=cutout_size,
            dont_psf_match_bands=dont_psf_match_bands,
            auto_photometry=auto_photometry,
            psf_matched_data=psf_matched_data,
            psf_matched_rms_err=psf_matched_rms_err,
            meta_properties=meta_properties,
            already_psf_matched=already_psf_matched,
            det_data=det_data,
            overwrite=False,
            h5_folder=h5_folder,
            **kwargs,
        )

    @classmethod
    def init_from_h5(
        cls,
        h5_name,
        h5_folder=resolved_galaxy_dir,
        return_attr=False,
        save_out=True,
    ):
        # Just get the mock data then call the super init_from h5
        if type(h5_name) != BytesIO:
            if not h5_name.endswith(".h5"):
                h5_name = f"{h5_name}.h5"
            if h5_name.startswith("/"):
                h5path = h5_name
            else:
                h5path = f"{h5_folder}{h5_name}"
        else:
            h5path = h5_name

        with h5.File(h5path, "r") as hfile:
            try:
                mock_galaxy = hfile["mock_galaxy"]
            except KeyError:
                raise Exception(
                    f"{h5_name} does not contain a mock_galaxy class, but has been opened as a mock resolved galaxy"
                )

            params = {}

            params["save_out"] = save_out

            if "synthesizer_galaxy" in mock_galaxy.keys():
                params["synthesizer_galaxy"] = mock_galaxy[
                    "synthesizer_galaxy"
                ][:]

            if "noise_images" in mock_galaxy.keys():
                params["noise_images"] = {}
                for key in mock_galaxy["noise_images"].keys():
                    params["noise_images"][key] = mock_galaxy["noise_images"][
                        key
                    ][:]

            if "property_images" in mock_galaxy.keys():
                params["property_images"] = {}
                for key in mock_galaxy["property_images"].keys():
                    params["property_images"][key] = mock_galaxy[
                        "property_images"
                    ][key][:]

            if "meta_properties" in mock_galaxy.keys():
                params["meta_properties"] = ast.literal_eval(
                    mock_galaxy["meta_properties"][()].decode("utf-8")
                )

            # seds

            if "seds" in mock_galaxy.keys():
                params["seds"] = {}
                for key in mock_galaxy["seds"].keys():
                    if type(mock_galaxy["seds"][key]) is h5.Dataset:
                        params["seds"][key] = mock_galaxy["seds"][key][()]
                    elif type(mock_galaxy["seds"][key]) is h5.Group:
                        params["seds"][key] = {}
                        for key2 in mock_galaxy["seds"][key].keys():
                            params["seds"][key][key2] = mock_galaxy["seds"][
                                key
                            ][key2][()]

            # sfh

            if "sfh" in mock_galaxy.keys():
                params["sfh"] = {}
                for key in mock_galaxy["sfh"].keys():
                    if type(mock_galaxy["sfh"][key]) is h5.Dataset:
                        params["sfh"][key] = mock_galaxy["sfh"][key][()]
                    elif type(mock_galaxy["sfh"][key]) is h5.Group:
                        params["sfh"][key] = {}
                        for key2 in mock_galaxy["sfh"][key].keys():
                            params["sfh"][key][key2] = mock_galaxy["sfh"][key][
                                key2
                            ][()]

        variables = super().init_from_h5(
            h5_name=h5_name,
            h5_folder=h5_folder,
            return_attr=True,
        )

        # Combine the two dictionaries
        variables.update(params)

        # rename survey to mock_survey
        variables["mock_survey"] = variables["survey"]
        del variables["survey"]

        # rename version to mock_version
        variables["mock_version"] = variables["galfind_version"]
        del variables["galfind_version"]

        if return_attr:
            return variables

        return cls(**variables)

    @classmethod
    def init_mock_from_synthesizer(
        cls,
        redshift_code,
        gal_region=None,
        gal_id=None,
        galaxy_index=None,
        psfs_dir="/nvme/scratch/work/tharvey/PSFs/JOF/",
        cutout_size="auto",
        mock_survey="JOF",
        mock_version="v11",
        mock_instruments=["ACS_WFC", "NIRCam"],
        mock_forced_phot_band=["F277W", "F356W", "F444W"],
        mock_aper_diams=[0.32] * u.arcsec,
        h5_folder=resolved_galaxy_dir,
        resolution=0.03,  # in arcsec
        psf_type="",
        file_path="/nvme/scratch/work/tharvey/EXPANSE/data/JOF_mock.h5",
        grid_name="bpass-2.2.1-bin_chabrier03-0.1,300.0_cloudy-c17.03",
        grid_dir="./grids/",
        debug=False,
        mock_rms_fit_path="",
        depth_file={
            "NIRCam": "/raid/scratch/work/austind/GALFIND_WORK/Depths/NIRCam/v11/JOF/old/0.32as/n_nearest/JOF_depths.ecsv",
            "ACS_WFC": "/raid/scratch/work/austind/GALFIND_WORK/Depths/ACS_WFC/v11/JOF/0.32as/n_nearest/JOF_depths.ecsv",
        },
        override_model_assumptions={},
    ):
        update_cli_interface = False  # is_cli()
        if update_cli_interface:
            cli = CLIInterface()
            cli.start()

            full_gal_id = (
                f"{redshift_code}_{gal_region}_{gal_id}"
                if gal_id is not None
                else f"{redshift_code}_{galaxy_index}"
            )

            lines = [
                [
                    f"Galaxy: {full_gal_id}",
                    f"Survey: {mock_survey}",
                    f"Version: {mock_version}",
                ],
                [f"Synthesizer: Grid: {grid_name}"],
                [""],
                [""],
            ]

            lines[2][0] = (
                "Current Task: Initializing Mock Galaxy from Synthesizer"
            )
            cli.update(lines)

        assert (
            gal_region is not None and gal_id is not None
        ) or galaxy_index is not None

        if False:
            with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
                from galfind import EAZY, Catalogue
                from galfind.Catalogue_Creator import GALFIND_Catalogue_Creator

            SED_code_arr = [EAZY()]
            templates_arr = ["fsps_larson"]
            lowz_zmax_arr = ([[4.0, 6.0, None]],)
            SED_fit_params_arr = make_EAZY_SED_fit_params_arr(
                SED_code_arr, templates_arr, lowz_zmax_arr
            )
            # Make cat creator
            cat_creator = GALFIND_Catalogue_Creator(
                "loc_depth", mock_aper_diams[0], 10
            )
            # Load catalogue and populate galaxies
            cat = Catalogue.from_pipeline(
                survey=mock_survey,
                version=mock_version,
                instruments=mock_instruments,
                aper_diams=mock_aper_diams,
                cat_creator=cat_creator,
                SED_fit_params_arr=SED_fit_params_arr,
                forced_phot_band=mock_forced_phot_band,
                excl_bands=[],
                loc_depth_min_flux_pc_errs=[10],
            )

            im_paths = cat.data.im_paths
            im_exts = cat.data.im_exts
            err_paths = cat.data.rms_err_paths
            rms_err_exts = cat.data.rms_err_exts
            seg_paths = cat.data.seg_paths
            im_zps = cat.data.im_zps
            im_pixel_scales = cat.data.im_pixel_scales
            bands = (
                galaxy.phot.instrument.band_names
            )  # should be bands just for galaxy!
            translate = {"NIRCam": "JWST/NIRCam", "ACS_WFC": "HST/ACS_WFC"}
            instrument = [
                translate(cat.data.instrument.instrument_from_band(band).name)
                for band in bands
            ]
        else:
            bands = [
                "F435W",
                "F606W",
                "F775W",
                "F814W",
                "F850LP",
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
                "F410M",
                "F444W",
            ]
            instruments = 5 * ["ACS_WFC"] + 14 * ["NIRCam"]
            observatories = 5 * ["HST"] + 14 * ["JWST"]

        try:
            from scipy import signal
            from synthesizer.emission_models import (
                IncidentEmission,
                PacmanEmission,
            )
            from synthesizer.emission_models.attenuation import (
                PowerLaw,
                Calzetti2000,
            )
            from synthesizer.emission_models.attenuation.igm import (
                Inoue14,
                Madau96,
            )
            from synthesizer.emission_models.dust.emission import Greybody
            from synthesizer.filters import FilterCollection as Filters
            from synthesizer.grid import Grid
            from synthesizer.imaging.image import Image
            from synthesizer.kernel_functions import Kernel
            from synthesizer.particle.galaxy import Galaxy
            from unyt import (
                Angstrom,
                Gyr,
                Hz,
                K,
                Mpc,
                Msun,
                Myr,
                arcsecond,
                erg,
                kpc,
                nJy,
                s,
                uJy,
                um,
                unyt_array,
                yr,
            )

            from .synthesizer_functions import (
                apply_pixel_coordinate_mask,
                convert_coordinates,
                get_spectra_in_mask,
            )

        except ImportError:
            raise Exception(
                "Synthesizer not installed. Please clone from Github and install using pip install ."
            )

        resolution = resolution * arcsecond

        grid = Grid(
            grid_name, grid_dir=grid_dir
        )  # TODO: Apply reasonable wavelength limits to grid to reduce computation time and memory usage

        # tags = ['010_z005p000', '008_z007p000', '007_z008p000', '005_z010p000']
        # regions = [['00', '00', '01', '10', '18'], ['00', '02', '09'], ['21', '17'], ['15']]
        # ids = [[12, 96, 1424, 1006, 233], [6, 46, 298], [111, 16], [99]]
        filter_codes = [
            f"{obs}/{inst}.{band}"
            for obs, inst, band in zip(observatories, instruments, bands)
        ]
        filters = Filters(filter_codes, new_lam=grid.lam)

        filter_code_from_band = {
            band: code for band, code in zip(bands, filter_codes)
        }
        # Need to get this from the .h5 file instead
        with h5.File(file_path, "r") as hf:  # opening the hdf5 file
            # coordinates of the stellar particles
            regions = {i: [] for i in hf.keys()}
            ids = {i: [] for i in hf.keys()}
            for region in regions.keys():
                regions[region] = [i for i in hf[region].keys()]
                for reg in regions[region]:
                    ids[region].extend(
                        [i for i in hf[f"{region}/{reg}"].keys()]
                    )

            """regions = {
                "010_z005p000": ["00", "00", "01", "10", "18"],
                "008_z007p000": ["00", "02", "09"],
                "007_z008p000": ["21", "17"],
                "005_z010p000": ["15"],
            }
            ids = {
                "010_z005p000": [12, 96, 1424, 1006, 233],
                "008_z007p000": [6, 46, 298],
                "007_z008p000": [111, 16],
                "005_z010p000": [99],
            }"""
            if gal_id is None and region is None:
                gal_id = ids[redshift_code][galaxy_index] - 1
                region = regions[redshift_code][galaxy_index]
            else:
                region = gal_region

            tag = redshift_code

            zed = float(tag[5:].replace("p", "."))

            if hf.get(f"{tag}/{region}/{gal_id}") is None:
                if update_cli_interface:
                    cli.update(
                        current_task=f"Error! Galaxy {gal_id} not found in region {region} of tag {tag} in {file_path}"
                    )
                print(f"{tag}/{region}/{gal_id}")
                raise Exception(
                    f"Galaxy {gal_id} not found in region {region} of tag {tag}"
                )

            coordinates = (
                np.array(
                    hf[f"{tag}/{region}/{gal_id}"].get("coordinates"),
                    dtype=np.float64,
                )
                * Mpc
            )
            # initial masses of the stellar particles
            initial_masses = (
                np.array(
                    hf[f"{tag}/{region}/{gal_id}"].get("initial_masses"),
                    dtype=np.float64,
                )
                * Msun
            )
            # current masses of the stellar particles, mass change due to mass loss as stars age
            current_masses = (
                np.array(
                    hf[f"{tag}/{region}/{gal_id}"].get("current_masses"),
                    dtype=np.float64,
                )
                * Msun
            )
            # ages of stars in log10, which is in units of years
            log10ages = np.array(
                hf[f"{tag}/{region}/{gal_id}"].get("log10ages"),
                dtype=np.float64,
            )
            # metallicities of the stars in log10
            log10metallicities = np.array(
                hf[f"{tag}/{region}/{gal_id}"].get("log10metallicities"),
                dtype=np.float64,
            )
            # optical depth in the V-band, it is the same prescription as shown in Vijayan+21, which
            # includes the V-band dust optical depth due to the diffuse ISM dust and birth cloud dust
            tau_v = np.array(
                hf[f"{tag}/{region}/{gal_id}"].get("tau_v"), dtype=np.float64
            )
            # the smoothing lengths of star particles
            smoothing_lengths = (
                np.array(
                    hf[f"{tag}/{region}/{gal_id}"].get("smoothing_lengths"),
                    dtype=np.float64,
                )
                * Mpc
            )

            gal = Galaxy(redshift=zed)
            gal.load_stars(
                initial_masses=initial_masses,
                ages=10**log10ages * yr,
                metallicities=10**log10metallicities,
                coordinates=coordinates,
                current_masses=current_masses,
                smoothing_lengths=smoothing_lengths,
                centre=np.zeros(3) * Mpc,
            )

        gal.tau_v = tau_v

        message = (
            f"Created Galaxy object from .h5 with {len(gal.tau_v)} particles."
        )
        if update_cli_interface:
            cli.update(current_task=message)
        else:
            print(message)

        model_assumptions = {
            "gridname": grid_name,
            "grid_dir": grid_dir,
            "file_path": file_path,
            "dust_type": "pacman",
            # "dust_curve": "power_law",
            # "dust_slope": -0.7,
            "dust_curve": "calzetti2000",
            "dust_slope": 0.0,
            "dust_bump_amplitude": 0.0,
            "dust_bump_wavelength": 0.2175,
            "dust_fwhm_bump": 0.035,
            "dust_emission": "greybody",
            "dust_temp": 30,
            "dust_beta": 1.2,
            "igm": "inoue14",
            "fesc_ly_alpha": 0.0,
            "fesc": 0.0,
            "cosmo": str(cosmo),
        }

        model_assumptions.update(
            override_model_assumptions
        )  # override with any user input

        if model_assumptions["dust_curve"] == "power_law":
            dust_curve = PowerLaw(model_assumptions["dust_slope"])
        elif model_assumptions["dust_curve"] == "calzetti2000":
            dust_curve = Calzetti2000(
                slope=model_assumptions["dust_slope"],
                ampl=model_assumptions["dust_bump_amplitude"],
                cent_lam=model_assumptions["dust_bump_wavelength"] * um,
                gamma=model_assumptions["dust_fwhm_bump"] * um,
            )

        if model_assumptions["dust_emission"] == "greybody":
            dust_emission = Greybody(
                temperature=model_assumptions["dust_temp"] * K,
                emissivity=model_assumptions["dust_beta"],
            )
        elif model_assumptions["dust_emission"] == "blackbody":
            dust_emission = Blackbody(
                temperature=model_assumptions["dust_temp"] * K,
            )

        emission_model = PacmanEmission(
            grid=grid,
            tau_v=tau_v,
            dust_curve=dust_curve,
            dust_emission=dust_emission,
            fesc_ly_alpha=model_assumptions["fesc_ly_alpha"],
            fesc=model_assumptions["fesc"],
            per_particle=True,
        )

        if model_assumptions["igm"] == "inoue14":
            igm = Inoue14
        elif model_assumptions["igm"] == "madau96":
            igm = Madau96
        else:
            raise ValueError(
                f"IGM model {model_assumptions['igm']} not recognised."
            )

        gal.stars.get_particle_spectra(emission_model)
        models = gal.stars.particle_spectra.keys()
        print(models)
        for model in models:
            # Generate spectra for each particle for each model component
            gal.stars.particle_spectra[model].get_fnu(
                cosmo, gal.redshift, igm=igm
            )
            # Need this for making luminosity images only
            gal.stars.particle_spectra[model].get_photo_lnu(filters)

        # gal.get_observed_spectra(cosmo)
        # Combines the spectra of all the particles
        gal.stars.integrate_particle_spectra()

        gal.stars.get_particle_photo_fnu(filters)

        gal.stars.get_photo_fnu(filters)
        # gal.stars.get_photo_lnu(filters) # unneeded apparently

        message = f"Generated spectra and photometry for Galaxy. Model assumptions: {model_assumptions}"
        if update_cli_interface:
            cli.update(current_task=message)
        else:
            print(message)

        if cutout_size == "auto":
            flux_enclosed = 0.95
            size_bands = [
                "JWST/NIRCam.F277W",
                "JWST/NIRCam.F356W",
                "JWST/NIRCam.F444W",
            ]
            half_light_radius = (
                np.nanmax(
                    [
                        gal.stars.get_flux_radius(
                            "total", band, flux_enclosed
                        ).to(kpc)
                        for band in size_bands
                    ]
                )
                * kpc
            )
            print(
                [
                    gal.stars.get_flux_radius("total", band, flux_enclosed).to(
                        kpc
                    )
                    for band in size_bands
                ]
            )

            if np.isnan(half_light_radius):
                print("95% flux radius is NaN, using 5 kpc")
                half_light_radius = 5 * kpc

            # convert to arcseconds
            print(f"95% flux radius: {half_light_radius}")
            half_light_radius = half_light_radius.to_astropy()
            d_A = cosmo.angular_diameter_distance(gal.redshift)
            half_light_radius_arcsec = (half_light_radius / d_A).to(
                u.arcsec, u.dimensionless_angles()
            )
            print(f"Half light radius arcsec: {half_light_radius_arcsec}")

            # convert to pixels
            pixel_scale = 0.03 * u.arcsecond
            half_light_radius_pix = half_light_radius_arcsec / pixel_scale

            message = f"Galaxy size: {half_light_radius:.2f} == {half_light_radius_arcsec:.2f} == {half_light_radius_pix:.2f} pixels"

            if not update_cli_interface:
                print(message)
            else:
                cli.update(current_task=message)
            # Cutout size 2 * 95% flux radius + 10% padding
            cutout_size = int(
                np.ceil(
                    2 * half_light_radius_pix + 0.3 * 2 * half_light_radius_pix
                )
            )

            # Min cutout size 65 pixels
            cutout_size = np.max([65, cutout_size])

        if update_cli_interface:
            lines[1].append(f"Cutout size: {cutout_size} pix")
            cli.update(lines)

        sph_kernel = Kernel()
        kernel_data = sph_kernel.get_kernel()

        img_type = "smoothed"
        spectra_type = "total"

        # Calculate size and resolution of the cutouts
        d_A = (
            float(
                cosmo.angular_diameter_distance(gal.redshift).to(u.kpc).value
            )
            * kpc
        )

        resolution_physical = resolution.to("rad").value * d_A

        fov = (cutout_size - 0.000001) * resolution_physical

        message = f"Creating smoothed image cutouts with resolution {resolution_physical:.3f} and FOV {fov:.3f} ({cutout_size} pixels)"
        if update_cli_interface:
            cli.update(current_task=message)
        else:
            print(message)

        # Get the image
        imgs = gal.get_images_flux(
            resolution_physical,
            fov=fov,
            img_type=img_type,
            kernel=kernel_data,
            kernel_threshold=1,
            emission_model=emission_model,
        )

        # update resolution_physical from the image

        fov = imgs[list(imgs.keys())[0]].fov
        resolution_physical = imgs[list(imgs.keys())[0]].resolution
        npix = imgs[list(imgs.keys())[0]].npix
        print(
            f"Synthesizer Images generated with resolution {resolution_physical} and fov {fov} and npix {npix}"
        )
        cutout_size = npix
        fov = npix * resolution_physical

        psfs = {}
        files = glob.glob(f"{psfs_dir}/*_psf.fits")
        bands = [i.split(".")[1] for i in filters.filter_codes]
        for band, code in zip(bands, filter_codes):
            psf_file = [f for f in files if band in f][0]
            psf = fits.open(psf_file)[0].data
            # Normalise the PSF
            psf /= np.sum(psf)

            psfs[code] = psf

        if update_cli_interface:
            message = f"Applying PSF models from {psfs_dir}"
            cli.update(current_task=message)

        # Apply the PSFs
        psf_imgs = imgs.apply_psfs(psfs)

        snrs = {f: 5 for f in psf_imgs.keys()}

        for pos, key in enumerate(depth_file.keys()):
            table = Table.read(depth_file[key], format="ascii.ecsv")

            if pos == 0:
                main_table = table
            else:
                main_table = vstack([main_table, table])

        table = main_table

        # Select 'all' region column
        table = table[table["region"] == "all"]
        # band_depths:
        bands = [i.split(".")[1] for i in filters.filter_codes]
        band_depths = {}
        for band in bands:
            row = table[table["band"] == band]
            data = row["median_depth"]
            band_depths[band] = data * u.ABmag

        if update_cli_interface:
            message = f"Applying noise from local depths in {depth_file}"
            cli.update(current_task=message)

        wavs = {
            f: np.squeeze(filters.pivot_lams[pos].value)
            for pos, f in enumerate(bands)
        }
        # Convert to uJy from AB mag
        depths = {
            img_key: band_depths[f]
            .to(u.uJy, equivalencies=u.spectral_density(wavs))
            .value
            * uJy
            for f, img_key in zip(bands, imgs.keys())
        }

        depths = {k: v[0] for k, v in depths.items()}

        radius_kpc = mock_aper_diams[0] * d_A

        radius_kpc = radius_kpc.value / 2 * kpc

        noise_app_imgs = psf_imgs.apply_noise_from_snrs(
            snrs=snrs, depths=depths, aperture_radius=radius_kpc
        )

        # Convert to uJy

        for img in [imgs, psf_imgs, noise_app_imgs]:
            for key in img.keys():
                unit = img[key].units
                if unit != uJy:
                    img[key].arr = (img[key].arr * unit).to(uJy).value
                    img[key].units = uJy
                    # Syntheszier can change the size of the image by a few pixels
                    cutout_size = img[key].arr.shape[0]

        possible_galaxies = glob.glob(
            f"{resolved_galaxy_dir}/{mock_survey}_*.h5"
        )

        print(possible_galaxies)
        # Remove those with 'mock' in the name
        possible_galaxies = [g for g in possible_galaxies if "mock" not in g]

        # only allow wildcard to be a number

        # This bit uses real ResolvedGalaxies to approximate an RMS map
        ids = [int(g.split("_")[-1].split(".")[0]) for g in possible_galaxies]

        rms_err_images = {}
        final_images = {}

        loaded = False

        if mock_rms_fit_path != "":
            if os.path.exists(mock_rms_fit_path):
                if update_cli_interface:
                    message = (
                        f"Loading RMS error fits from {mock_rms_fit_path}"
                    )
                    cli.update(current_task=message)

                for band in bands:
                    data = np.genfromtxt(
                        f"{mock_rms_fit_path}/{band}_rms_fit.csv",
                        delimiter=",",
                    )
                    f = interp1d(
                        data[:, 0],
                        data[:, 1],
                        bounds_error=False,
                        fill_value="extrapolate",
                    )
                    # Generate RMS err images from unnoised psf images
                    rms_err_images[band] = f(
                        psf_imgs[filter_code_from_band[band]].arr
                    )

                    # Use the images with PSF and noise applied as the final images
                    final_images[band] = noise_app_imgs[
                        filter_code_from_band[band]
                    ].arr
                loaded = True

        if len(ids) > 0 and not loaded:
            # ids = ids[:2]
            if update_cli_interface:
                message = f"Approximating RMS error maps from {len(ids)} real galaxies."
                cli.update(current_task=message)
            else:
                print(
                    "Found real ResolvedGalaxies to approximate errors from."
                )
                print(f"Using {ids}")

            galaxies = ResolvedGalaxy.init(ids, "JOF_psfmatched", "v11")

            data_type = "PSF"
            for prog, band in tqdm(enumerate(bands)):
                if update_cli_interface:
                    message = f"Generating RMS error fits for {band}"
                    cli.update(
                        current_task=message, progress=prog / len(bands)
                    )

                # Get data to generate RMS error from
                image_band = noise_app_imgs[filter_code_from_band[band]]
                unit = image_band.units
                image_data = image_band.arr * unit
                # have to do this for conversion
                image_data = image_data.to_astropy()
                unit = image_data.unit

                # image_data = image_data.
                total_err, total_data = [], []
                for pos, galaxy in tqdm(enumerate(galaxies)):
                    actual_unit = galaxy.phot_pix_unit
                    if data_type == "PSF":
                        if (
                            band
                            in galaxy.psf_matched_data["star_stack"].keys()
                        ):
                            im = galaxy.psf_matched_data["star_stack"][band]
                            err = galaxy.psf_matched_rms_err["star_stack"][
                                band
                            ]

                        else:
                            continue

                    elif data_type == "ORIGINAL":
                        if band in galaxy.unmatched_data.keys():
                            im = galaxy.unmatched_data[band]
                            err = galaxy.unmatched_rms_err[band]
                        else:
                            continue
                    else:
                        breakmeee

                    actual_unit = galaxy.phot_pix_unit[band]
                    err *= actual_unit
                    im *= actual_unit

                    # Match units
                    err = err.to(unit).value
                    im = im.to(unit).value

                    total_err.extend(list(err.flatten()))
                    total_data.extend(list(im.flatten()))

                total_data = np.array(total_data)
                total_err = np.array(total_err)

                # Remove duplicates and reorder
                unique_x, unique_indices = np.unique(
                    total_data, return_index=True
                )
                x_unique = total_data[unique_indices]
                y_unique = total_err[unique_indices]

                # Use LOWESS to smooth the data
                import statsmodels.api as sm

                lowess = sm.nonparametric.lowess(y_unique, x_unique, frac=0.1)
                # unpack the lowess smoothed points to their values
                lowess_x = list(zip(*lowess))[0]
                lowess_y = list(zip(*lowess))[1]
                if debug:
                    plt.plot(lowess_x, lowess_y, label="lowess", color="red")

                    plt.scatter(total_data, total_err, s=1)
                    plt.title(f"{band} RMS error")
                    plt.show()

                # plt.plot(b.bin_edges[1:], b.statistic, label='binned', color='red')

                f = interp1d(
                    lowess_x,
                    lowess_y,
                    bounds_error=False,
                    fill_value="extrapolate",
                )
                if debug:
                    print(image_data.max(), np.max(lowess_x))
                    print(image_data.min(), image_data.max())
                    print(f(image_data.max()), f(image_data.min()))

                rms_err_images[band] = f(image_data)
                final_images[band] = image_data

                if mock_rms_fit_path != "":
                    if not os.path.exists(mock_rms_fit_path):
                        os.makedirs(mock_rms_fit_path)

                    np.savetxt(
                        f"{mock_rms_fit_path}/{band}_rms_fit.csv",
                        np.array([lowess_x, lowess_y]).T,
                        delimiter=",",
                    )
        elif loaded:
            if update_cli_interface:
                message = f"Loaded RMS error fits from {mock_rms_fit_path}"
                cli.update(current_task=message)

        elif len(ids) == 0 and not loaded:
            # Add rms of noise map as rms_err_iamges
            raise NotImplementedError("Need to add noise map rms errors")

        if update_cli_interface:
            cli.update(
                current_task="Finished approximating RMS error maps.",
                progress=0,
            )

        # Generate noise to add from the RMS error
        noise_images = {}
        for band in bands:
            # Generate noise from the RMS error
            # noise_images[band] = np.random.normal(loc = np.zeros_like(rms_err_images[band]), scale = rms_err_images[band])

            # Compare to the Synthesizer generated noise maps
            syn_noise = noise_app_imgs[filter_code_from_band[band]].noise_arr
            # Check it is 2D
            assert (
                np.ndim(syn_noise) == 2
            ), f"Noise map is not 2D {np.shape(syn_noise)}"
            syn_noise_data = (
                syn_noise * noise_app_imgs[filter_code_from_band[band]].units
            )
            noise_images[band] = syn_noise

            # Need to convert from

            if debug:
                fig, ax = plt.subplots(
                    nrows=1,
                    ncols=3,
                    figsize=(10, 5),
                    constrained_layout=True,
                    facecolor="white",
                )
                # one = ax[0].imshow(noise_images[band], origin='lower', cmap='viridis')
                one = ax[0].imshow(
                    noise_app_imgs[filter_code_from_band[band]].arr
                    / rms_err_images[band],
                    origin="lower",
                    cmap="viridis",
                )
                two = ax[1].imshow(
                    syn_noise_data, origin="lower", cmap="viridis"
                )
                three = ax[2].imshow(
                    rms_err_images[band], origin="lower", cmap="viridis"
                )

                ax[0].set_title("SNR")
                fig.colorbar(one, ax=ax[0])
                fig.colorbar(two, ax=ax[1])
                fig.colorbar(three, ax=ax[2])
                ax[1].set_title("Synthesizer noise")
                ax[2].set_title("RMS error")
                fig.suptitle(f"{band} noise comparison")
                plt.show()

        # Would also be good to get the property maps

        # Things to save - background cutouts, background error maps, background position, noise_data
        # Properties - mass, SFR, tau_v, age, metallicity, ID, redshift etc.
        meta_properties = {
            "redshift": gal.redshift,
            "id": gal_id,
            "tag": tag,
            "region": region,
            "grid": grid_name,
            "filters": filter_codes,
            "forced_phot_band": mock_forced_phot_band,
            "stellar_mass": float(gal.stellar_mass.value),
            "tau_v": float(np.median(gal.tau_v)),
            "age": float(gal.stellar_mass_weighted_age.value),
            "model_assumptions": model_assumptions,
        }

        im_zps = {band: 23.9 for band in bands}
        phot_pix_unit = {band: u.uJy for band in bands}
        im_pixel_scales = {band: resolution.to_astropy() for band in bands}

        # galaxy_id = f"{redshift_code}_{galaxy_index}_mock"  # survey is added later to the filename
        galaxy_id = f"{redshift_code}_{region}_{gal_id}_mock"

        # stellar mass, stellar age, stellar metallicity, sfr, ssfr (10 and 100 Myr) - all have gal.get_map_ ... method

        # save total SED.

        # Other property maps: - make an image instance, and pass in a 'signal'
        # Could signal be e.g. mass - but only for things in a certain region?
        # Get the image instance
        # img = Image(resolution, fov=width)

        # Get an image of the gas dust-to-metal ratio
        """
        img.get_img_smoothed(
            signal=gal.stars.s_oxygen,
            coordinates=gal.stars.coordinates,
            smoothing_lengths=gal.stars.smoothing_lengths,
            kernel=kernel_data,
            kernel_threshold=1,
        )
        """
        # Add astropy unit to self.rms_err_images
        rms_err_images = {
            k: v * phot_pix_unit[k] for k, v in rms_err_images.items()
        }

        # save SED in aperture and in mask. Maybe save just intrinsic

        seds = {}

        if update_cli_interface:
            cli.update(current_task="Saving Synthesizer SEDs.")

        seds["wav"] = gal.stars.spectra["total"].obslam.to(Angstrom).value

        seds["total_fnu"] = {}
        seds["total_fnu"]["total"] = (
            gal.stars.spectra["total"].fnu.to(uJy).value
        )
        seds["0.32as_fnu"] = {}
        seds["0.32as_fnu"]["total"] = (
            get_spectra_in_mask(
                gal, aperture_mask_radii=0.16 * u.arcsec, spectra_type="total"
            )
            .to(uJy)
            .value
        )

        """
        seds["det_segmap_fnu"] = {}
        seds["det_segmap_fnu"]["total"] = (
            get_spectra_in_mask(
                gal, pixel_mask=det_data["seg"], spectra_type="total", pixel_mask_value='center',
            )
            .to(uJy)
            .value
        )
        """

        # other maps - tau_v

        # Get other property maps
        property_images = {}
        for prop in [
            "stellar_mass",
            "sfr",
            "ssfr",
            "stellar_age",
            "stellar_metallicity",
        ]:
            func = getattr(gal, f"get_map_{prop}")
            arguments = {
                "resolution": resolution_physical,
                "fov": fov,
                "img_type": img_type,
                "kernel": kernel_data,
                "kernel_threshold": 1,
            }

            if update_cli_interface:
                cli.update(current_task=f"Generating {prop} map")

            if prop in ["ssfr", "sfr"]:
                if prop == "ssfr":
                    unit = yr**-1
                if prop == "sfr":
                    unit = Msun / yr

                for age_bin in [10, 100] * Myr:
                    arguments["age_bin"] = age_bin

                    with warnings.catch_warnings():
                        warnings.filterwarnings("error")

                        data = np.zeros((1, 1))
                        size_diff = 0
                        # Hard to predict image size. This is an iterative approach.
                        while np.shape(data) != (cutout_size, cutout_size):
                            npix = arguments["fov"] / arguments["resolution"]
                            # Change FOV by size_diff pixels
                            arguments["fov"] = (npix + size_diff) * arguments[
                                "resolution"
                            ]

                            try:
                                prop_im = func(**arguments)
                            except Warning:
                                # Catch case where no star formation is happening
                                prop_im = Image(
                                    resolution=resolution_physical,
                                    fov=fov,
                                    img=unyt_array(
                                        np.zeros((cutout_size, cutout_size)),
                                        units=unit,
                                    ),
                                )

                            data_im = prop_im.arr * prop_im.units
                            data = data_im.to_astropy()
                            size_diff = cutout_size - np.shape(data)[0]

                    # Check the size matches the other cutouts
                    assert np.shape(data) == (
                        cutout_size,
                        cutout_size,
                    ), f"Shape of {prop} map is {np.shape(data)}"
                    property_images[f"{prop}_{age_bin}"] = data
            else:
                prop_im = func(**arguments)
                data_im = prop_im.arr * prop_im.units
                property_images[prop] = data_im.to_astropy()

        if update_cli_interface:
            message = "Finished generating mock galaxy."
            cli.update(current_task=message)
            cli.stop()

        return cls(
            galaxy_id,
            mock_survey,
            mock_version,
            instruments=mock_instruments,
            bands=bands,
            excl_bands=[],
            cutout_size=cutout_size,
            forced_phot_band=mock_forced_phot_band,
            aper_diams=mock_aper_diams,
            output_flux_unit=u.uJy,
            h5_folder=h5_folder,
            dont_psf_match_bands=[],
            already_psf_matched=False,
            synthesizer_galaxy=gal,
            # aperture_dict=aperture_dict,
            # auto_photometry=auto_photometry,
            redshift=zed,
            rms_err_imgs=rms_err_images,
            seg_imgs=None,
            psf_matched_data=None,
            psf_matched_rms_err=None,
            im_pixel_scales=im_pixel_scales,
            im_zps=im_zps,
            phot_pix_unit=phot_pix_unit,
            # det_data=det_data,
            phot_imgs=final_images,
            noise_images=noise_images,
            meta_properties=meta_properties,
            property_images=property_images,
            seds=seds,
        )

    @classmethod
    def init(
        cls,
        mock_survey=None,
        redshift_code=None,
        gal_region=None,
        gal_id=None,
        galaxy_name=None,
        galaxy_index=None,
        overwrite=False,
        h5_folder=resolved_galaxy_dir,
        save_out=True,
        **kwargs,
    ):
        # load from h5 if it exists
        # Initialize CLI if relevant

        assert (
            (mock_survey is not None)
            and (redshift_code is not None)
            and (gal_id is not None)
            or (galaxy_name is not None)
        ), "Need to provide either galaxy_id or mock_survey, redshift_code, and galaxy_index"
        if galaxy_name is not None:
            galaxy_id = f"{mock_survey}_{redshift_code}_{galaxy_name}_mock"
        elif gal_id is not None:
            galaxy_id = (
                f"{mock_survey}_{redshift_code}_{gal_region}_{gal_id}_mock"
            )

        if not galaxy_id.startswith(mock_survey) and mock_survey is not None:
            galaxy_id = f"{mock_survey}_{galaxy_id}"

        file_path = f"{h5_folder}/{galaxy_id}"
        if not file_path.endswith(".h5"):
            file_path += ".h5"

        if os.path.exists(file_path) and not overwrite:
            print("Loading from .h5")
            return cls.init_from_h5(
                galaxy_id, h5_folder=h5_folder, save_out=save_out
            )
        else:
            print("Generating from synthesizer.")
            return cls.init_mock_from_synthesizer(
                redshift_code,
                gal_region=gal_region,
                gal_id=gal_id,
                h5_folder=h5_folder,
                galaxy_index=galaxy_index,
                mock_survey=mock_survey,
                **kwargs,
            )

    @classmethod
    def init_mock_from_sphinx(cls):
        # TODO: Implement this method
        raise NotImplementedError("Need to implement this method.")

    @classmethod
    def init_all_field_from_h5(
        cls,
        field,
        h5_folder=resolved_galaxy_dir,
        save_out=True,
        n_jobs=1,
    ):
        h5_names = glob.glob(f"{h5_folder}{field}*.h5")

        h5_names = [h5_name.split("/")[-1] for h5_name in h5_names]

        # sort them
        h5_names = sorted(h5_names)

        # Remove any with 'mock' in the name
        h5_names = [
            h5_name
            for h5_name in h5_names
            if "mock" in h5_name and "temp" not in h5_name
        ]

        print("Found", len(h5_names), "galaxies in field", field)
        # print(h5_names)
        return cls.init_multiple_from_h5(
            h5_names, h5_folder=h5_folder, save_out=save_out, n_jobs=n_jobs
        )

    """@classmethod
    def init_multiple_from_h5(
        cls,
        h5_names,
        h5_folder=resolved_galaxy_dir,
        save_out=True,
        n_jobs=1,
    ):"""
    # Load multiple galaxies from h5 files

    def save_new_synthesizer_sed(
        self,
        save_name="",
        synthesizer_sed_name="total",
        save_in_aperture=True,
        save_in_det_segmap=True,
        regenerate_original=False,
        overwrite=False,
    ):
        # Function to save a regenerated Synthesizer SED (maybe with overridden parameters) to self.seds and the h5 file
        from unyt import uJy, Angstrom
        from .synthesizer_functions import get_spectra_in_mask

        if save_name != "" and save_name[-1] != "_":
            save_name += "_"

        if regenerate_original:
            self.regenerate_synthesizer_galaxy()

        if (
            self.seds is not None
            and f"{save_name}total_fnu" in self.seds.keys()
            and not overwrite
        ):
            print(
                f"SED with name {save_name}total_fnu already exists in self.seds. Set overwrite=True to overwrite."
            )
            return

        if self.synthesizer_galaxy is not None:
            gal = self.synthesizer_galaxy
            # Get the SED
            seds = {}
            seds["wav"] = (
                gal.stars.spectra[synthesizer_sed_name]
                .obslam.to(Angstrom)
                .value
            )
            seds[f"{save_name}total_fnu"] = {}
            seds[f"{save_name}total_fnu"]["total"] = (
                gal.stars.spectra[synthesizer_sed_name].fnu.to(uJy).value
            )
            if save_in_aperture:
                seds[f"{save_name}0.32as_fnu"] = {}
                seds[f"{save_name}0.32as_fnu"]["total"] = (
                    get_spectra_in_mask(
                        gal,
                        aperture_mask_radii=0.16 * u.arcsec,
                        spectra_type=synthesizer_sed_name,
                    )
                    .to(uJy)
                    .value
                )
            if save_in_det_segmap:
                seds[f"{save_name}det_segmap_fnu"] = {}
                seds[f"{save_name}det_segmap_fnu"]["total"] = (
                    get_spectra_in_mask(
                        gal,
                        pixel_mask=self.det_data["seg"],
                        spectra_type=synthesizer_sed_name,
                        pixel_mask_value="center",
                    )
                    .to(uJy)
                    .value
                )
            self.seds.update(seds)
            self.dump_to_h5()

    def eazy_sed_fit_mock_photometry(
        self,
        sed_name=None,
        scatter_phot=0.1,
        n_proc=1,
        min_percentage_err=0.05,
        template_name="fsps_larson",
        tempfilt=None,
        template_dir="/nvme/scratch/work/tharvey/EAZY/inputs/scripts/",
        errors_based_on_depths=True,
    ):
        """
        This lets you run EAZY on the Synthesizer galaxy photometry - either the internal SED or from the regenerated object.

        """
        if sed_name in self.seds.keys():
            sed = self.seds[sed_name]
            wav = self.seds["wav"] * u.Angstrom
            flux = sed["total"] * u.uJy
            # Get photometry
            phot = self._convolve_sed(flux, wav)
        else:
            if self.synthesizer_galaxy is not None:
                from unyt import uJy, Angstrom

                gal = self.synthesizer_galaxy
                wav = (
                    gal.stars.spectra[sed_name].obslam.to(Angstrom).value
                    * u.Angstrom
                )
                flux = gal.stars.spectra[sed_name].fnu.to(uJy).value * u.uJy
                phot = (
                    gal.stars.photo_fnu[sed_name].photo_fnu.to(uJy).value
                    * u.uJy
                )

                # Can probably get photometry directly.
            else:
                raise ValueError(
                    "Need to provide a valid SED name or first regenerate the Synthesizer galaxy to get the SED from."
                )

        # Add scatter
        # phot = np.random.normal(phot, scatter_phot * phot) * u.uJy
        # print(phot)

        if errors_based_on_depths:
            # Get the depths
            if self.aperture_dict is not None:
                errors = self.aperture_dict["0.32 arcsec"]["flux_err"]
                assert len(errors) == len(
                    phot
                ), "Length of errors does not match length of photometry"

                phot_err = errors * u.Jy
                phot_err = phot_err.to(u.uJy)

                print(f"Errors from {phot_err}")
        else:
            phot_err = min_percentage_err * np.abs(phot)

        mask = ((phot_err / phot) < min_percentage_err) & (phot_err > 0)
        phot_err[mask] = min_percentage_err * phot[mask]

        print(100 * phot_err / phot)

        ez = self.fit_eazy_photometry(
            phot,
            phot_err,
            n_proc=n_proc,
            template_name=template_name,
            template_dir=template_dir,
            meta_details=None,
            save_tempfilt=True,
            load_tempfilt=True,
            save_tempfilt_path="internal",
            tempfilt=tempfilt,
            exclude_bands=[],
        )

        return ez

    def plot_mock_spectra(
        self,
        components,
        fig=None,
        ax=None,
        facecolor="white",
        wav_unit=u.um,
        flux_unit=u.uJy,
        label=True,
        show_phot=False,
        phot_color="black",
        phot_ecolor="black",
        **kwargs,
    ):
        if fig is None:
            fig, ax = plt.subplots(
                1, 1, figsize=(8, 4.5), facecolor=facecolor, dpi=200
            )
            made_fig = True
        else:
            made_fig = False

        if type(components) is str:
            components = [components]

        for component in components:
            if type(label) is str:
                lab = label
            elif label:
                lab = component
            else:
                lab = ""
            wav = self.seds["wav"] * u.Angstrom
            flux = self.seds[component]["total"] * u.uJy
            ax.plot(
                wav.to(wav_unit),
                flux.to(flux_unit, equivalencies=u.spectral_density(wav)),
                **kwargs,
                label=lab,
            )

        if show_phot:
            phot = self._convolve_sed(flux, wav)
            phot_wav = [
                self.filter_wavs[band].to(u.um).value for band in self.bands
            ] * u.um

            ax.scatter(
                phot_wav,
                phot.to(flux_unit, equivalencies=u.spectral_density(phot_wav)),
                color=phot_color,
                edgecolor=phot_ecolor,
                label="True Photometry",
                marker="s",
                zorder=10,
            )

        if label and made_fig:
            ax.set_xlim(0.5, 5)
            max_flux = np.max(flux[(wav > 0.5 * u.um) & (wav < 5 * u.um)])
            # Get max flux in the xlim range
            ax.set_ylim(0, 1.1 * max_flux.to(flux_unit).value)
            # ax.set_ylim(0, 5)
            ax.legend()
            ax.set_xlabel("Wavelength (um)")
            ax.set_ylabel("Flux (uJy)")

        return fig

    def plot_property_maps(
        self,
        parameters=[
            "stellar_mass",
            "sfr_10 Myr",
            "ssfr_10 Myr",
            "stellar_age",
            "stellar_metallicity",
        ],
        parameter_units={
            "stellar_mass": u.Msun,
            "sfr": u.Msun / u.yr,
            "ssfr": u.yr**-1,
            "stellar_age": u.Gyr,
            "stellar_metallicity": u.dimensionless_unscaled,
        },
        save=False,
        facecolor="white",
        max_on_row=4,
        weight_mass_sfr=True,
        norm="linear",
        logmap=[True, True, True, True, False],
        total_params=[],
    ):
        cmaps = [
            "magma",
            "RdYlBu",
            "cmr.ember",
            "cmr.cosmic",
            "cmr.lilac",
            "cmr.eclipse",
            "cmr.sapphire",
            "cmr.dusk",
            "cmr.emerald",
        ]

        fig, axes = plt.subplots(
            len(parameters) // max_on_row + 1,
            max_on_row,
            figsize=(
                2.5 * max_on_row,
                2.5 * (len(parameters) // max_on_row + 1),
            ),
            constrained_layout=True,
            facecolor=facecolor,
            sharex=True,
            sharey=True,
        )
        # add gap between rows using get_layout_engine
        fig.get_layout_engine().set(h_pad=4 / 72, hspace=0.2)

        axes = axes.flatten()
        # Remove empty axes
        for i in range(len(parameters), len(axes)):
            fig.delaxes(axes[i])

        # These should really be computed at the best-fit Bagpipes redshift, which is not necessarily the same as the input redshift.
        redshift = self.redshift

        if type(norm) is str:
            norm = [norm] * len(parameters)

        for i, param in enumerate(parameters):
            ax_divider = make_axes_locatable(axes[i])
            cax = ax_divider.append_axes("top", size="5%", pad="2%")

            map = self.property_images[param]

            # replace inf with nan
            map = np.nan_to_num(map, nan=np.nan, posinf=np.nan, neginf=np.nan)

            if logmap[i]:
                map = np.log10(map)

            map = np.nan_to_num(map, nan=np.nan, posinf=np.nan, neginf=np.nan)

            r"""
            if param in ['stellar_mass', 'sfr']:
                ref_band = {'stellar_mass':'F444W', 'sfr':'1500A'}
                print(param, np.nanmin(map), np.nanmax(map))

                if weight_mass_sfr:
                    weight = ref_band[param]
                else:
                    weight = False
                

                map = self.map_to_density_map(map, redshift = redshift, weight_by_band=weight, logmap = True)
                #map = self.map_to_density_map(map, redshift = redshift, logmap = True) 
                log = '$\log_{10}$ '
                param = f'{param}_density'
            """

            if norm[i] == "log":
                anorm = LogNorm(vmin=np.nanmin(map), vmax=np.nanmax(map))
            else:
                anorm = Normalize(vmin=np.nanmin(map), vmax=np.nanmax(map))

            r"""
            gunit = self.param_unit(param.split(':')[-1])
            unit = f' ({log}{gunit:latex})' if gunit != u.dimensionless_unscaled else ''
            param_str = param.replace("_", r"\ ")
            
            for pos, total_param in enumerate(total_params):
                if total_param in table['#ID']:
                    value = table[table['#ID'] == total_param][param_name]
                    if param_name.endswith('_50'):
                        err_up = table[table['#ID'] == total_param][param_name.replace('50', '84')] - value
                        err_down = value - table[table['#ID'] == total_param][param_name.replace('50', '16')]
                        
                        berr = f'$^{{+{err_up[0]:.2f}}}_{{-{err_down[0]:.2f}}}$'
                    else:
                        berr = ''

                    if 0 < norm(value) < 1:
                        c_map = plt.cm.get_cmap(cmaps[i])
                        color = c_map(norm(value))
                    else:
                        color = 'black'
                    
                    axes[i].text(0.5, 0.03+0.1*pos, f'{total_param}:{value[0]:.2f}{berr} {unit}', ha='center', va='bottom', fontsize=8, transform=axes[i].transAxes, color = color, path_effects=[PathEffects.withStroke(linewidth=0.2, foreground='black')])
            """

            # Create actual normalisation

            mappable = axes[i].imshow(
                map,
                origin="lower",
                interpolation="none",
                cmap=cmaps[i],
                norm=anorm,
            )
            cbar = fig.colorbar(mappable, cax=cax, orientation="horizontal")

            # ensure cbar is using ScalarFormatter
            cbar.ax.xaxis.set_major_formatter(ScalarFormatter())
            cbar.ax.xaxis.set_minor_formatter(ScalarFormatter())

            param_str = param.replace("_", r"\ ")
            unit = parameter_units[param]
            cbar.set_label(
                rf"$\rm{{{param_str}}}${unit}", labelpad=10, fontsize=8
            )
            cbar.ax.xaxis.set_ticks_position("top")
            cbar.ax.xaxis.set_tick_params(labelsize=6, which="both")
            cbar.ax.xaxis.set_label_position("top")
            # disable xtick labels
            # cax.set_xticklabels([])
            # Fix colorbar tick size and positioning if log
            if norm == "log":
                # Generate reasonable ticks
                if logmap[i]:
                    ticks = np.linspace(np.nanmin(map), np.nanmax(map), num=5)
                else:
                    ticks = np.logspace(
                        np.log10(np.nanmin(map)),
                        np.log10(np.nanmax(map)),
                        num=5,
                    )
                cbar.set_ticks(ticks)
                cbar.ax.xaxis.set_major_formatter(ScalarFormatter())
                cbar.ax.xaxis.set_minor_formatter(ScalarFormatter())
                cbar.ax.xaxis.set_tick_params(labelsize=6, which="both")
                cbar.ax.xaxis.set_label_position("top")
                cbar.ax.xaxis.set_ticks_position("top")
                cbar.update_ticks()

        if save:
            fig.savefig(
                f"{resolved_galaxy_dir}/{run_name}_mock_maps.png",
                dpi=300,
                bbox_inches="tight",
            )

        return fig

    def regenerate_synthesizer_galaxy(
        self,
        file_path=None,
        basic=False,
        grid_dir="/nvme/scratch/work/tharvey/synthesizer/grids/",
        override_internal_model={},
    ):
        # Regenerate the synthesizer galaxy

        if self.synthesizer_galaxy is not None:
            return self.synthesizer_galaxy

        try:
            from scipy import signal
            from synthesizer.emission_models import (
                IncidentEmission,
                PacmanEmission,
                IntrinsicEmission,
                TotalEmission,
                # CharlottFall2000,
                # BiModalPacmanEmission,
            )
            from synthesizer.emission_models.attenuation import (
                PowerLaw,
                Calzetti2000,
            )
            from synthesizer.emission_models.attenuation.igm import (
                Inoue14,
                Madau96,
            )
            from synthesizer.emission_models.dust.emission import Greybody
            from synthesizer.filters import FilterCollection as Filters
            from synthesizer.grid import Grid
            from synthesizer.imaging.image import Image
            from synthesizer.kernel_functions import Kernel
            from synthesizer.particle.galaxy import Galaxy
            from unyt import (
                Angstrom,
                Gyr,
                Hz,
                K,
                Mpc,
                Msun,
                Myr,
                arcsecond,
                erg,
                kpc,
                nJy,
                s,
                uJy,
                um,
                unyt_array,
                yr,
            )

            from .synthesizer_functions import (
                apply_pixel_coordinate_mask,
                convert_coordinates,
                get_spectra_in_mask,
            )

        except ImportError as e:
            print(e)
            raise Exception(
                "Synthesizer not installed. Please clone from Github and install using pip install ."
            )

        grid = Grid(
            grid_name=self.meta_properties["grid"],
            grid_dir=grid_dir,
        )

        # Get the synthesizer galaxy

        filter_codes = self.meta_properties.get("filters", None)

        if filter_codes is None:
            # print(
            #    "No filter codes found in meta_properties. Using default filters."
            # )
            observatories = 5 * ["HST"] + 14 * ["JWST"]
            instruments = 5 * ["ACS_WFC"] + 14 * ["NIRCam"]
            bands = self.bands

            filter_codes = [
                f"{obs}/{inst}.{band}"
                for obs, inst, band in zip(observatories, instruments, bands)
            ]
        filters = Filters(filter_codes, new_lam=grid.lam)

        filter_code_from_band = {
            band: code for band, code in zip(self.bands, filter_codes)
        }

        gal_id = self.meta_properties["id"]
        tag = self.meta_properties["tag"]
        region = self.meta_properties["region"]

        if int(region) == 0:
            region = "00"  # 00 is getting lost
        if file_path is None:
            file_path = self.meta_properties["model_assumptions"]["file_path"]

        zed = float(tag[5:].replace("p", "."))

        with h5.File(file_path, "r") as hf:  # opening the hdf5 file
            # coordinates of the stellar particles
            coordinates = (
                np.array(
                    hf[f"{tag}/{region}/{gal_id}"].get("coordinates"),
                    dtype=np.float64,
                )
                * Mpc
            )
            # initial masses of the stellar particles
            initial_masses = (
                np.array(
                    hf[f"{tag}/{region}/{gal_id}"].get("initial_masses"),
                    dtype=np.float64,
                )
                * Msun
            )
            # current masses of the stellar particles, mass change due to mass loss as stars age
            current_masses = (
                np.array(
                    hf[f"{tag}/{region}/{gal_id}"].get("initial_masses"),
                    dtype=np.float64,
                )
                * Msun
            )
            # ages of stars in log10, which is in units of years
            log10ages = np.array(
                hf[f"{tag}/{region}/{gal_id}"].get("log10ages"),
                dtype=np.float64,
            )
            # metallicities of the stars in log10
            log10metallicities = np.array(
                hf[f"{tag}/{region}/{gal_id}"].get("log10metallicities"),
                dtype=np.float64,
            )
            # optical depth in the V-band, it is the same prescription as shown in Vijayan+21, which
            # includes the V-band dust optical depth due to the diffuse ISM dust and birth cloud dust
            tau_v = np.array(
                hf[f"{tag}/{region}/{gal_id}"].get("tau_v"), dtype=np.float64
            )
            # the smoothing lengths of star particles
            smoothing_lengths = (
                np.array(
                    hf[f"{tag}/{region}/{gal_id}"].get("smoothing_lengths"),
                    dtype=np.float64,
                )
                * Mpc
            )

            gal = Galaxy(redshift=zed)
            gal.load_stars(
                initial_masses=initial_masses,
                ages=10**log10ages * yr,
                metallicities=10**log10metallicities,
                coordinates=coordinates,
                current_masses=current_masses,
                smoothing_lengths=smoothing_lengths,
                centre=np.zeros(3) * Mpc,
            )

        gal.tau_v = tau_v

        if not basic:
            model_assumptions = copy.deepcopy(
                self.meta_properties["model_assumptions"]
            )
            model_assumptions.update(override_internal_model)

            if type(model_assumptions["dust_temp"]) is tuple:
                model_assumptions["dust_temp"] = float(
                    model_assumptions["dust_temp"][0]
                )

            if "dust_type" in model_assumptions.keys():
                if model_assumptions["dust_type"] == "pacman":
                    dust_object = PacmanEmission

                elif model_assumptions["dust_type"] == "charlotfall2000":
                    dust_object = CharlottFall2000
                elif model_assumptions["dust_type"] == "bimodal_pacman":
                    dust_object = BiModalPacmanEmission

            else:
                dust_object = IntrinsicEmission

            fesc_ly_alpha = model_assumptions.get("fesc_ly_alpha", 1.0)
            print(f"fesc_ly_alpha: {fesc_ly_alpha}")
            fesc = model_assumptions.get("fesc", 0.0)

            if model_assumptions["dust_curve"] == "power_law":
                dust_curve = PowerLaw(slope=model_assumptions["dust_slope"])
            elif model_assumptions["dust_curve"] == "calzetti2000":
                slope = model_assumptions.get("dust_slope", 0.0)
                bump_amplitude = model_assumptions.get(
                    "dust_bump_amplitude", 0.0
                )
                bump_wavelength = (
                    model_assumptions.get("dust_bump_wavelength", 0.2175) * um
                )
                fwhm_bump = model_assumptions.get("dust_fwhm_bump", 0.035) * um
                dust_curve = Calzetti2000(
                    ampl=bump_amplitude,
                    cent_lam=bump_wavelength,
                    slope=slope,
                    gamma=fwhm_bump,
                )
            else:
                dust_curve = None

            if "dust_emission" in model_assumptions.keys():
                if model_assumptions["dust_emission"] == "greybody":
                    dust_emission = Greybody(
                        temperature=model_assumptions["dust_temp"] * K,
                        emissivity=model_assumptions["dust_beta"],
                    )
                if model_assumptions["dust_emission"] == "blackbody":
                    dust_emission = Blackbody(
                        temperature=model_assumptions["dust_temp"] * K
                    )
            else:
                dust_emission = None

            if "igm" in model_assumptions.keys():
                if model_assumptions["igm"] == "inoue14":
                    igm = Inoue14
                elif model_assumptions["igm"] == "madau96":
                    igm = Madau96

            if "cosmo" in model_assumptions.keys():
                cosmo = model_assumptions["cosmo"]
                # Check if cosmo is a string or a Cosmology object
                if cosmo.startswith("FlatLambdaCDM"):
                    from astropy.cosmology import FlatLambdaCDM

                    H0 = float(cosmo.split("H0=")[1].split(" ")[0])
                    Om0 = float(cosmo.split("Om0=")[1].split(",")[0])

                    cosmo = FlatLambdaCDM(H0=H0, Om0=Om0)

            emission_model = dust_object(
                grid=grid,
                tau_v=gal.tau_v,
                dust_curve=dust_curve,
                dust_emission=dust_emission,
                fesc_ly_alpha=fesc_ly_alpha,
                fesc=fesc,
                per_particle=True,
            )

            gal.stars.get_particle_spectra(emission_model)
            models = gal.stars.particle_spectra.keys()
            for model in models:
                # Generate spectra for each particle for each model component
                gal.stars.particle_spectra[model].get_fnu(
                    cosmo, gal.redshift, igm=igm
                )
                # Need this for making luminosity images only
                gal.stars.particle_spectra[model].get_photo_lnu(filters)

            # Combines the spectra of all the particles
            gal.stars.integrate_particle_spectra()

            gal.stars.get_particle_photo_fnu(filters)

            gal.stars.get_photo_fnu(filters)
            # gal.stars.get_photo_lnu(filters) # unneeded apparently

        self.synthesizer_galaxy = gal

    def property_map_from_synthesizer(self):
        pass

    def synthesizer_property_in_mask(
        self,
        property,
        mask=None,
        mask_name=None,
        func="sum",
        density=False,
        return_total=False,
    ):
        # Wrapper for self.property_in_mask for Synthesizer specific properties
        # Which should be saved in the h5 file.

        # E.g. this should be able to get the stellar mass in a mask, or the SFR in a mask, or average age in a mask etc.
        if property not in self.property_images.keys():
            if getattr(self, "synthesizer_galaxy") is None:
                self.regenerate_synthesizer_galaxy()
            print("Don't have property map. Generating from synthesizer.")

        assert (
            mask is not None or mask_name is not None
        ), "Need to provide either a mask or a mask_name"

        if mask_name == "pixedfit":
            mask = self.pixedfit_map

        elif mask_name == "pixel_by_pixel":
            mask = self.pixel_by_pixel_map

        elif mask_name == "voronoi":
            mask = self.voronoi_map

        elif mask_name in self.gal_region.keys():
            mask = self.gal_region[mask_name]

        elif hasattr(self, mask_name):
            mask = getattr(self, mask_name)

        else:
            raise ValueError(f"Mask name {mask_name} not recognised.")

        return self.property_in_mask(
            self.property_images[property],
            mask,
            func=func,
            density=density,
            return_total=return_total,
        )

    def plot_overview(
        self,
        figsize=(12, 8),
        bands_to_show=["F814W", "F115W", "F200W", "F335M", "F444W"],
        bins_to_show=["TOTAL_BIN", "MAG_APER_TOTAL", "1"],
        show=True,
        flux_unit=u.ABmag,
        save=False,
        rgb_stretch=0.001,
        rgb_q=0.001,
        binmap_type="pixedfit",
        min_sed_flux=31 * u.ABmag,
    ):
        # Call parent method

        fig = super().plot_overview(
            figsize=figsize,
            bands_to_show=bands_to_show,
            bins_to_show=bins_to_show,
            show=show,
            flux_unit=flux_unit,
            save=False,
            legend=False,
            binmap_type=binmap_type,
            rgb_stretch=rgb_stretch,
            rgb_q=rgb_q,
            min_sed_flux=min_sed_flux,
        )

        # Get axes and plot the SED

        axes = fig.get_axes()
        ax = axes[0]

        ax.set_xlim(ax.get_xlim())
        ax.set_ylim(ax.get_ylim())

        self.plot_mock_spectra(
            components=["det_segmap_fnu"],
            fig=fig,
            ax=ax,
            facecolor="white",
            wav_unit=u.um,
            flux_unit=flux_unit,
            show_phot=True,
            phot_color="teal",
            zorder=2,
            color="teal",
        )

        props = dict(boxstyle="square", facecolor="white", alpha=0.5)

        mask_mass = self.synthesizer_property_in_mask(
            "stellar_mass",
            mask_name="pixedfit",
            density=False,
            return_total=True,
        )
        if type(mask_mass) is u.Quantity:
            mask_mass = mask_mass.value

        total_mass = float(self.meta_properties["stellar_mass"])

        textstr = f"True Properties\nRedshift: {self.redshift:.2f}\nM$_\star (tot)$:{np.log10(total_mass):.2f} M$_\odot$\nM$_\star (seg)$:{np.log10(mask_mass):.2f} M$_\odot$"
        fig.text(
            0.5,
            0.45,
            textstr,
            transform=fig.transFigure,
            fontsize=13,
            verticalalignment="top",
            bbox=props,
        )

        ax.legend(frameon=False)

        if save:
            if not os.path.exists(f"{resolved_galaxy_dir}/diagnostic_plots"):
                os.makedirs(f"{resolved_galaxy_dir}/diagnostic_plots")

            fig.savefig(
                f"{resolved_galaxy_dir}/diagnostic_plots/{self.survey}_{self.galaxy_id}_overview.png",
                dpi=200,
            )

        return fig

        # Plot

    def plot_bagpipes_sfh(
        self,
        run_name=None,
        bins_to_show=["RESOLVED"],
        save=False,
        facecolor="white",
        marker_colors="black",
        time_unit="Gyr",
        cmap="viridis",
        plotpipes_dir="pipes_scripts/",
        run_dir="pipes/",
        plottype="lookback",
        cache=None,
        fig=None,
        axes=None,
        add_true=True,
        mask="pixedfit",
    ):
        # Call parent method

        fig, cache = super().plot_bagpipes_sfh(
            run_name=run_name,
            bins_to_show=bins_to_show,
            save=save,
            facecolor=facecolor,
            marker_colors=marker_colors,
            time_unit=time_unit,
            cmap=cmap,
            plotpipes_dir=plotpipes_dir,
            run_dir=run_dir,
            plottype="lookback",
            cache=cache,
            fig=fig,
            axes=axes,
        )
        if add_true:
            axes = fig.get_axes()
            if len(axes) == 1:
                axes = axes[0]
                # Check if 'True SFH' is in the legend
                if "True SFH" not in [l.get_label() for l in axes.get_lines()]:
                    # Plot the true SFH
                    self.plot_real_sfh(
                        mask={"attr": "gal_region", "key": "pixedfit"},
                        ax=axes,
                        fig=fig,
                        time_unit=time_unit,
                        plottype=plottype,
                        mask_str=mask,
                    )

        return fig, cache

    def plot_real_sfh(
        self,
        mask=None,
        ax=None,
        fig=None,
        mask_str=None,
        time_unit="Gyr",
        plottype="lookback",
        overwrite=False,
    ):
        if self.sfh is not None and not overwrite:
            if mask_str in self.sfh.keys():
                print("Found SFH in cache. Not regenerating.")
                time, sfh = self.sfh[mask_str][:, 0], self.sfh[mask_str][:, 1]

        else:
            if mask is not None:
                if type(mask) is dict:
                    d = getattr(self, mask["attr"])
                    mask = d[mask["key"]]

                if type(mask) is str:
                    mask_str = copy.copy(mask)

                mask = mask.astype(bool)

            from .synthesizer_functions import calculate_sfh

            if self.synthesizer_galaxy is None:
                self.regenerate_synthesizer_galaxy(basic=True)

            time, sfh = calculate_sfh(self.synthesizer_galaxy, pixel_mask=mask)

            time = time.to_astropy().to(u.Gyr).value
            sfh = sfh.to_astropy().to(u.Msun / u.yr).value

            # add to h5
            savedata = np.vstack([time, sfh]).T

            if mask_str is None:
                mask_str = "total"

            self.add_to_h5(savedata, "mock_galaxy/sfh", mask_str)

            if self.sfh is None:
                self.sfh = {}

            self.sfh[mask_str] = savedata

        if plottype == "absolute":
            # Convert time to absolute time given looking back from self.redshift
            time = cosmo.lookback_time(self.redshift).to(u.Gyr).value - time

        time *= u.Gyr
        time = time.to(time_unit).value

        if fig is None:
            fig, ax = plt.subplots(
                1, 1, figsize=(5, 5), facecolor="white", dpi=200
            )

        ax.plot(time, sfh, linestyle="solid", color="black", label="True SFH")

        return fig


class MethodForwardingMeta(type):
    def __new__(cls, name, bases, attrs):
        def create_forwarding_method(method_name):
            def forwarding_method(self, *args, **kwargs):
                results = []
                for obj in self.objects:
                    if hasattr(obj, method_name):
                        method = getattr(obj, method_name)
                        results.append(method(*args, **kwargs))
                return results

            return forwarding_method

        new_class = super().__new__(cls, name, bases, attrs)

        # Get methods from the contained class
        contained_class = attrs.get("contained_class", None)
        if contained_class:
            for method_name in dir(contained_class):
                if not method_name.startswith("__") and isinstance(
                    getattr(contained_class, method_name), types.FunctionType
                ):
                    setattr(
                        new_class,
                        method_name,
                        create_forwarding_method(method_name),
                    )

        return new_class


# Make ResolvedGalaxies inherit from a numpy array so than self.galaxies behaves like a numpy array
class ResolvedGalaxies(np.ndarray):
    def __new__(cls, list_of_galaxies, *args, **kwargs):
        return np.array(list_of_galaxies, dtype=object).view(cls)

    """
    def __init__(self, list_of_galaxies):
        contained_class = list_of_galaxies[0].__class__

        self.galaxies = list_of_galaxies
    """

    def __array_finalize__(self, obj):
        if obj is None:
            return

        self.galaxies = self  # obj.galaxies

    @classmethod
    def init(cls, list_of_ids, survey, version, **kwargs):
        list_of_galaxies = []
        for gal_id in list_of_ids:
            if "mock" in str(gal_id):
                galaxy = MockResolvedGalaxy.init(
                    galaxy_id=gal_id,
                    mock_survey=survey,
                    mock_version=version,
                    **kwargs,
                )
            else:
                galaxy = ResolvedGalaxy.init(
                    galaxy_id=gal_id, survey=survey, version=version, **kwargs
                )
            list_of_galaxies.append(galaxy)

        return cls(list_of_galaxies)

    """
    def __getitem__(self, key):
        return self.galaxies[key]

    def __len__(self):
        return len(self.galaxies)

    def __iter__(self):
        return iter(self.galaxies)

    def __next__(self):
        return next(self.galaxies)
    """

    def total_number_of_bins(self, **kwargs):
        return sum(
            [galaxy.get_number_of_bins(**kwargs) for galaxy in self.galaxies]
        )

    def comparison_plot(
        self,
        aperture_run,
        resolved_run,
        parameter="stellar_mass",
        sed_fitting_tool="bagpipes",
        aperture_label="TOTAL_BIN",
        filter_single_bins_for_map=None,  # This lets you not plot single bins for a given input map.
        label=False,
        color_by=None,
        xlim=None,
        ylim=None,
        cmap="viridis",
        n_jobs=1,
        overwrite=False,
        fig=None,
        ax=None,
        norm=None,
        add_colorbar=True,
        **kwargs,
    ):
        if fig is None:
            fig, ax = plt.subplots(
                1, 1, figsize=(5, 5), facecolor="white", dpi=200
            )
        if ax is None:
            ax = fig.add_subplot(111)

        if parameter == "stellar_mass":
            ax.set_xlabel(
                rf"Integrated Stellar Mass (M$_\odot$) [{aperture_run}]"
            )
            ax.set_ylabel(
                rf"Resolved Stellar Mass (M$_\odot$) [{resolved_run}]"
            )

            # Plot a 1:1 line

            if xlim is None:
                ax.set_xlim(6.5, 10)
            if ylim is None:
                ax.set_ylim(6.5, 10)

            # ax.plot([5, 12], [5, 12], "k--")

        if color_by is not None:
            if add_colorbar:
                cbar = make_axes_locatable(ax).append_axes(
                    "top", size="5%", pad="2%"
                )
                # Move x-axis to top for cbar
                cbar.xaxis.set_ticks_position("top")
                cbar.xaxis.set_label_position("top")

            vals = []
            colors = {}
            if color_by.startswith("int_"):
                for galaxy in self.galaxies:
                    table = galaxy.sed_fitting_table[sed_fitting_tool][
                        aperture_run
                    ]
                    table = table[table["#ID"] == aperture_label]
                    if color_by[4:] == "burstiness":
                        value = float(table["sfr_10myr_50"]) / float(
                            table["sfr_50"]
                        )
                        vals.append(value)
                    else:
                        vals.append(float(table[color_by[4:]][0]))
                if add_colorbar:
                    cbar.set_xlabel(color_by[4:])

            if color_by == "nbins":
                vals = [
                    galaxy.get_number_of_bins() for galaxy in self.galaxies
                ]
                if add_colorbar:
                    cbar.set_xlabel("Number of Bins")

            if color_by == "redshift":
                vals = [galaxy.redshift for galaxy in self.galaxies]

            if color_by.startswith("kron_radius"):
                band = color_by.split("_")[-1]
                if band == "DET":
                    band = "detection"
                vals = []
                for galaxy in self.galaxies:
                    a, b, _ = galaxy.plot_kron_ellipse(
                        band=band, ax="", center="", return_params=True
                    )
                    radii = np.sqrt(a * b)
                    vals.append(radii)

            norm = (
                Normalize(vmin=min(vals), vmax=max(vals))
                if norm is None
                else norm
            )
            import matplotlib.cm as cm

            cmap = cm.get_cmap(cmap)

            colors = {
                galaxy.galaxy_id: cmap(norm(val))
                for galaxy, val in zip(self.galaxies, vals)
            }
        else:
            colors = {galaxy.galaxy_id: "black" for galaxy in self.galaxies}

        for galaxy in self.galaxies:
            integrated, resolved = [], []
            if filter_single_bins_for_map is not None:
                nbins = galaxy.get_number_of_bins(filter_single_bins_for_map)
                if nbins == 1:
                    continue

            for run, label in zip(
                [aperture_run, resolved_run], [aperture_label, ""]
            ):
                if (
                    run in galaxy.sed_fitting_table[sed_fitting_tool]
                    and label
                    in galaxy.sed_fitting_table[sed_fitting_tool][run]["#ID"]
                ):
                    table = galaxy.sed_fitting_table[sed_fitting_tool][run]
                    table = table[table["#ID"] == label]
                    assert (
                        len(table) == 1
                    ), f"Aperture label {label} not found in table or table is not unique: length {len(table)}"

                    param_16, param_50, param_84 = (
                        table[f"{parameter}_16"],
                        table[f"{parameter}_50"],
                        table[f"{parameter}_84"],
                    )

                    integrated.append((param_50, param_16, param_84))
                else:
                    # resolved mass
                    logp = True if parameter == "stellar_mass" else False
                    quantities = galaxy.get_total_resolved_property(
                        run,
                        sed_fitting_tool=sed_fitting_tool,
                        n_cores=n_jobs,
                        property=parameter,
                        log=logp,
                        overwrite=overwrite,
                    )

                    param_16, param_50, param_84 = quantities
                    resolved.append((param_50, param_16, param_84))

            if len(resolved) == 1 and len(integrated) == 1:
                x = integrated[0]
                y = resolved[0]
            elif len(resolved) == 2 and len(integrated) == 0:
                x, y = resolved[1], resolved[0]
                ax.set_xlabel(
                    rf"Resolved Stellar Mass (M$_\odot$) [{aperture_run}]"
                )

            elif len(resolved) == 0 and len(integrated) == 2:
                x, y = integrated[1], integrated[0]
                ax.set_ylabel(
                    rf"Integrated Stellar Mass (M$_\odot$) [{resolved_run}]"
                )
            else:
                raise ValueError(
                    "Resolved and integrated properties must be unique"
                )

            # Force xerr and yerr to always be (2, n ) shape using np.atleast_2d and np.squeeze
            xerr = [
                [np.squeeze(np.array(x[0] - x[1]))],
                [np.squeeze(np.array(x[2] - x[0]))],
            ]
            yerr = [
                [np.squeeze(np.array(y[0] - y[1]))],
                [np.squeeze(np.array(y[2] - y[0]))],
            ]

            ax.errorbar(
                x[0],
                y[0],
                xerr=xerr,
                yerr=yerr,
                fmt="o",
                label=galaxy.galaxy_id if label else "",
                color=colors[galaxy.galaxy_id],
                **kwargs,
            )
            if color_by is not None and add_colorbar:
                fig.colorbar(
                    cm.ScalarMappable(norm=norm, cmap=cmap),
                    cax=cbar,
                    orientation="horizontal",
                )
                # Move x-axis to top for cbar
                cbar.xaxis.set_ticks_position("top")
                cbar.xaxis.set_label_position("top")
                cbar.set_xlabel(color_by)

        if label:
            ax.legend()

        # Plot a 1:1 line for any parameter
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax.plot(
            [min(xlim[0], ylim[0]), max(xlim[1], ylim[1])],
            [min(xlim[0], ylim[0]), max(xlim[1], ylim[1])],
            "k--",
        )

        return fig

    def scatter_plot(
        self,
        x_attr,
        y_attr,
        c_attr=None,
        facecolor="white",
        cmap="viridis",
        **kwargs,
    ):
        fig, ax = plt.subplots(1, 1, figsize=(5, 5), facecolor=facecolor)

        if "x_label" in kwargs:
            ax.set_xlabel(kwargs["x_label"])
            kwargs.pop(x_label)

        if "y_label" in kwargs:
            ax.set_ylabel(kwargs["y_label"])
            kwargs.pop(y_label)

        if "c_label" in kwargs:
            cbar_label = kwargs["c_label"]
            kwargs.pop(c_label)
        else:
            cbar_label = None

        if c_attr is not None:
            cbar = make_axes_locatable(ax).append_axes(
                "top", size="5%", pad="2%"
            )

        for galaxy in self.galaxies:
            x = getattr(galaxy, x_attr)
            y = getattr(galaxy, y_attr)
            if c_attr is not None:
                c = getattr(galaxy, c_attr)
                assert (
                    len(x) == len(y) == len(c)
                ), "x, y and c must have same length"
            else:
                assert len(x) == len(y), "x and y must have same length"

            ax.scatter(x, y, c=c, cmap=cmap, **kwargs)

        return fig

    # Run any function on all galaxies
    def run_function(self, function, *args, **kwargs):
        for galaxy in self.galaxies:
            if type(function) is str:
                rfunction = getattr(galaxy, function)
            else:
                rfunction = function
            rfunction(*args, **kwargs)

    def filter_single_bins(self, binmap):
        bins = np.array(
            [galaxy.get_number_of_bins(binmap) for galaxy in self.galaxies]
        )
        # return a view of the galaxy array with galaxies which have > 1 bin
        return self[bins > 1]

    def run_bagpipes_parallel(
        self,
        bagpipes_configs,
        filt_dir=bagpipes_filter_dir,
        fit_photometry="all",
        run_dir="pipes/",
        n_jobs=8,
        out_subdir_name="parallel_temp",
        load_only=False,  # If set to true, will force reloading of properties and skip running
        properties_to_load=["stellar_mass", "sfr", "sfr_10myr"],
        alert=True,  # If crash, email me!
        **kwargs,
    ):
        """
         Convenience function to run a set of run_dicts in parallel, and save results into one catalogue.
        Can differ in terms of photo-z, priors etc but should be the same overall model
        Better for mpirun.

        Steps:
            Get list of IDs from each galaxy that will be fit.
            Call get_bagpipes_phot for each and store it.
            Store each input argument for bagpipes in a file (and generate new unique IDs)
            Trigger own seperate mpirun bash script to load in all arguments and IDs, without class overhead.
            Will need to provide its own singular function to provide photometry given an ID.
            After run, move things back to where they need to be and seperate catalogue out again.

        """
        import json
        import tempfile
        import subprocess
        import os

        try:
            assert all(
                bagpipes_configs[0]["meta"]["run_name"]
                == config["meta"]["run_name"]
                for config in bagpipes_configs
            ), "All bagpipes configs must have the same run_name (and therefore same overall model)"

            run_name = bagpipes_configs[0]["meta"]["run_name"]

            if type(bagpipes_configs) is dict:
                bagpipes_configs = [
                    copy.deepcopy(bagpipes_configs)
                    for _ in range(len(self.galaxies))
                ]

            configs = {
                galaxy.galaxy_id: galaxy.run_bagpipes(
                    bagpipes_config,
                    filt_dir=filt_dir,
                    fit_photometry=fit_photometry,
                    run_dir=run_dir,
                    return_run_args=True,
                    **kwargs,
                )
                for galaxy, bagpipes_config in zip(
                    self.galaxies, bagpipes_configs
                )
            }

            # drop configs that are None

            remove_keys = []
            for key, config in configs.items():
                if config is None:
                    remove_keys.append(key)

            for key in remove_keys:
                configs.pop(key)

            # Check if all outputs exist in SED fitting table

            if (
                all(
                    [
                        "bagpipes" in galaxy.sed_fitting_table.keys()
                        and run_name
                        in galaxy.sed_fitting_table["bagpipes"].keys()
                        for galaxy in self.galaxies
                    ]
                )
                and not load_only
            ):
                print("All galaxies already run, skipping.")
                return

            assert all(
                type(config) in [dict, None] for config in configs.values()
            ), f"All configs must be dictionaries, got {[type(config) for config in configs.values()]}"
            # Dump configs to json

            # check if all None
            if len(configs) == 0:
                print("All configs are None, skipping.")
                return

            file = tempfile.NamedTemporaryFile(delete=False)
            file_path = file.name

            delete = []
            write_configs = copy.deepcopy(configs)
            for key, config in write_configs.items():
                if config["already_run"]:
                    print(f"Skipping {key} as already run.")
                    delete.append(key)

            for key in delete:
                write_configs.pop(key)

            done = False
            if len(write_configs) == 0:
                print("All configs already run, skipping.")
                done = True

            with open(file_path, "w") as f:
                json.dump(write_configs, f)

            # Run the mpirun script

            # Get path of script

            script_path = os.path.abspath(__file__).replace(
                "ResolvedGalaxy.py", "bagpipes/run_bagpipes_parallel.py"
            )
            # get path of 'python' executable

            run_dir = os.path.abspath(run_dir)  # .replace("pipes", "")

            # cd to run_dir
            os.chdir(os.path.dirname(run_dir))

            # Check if already run
            # If seperate configs exist or if a single catalogue exists, then it is done
            done = done | all(
                [
                    os.path.exists(
                        f"{os.path.join(run_dir, config['out_dir']).replace('posterior', 'cats')}/{galaxy_id}.fits"
                    )
                    for galaxy_id, config in configs.items()
                ]
            ) or os.path.exists(
                os.path.join(run_dir, f"cats/{out_subdir_name}.fits")
            )
            # also check if .h5 files exist in either location
            for galaxy_id, config in configs.items():
                # Check all IDs and see if .h5 exists in either new or old location
                cat_ids = [f"{galaxy_id}_{id}" for id in config["ids"]]
                for cat_id, gal_id in zip(cat_ids, config["ids"]):
                    new_path = os.path.join(
                        run_dir, config["out_dir"], f"{gal_id}.h5"
                    )
                    old_path = os.path.join(
                        run_dir, f"posterior/{out_subdir_name}/{cat_id}.h5"
                    )

                    if not (
                        os.path.exists(old_path) or os.path.exists(new_path)
                    ):
                        done = False
                        print(f"Missing .h5 file for {cat_id}")
                        print(new_path)
                        print(old_path)
                        break

            n_fits = int(
                np.sum(
                    np.ones(len(configs))
                    * np.array(
                        [len(config["ids"]) for config in configs.values()]
                    )
                )
            )

            if not load_only and not done:
                print(f"Starting mpi process with {n_jobs} cores.")
                print(f"Run directory: {run_dir}")

                process_args = [
                    "mpirun",
                    "-n",
                    str(n_jobs),
                    "python",
                    script_path,
                    file_path,
                    out_subdir_name,
                ]
                print(" ".join(process_args))
                # Run and block until finished, check for errors

                process = subprocess.Popen(
                    process_args,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    universal_newlines=True,
                    env=os.environ,
                )

                for line in iter(process.stdout.readline, ""):
                    print(
                        line, end="", flush=True
                    )  # end='' because the line already contains a newline
                    sys.stdout.flush()  # Ensure output is printed immediately

                process.stdout.close()
                return_code = process.wait()

                if return_code != 0:
                    raise subprocess.CalledProcessError(
                        return_code, process_args
                    )

            else:
                print("Already run, skipping.")
            # Folder for posteriors is config[galaxy]['out_dir']

            # Current folder will be run_dir /pipes/parallel_temp

            current_posterior_folder = os.path.join(
                run_dir, f"posterior/{out_subdir_name}"
            )

            # Move all files back to where they should be and rename - currently gal_id_bin.h5, should be bin.h5 only

            output_catalogue = os.path.join(
                run_dir, f"cats/{out_subdir_name}.fits"
            )
            done = False

            print(
                [
                    f"{os.path.join(run_dir, config['out_dir']).replace('posterior', 'cats')}.fits"
                    for galaxy_id, config in configs.items()
                ]
            )

            if all(
                [
                    os.path.exists(
                        f"{os.path.join(run_dir, config['out_dir']).replace('posterior', 'cats')}.fits"
                    )
                    for galaxy_id, config in configs.items()
                ]
            ):
                print("Individual galaxy catalogues found. Skipping.")
                done = True

            elif os.path.exists(output_catalogue):
                output_catalogue = Table.read(output_catalogue)
                assert (
                    len(output_catalogue) == n_fits
                ), f"Catalogue length mismatch - {len(output_catalogue)} vs {n_fits}"
            else:
                raise FileNotFoundError(
                    f"Could not find output catalogue at {output_catalogue}"
                )

            if not done:
                for galaxy_id, config in configs.items():
                    new_dir = os.path.join(run_dir, config["out_dir"])
                    new_cat_dir = os.path.dirname(
                        new_dir.replace("posterior", "cats")
                    )

                    if not os.path.exists(new_dir):
                        os.makedirs(new_dir)

                    cat_ids = [f"{galaxy_id}_{id}" for id in config["ids"]]

                    mask = [id in cat_ids for id in output_catalogue["#ID"]]
                    mask = np.array(mask)
                    gal_cat = copy.deepcopy(output_catalogue[mask])

                    assert (
                        len(gal_cat) == len(cat_ids)
                    ), f"Catalogue length mismatch - {len(gal_cat)} vs {len(cat_ids)}"

                    for id in config["ids"]:
                        old_path = os.path.join(
                            current_posterior_folder, f"{galaxy_id}_{id}.h5"
                        )
                        new_path = os.path.join(new_dir, f"{id}.h5")

                        if not os.path.exists(old_path) and not os.path.exists(
                            new_path
                        ):
                            raise FileNotFoundError(
                                f"Could not find .h5 file for {id} at {old_path} or {new_path}"
                            )

                        if os.path.exists(old_path):
                            file = h5.File(old_path, "r")
                            fit_instructions = copy.deepcopy(
                                file.attrs["fit_instructions"]
                            )
                            file.close()
                            # Parse as dict
                            import ast

                            fit_instructions = ast.literal_eval(
                                fit_instructions
                            )
                            # Check if the fit instructions are the same
                            remove = []
                            for instruction in fit_instructions:
                                if "redshift" in instruction:
                                    print(
                                        f'Removing redshift instruction "{instruction}"'
                                    )
                                    remove.append(instruction)
                            for r in remove:
                                fit_instructions.pop(r)

                            # Convert all lists to np arrays
                            for key, value in fit_instructions.items():
                                if type(value) is list:
                                    fit_instructions[key] = np.array(value)
                                if type(value) is dict:
                                    for k, v in value.items():
                                        if type(v) is list:
                                            value[k] = np.array(v)

                            from .utils import find_dict_differences

                            # Find and print differences
                            diff = find_dict_differences(
                                fit_instructions, config["fit_instructions"]
                            )

                            if diff["added"] != {}:
                                print("Added:", diff["added"])
                            if diff["removed"] != {}:
                                print("Removed:", diff["removed"])
                            if diff["modified"] != {}:
                                print("Modified:", diff["modified"])

                            assert (
                                list(fit_instructions.keys())
                                == list(config["fit_instructions"].keys())
                            ), f"Fit instructions do not match for {id}. - {fit_instructions.keys()} vs {config['fit_instructions'].keys()}"
                        if not os.path.exists(os.path.dirname(new_path)):
                            os.makedirs(os.path.dirname(new_path))

                        if not os.path.exists(old_path) and os.path.exists(
                            new_path
                        ):
                            print(
                                f"Output .h5 found in new location for {id}. Skipping."
                            )
                            continue

                        os.rename(old_path, new_path)

                    gal_cat["#ID"] = config["ids"]
                    print("New cat dir:", new_cat_dir)
                    if not os.path.exists(new_cat_dir):
                        os.makedirs(new_cat_dir)

                    gal_cat.write(
                        f"{new_cat_dir}/{galaxy_id}.fits",
                        overwrite=True,
                    )

                # Now rename the bulk catalgoue - just add current time to it as a backup
                os.rename(
                    os.path.join(run_dir, f"cats/{out_subdir_name}.fits"),
                    os.path.join(
                        run_dir,
                        f"cats/{out_subdir_name}_{time.strftime('%Y%m%d_%H%M%S')}.fits",
                    ),
                )

            print("Moving on to loading properties into galaxies.")

            for galaxy, config in zip(self.galaxies, bagpipes_configs):
                run_name = config["meta"]["run_name"]
                bins_to_show = config["meta"]["fit_photometry"]
                # Generate storable meta for Bagpipes table which lists some of config
                storeable_meta = {"binmap_type": config["meta"]["binmap_type"]}
                galaxy.load_bagpipes_results(run_name, meta=storeable_meta)
                if bins_to_show == "bin" or "RESOLVED" in run_name:
                    try:
                        print(f"Loading resolved properties for {run_name}")
                        galaxy.get_resolved_bagpipes_sed(run_name)
                        # Save the resolved SFH
                        galaxy.plot_bagpipes_sfh(
                            run_name, bins_to_show=["RESOLVED"], plot=False
                        )

                        # Save the resolved properties
                        for prop in properties_to_load:
                            galaxy.get_total_resolved_property(
                                run_name,
                                property=prop,
                                log=prop == "stellar_mass",
                            )
                    except Exception as e:
                        print(f"error: {e}")
                        print(traceback.format_exc())
                        print(
                            f"Could not load resolved properties for {run_name}"
                        )
        except Exception as e:
            print(f"Error!: {e}")
            print(traceback.format_exc())
            if alert:
                from .utils import send_email

                ctime = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
                send_email(
                    contents=f"Crash \n {e} \n {traceback.format_exc()}",
                    subject=f"{sys.argv[0]} crash at {ctime} on {computer}",
                )
        # When it has run, need to move and rename posterior files for each galaxy, and
        # split catalogues back out again.

    def save_to_fits(
        self,
        filename="auto",
        overwrite=False,
        bagpipes_row="TOTAL_BIN",
        save=True,
    ):
        """
        # Saves key properties of all galaxies to a fits file.
        Only save properties that exist in all galaxies.
        Save name, redshift, resolved mass, sfr, bagpipes results, cutout size, nbins etc

        """
        from astropy.table import Table

        rows = []

        properties = [
            "galaxy_id",
            "survey",
            "galfind_version",
            "redshift",
            "cutout_size",
        ]
        # For matching Bagpipes run_names
        # Get run_name from first galaxy

        for galaxy in self.galaxies:
            row = {}
            for prop in properties:
                if hasattr(galaxy, prop):
                    row[prop] = getattr(galaxy, prop)
                else:
                    row[prop] = np.nan
            if "bagpipes" not in galaxy.sed_fitting_table.keys():
                print(f"No bagpipes results found for {galaxy.galaxy_id}")
            run_names = galaxy.sed_fitting_table["bagpipes"].keys()

            for run_name in run_names:
                if not getattr(galaxy, "resolved_mass", None) is None:
                    if run_name in galaxy.resolved_mass.keys():
                        row[f"{run_name}_resolved_mass"] = (
                            galaxy.resolved_mass[run_name]
                        )
                    else:
                        row[f"{run_name}_resolved_mass"] = np.nan
                else:
                    print(f"Resolved mass not found for {galaxy.galaxy_id}")

                if not getattr(galaxy, "resolved_sfr_10myr", None) is None:
                    if run_name in galaxy.resolved_sfr_10myr.keys():
                        row[f"{run_name}_resolved_sfr_10myr"] = (
                            galaxy.resolved_sfr_10myr[run_name]
                        )
                    else:
                        row[f"{run_name}_resolved_sfr_10myr"] = np.nan
                if not getattr(galaxy, "resolved_sfr_100myr", None) is None:
                    if run_name in galaxy.resolved_sfr_100myr.keys():
                        row[f"{run_name}_resolved_sfr_100myr"] = (
                            galaxy.resolved_sfr_100myr[run_name]
                        )
                    else:
                        row[f"{run_name}_resolved_sfr_100myr"] = np.nan

                pipes_table = galaxy.sed_fitting_table["bagpipes"][run_name]
                if bagpipes_row in pipes_table["#ID"]:
                    pipes_table = pipes_table[
                        pipes_table["#ID"] == bagpipes_row
                    ]
                    pipes_row = {
                        col: pipes_table[col][0]
                        for col in pipes_table.colnames
                    }
                    # Rename keys with run_name
                    pipes_row = {
                        f"{run_name}_{key}": item
                        for key, item in pipes_row.items()
                    }
                    row.update(pipes_row)
                else:
                    print(
                        f"Could not find {bagpipes_row} in {run_name} table for {galaxy.galaxy_id}"
                    )

            rows.append(row)

        table = Table(rows)
        # Delete all nan columns
        for col in table.colnames:
            vals = list(table[col])
            if type(vals[0]) is float:
                if all(np.isnan(vals)):
                    table.remove_column(col)

        if filename == "auto":
            filename = f"galaxies.fits"
        print(f"Saving to {os.path.abspath(filename)}")

        if save:
            table.write(filename, overwrite=overwrite)

        return table


def run_bagpipes_wrapper(
    galaxy_id,
    resolved_dict,
    cutout_size,
    h5_folder,
    field="JOF_psfmatched",
    version="v11",
    overwrite=False,
    overwrite_internal=False,
    alert=False,
    mpi_serial=False,
    use_mpi=False,
    update_h5=True,
):
    # print('Doing', galaxy_id, resolved_dict['meta']['run_name'], overwrite, overwrite_internal)
    # return

    try:
        if "mock" in str(galaxy_id):
            galaxy = MockResolvedGalaxy.init(
                galaxy_id=galaxy_id,
                mock_survey=field,
                mock_version=version,
                cutout_size=cutout_size,
                h5_folder=h5_folder,
                save_out=update_h5,
            )
        else:
            galaxy = ResolvedGalaxy.init(
                galaxy_id=galaxy_id,
                survey=field,
                version=version,
                cutout_size=cutout_size,
                h5_folder=h5_folder,
                save_out=update_h5,
            )

        # Run bagpipes
        galaxy.run_bagpipes(
            resolved_dict,
            overwrite=overwrite,
            overwrite_internal=overwrite_internal,
            mpi_serial=mpi_serial,
            use_mpi=use_mpi,
            only_run=~update_h5,
        )

        if (
            resolved_dict["meta"]["fit_photometry"] in ["bin", "all"]
            and update_h5
        ):
            print(f"Adding resolved properties to .h5 for {galaxy.galaxy_id}")
            # Save the resolved SED
            galaxy.get_resolved_bagpipes_sed(resolved_dict["meta"]["run_name"])
            # Save the resolved SFH - broken for some reason
            # fig, _ = galaxy.plot_bagpipes_sfh(
            #    resolved_dict["meta"]["run_name"], bins_to_show=["RESOLVED"]
            # )
            plt.close(fig)
            # Save the resolved mass
            galaxy.get_total_resolved_property(
                resolved_dict["meta"]["run_name"], property="stellar_mass"
            )

        return galaxy

    except Exception as e:
        print(f"Error in {galaxy_id}: {e}")
        print(traceback.format_exc())
        if alert:
            from .utils import send_email

            ctime = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
            send_email(
                contents=f'Crash for {galaxy_id} and {resolved_dict["meta"]["run_name"]} \n {e} \n {traceback.format_exc()}',
                subject=f"{sys.argv[0]} crash at {ctime} for {galaxy_id}",
            )

        return None
