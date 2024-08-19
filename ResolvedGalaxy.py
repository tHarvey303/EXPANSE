from astropy.io import fits
from astropy.io.fits import Header
import numpy as np
import matplotlib.pyplot as plt
import h5py as h5
from io import BytesIO
import astropy.units as u
import matplotlib
from astropy.coordinates import SkyCoord
from joblib import Parallel, delayed
from astropy.nddata import Cutout2D
import ast
from astropy.convolution import convolve_fft
import glob
from astropy.nddata import block_reduce
from matplotlib.colors import Normalize, LogNorm
import shutil
from astropy.cosmology import FlatLambdaCDM
from pathlib import Path
from matplotlib.animation import FuncAnimation
import os
from astropy.table import Table, QTable, Column
import typing
import matplotlib as mpl
from scipy.ndimage import zoom
from astropy.utils.exceptions import AstropyWarning
from astropy.visualization import make_lupton_rgb, simple_norm
import warnings
from astropy.utils.masked import Masked
from astropy.wcs import WCS
from astropy.io.misc.hdf5 import write_table_hdf5, read_table_hdf5
# can write astropy to h5
import copy
import cmasher as cm
import sys
from tqdm import tqdm
from inspect import signature
# import Ellipse
from matplotlib.patches import Ellipse
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
# import make_axis_locatable
from mpl_toolkits.axes_grid1 import make_axes_locatable
#import ScalarFormatter
from matplotlib.ticker import ScalarFormatter
warnings.simplefilter('ignore', category=AstropyWarning)
from astropy.cosmology import FlatLambdaCDM
import matplotlib.patheffects as PathEffects
import matplotlib.cm as mcm
#import FontProperties
from matplotlib.font_manager import FontProperties

try:
    from mpi4py import MPI
    rank = MPI.COMM_WORLD.Get_rank()
    size = MPI.COMM_WORLD.Get_size()
    from mpi4py.futures import MPIPoolExecutor

except ImportError:
    rank = 0
    size = 1

cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
# This class is designed to hold all the data for a galaxy, including the cutouts, segmentation maps, and RMS error maps.

'''TODO:
        1. Store aperture photometry, auto photometry from catalogue. 
        2. Pixel binning.
        3. Does ERR map need to be convolved with PSF?
'''

# What computer is this:
# Get path of this file
file_path = os.path.abspath(__file__)
db_dir = ''

if os.path.exists('/.singularity.d/Singularity'):
    computer = 'singularity'
elif 'nvme' in file_path:
    computer = 'morgan'
elif 'Users' in file_path:
    computer = 'mac'


if computer == 'mac':
    bagpipes_dir = '/Users/user/Documents/PhD/bagpipes_dir/'
    print('Running on Mac.')
    bagpipes_filter_dir = bagpipes_dir + 'inputs/filters/'

elif computer == 'morgan':
    bagpipes_dir = '/nvme/scratch/work/tharvey/bagpipes/'
    db_dir = '/nvme/scratch/work/tharvey/dense_basis/pregrids/'
    print('Running on Morgan.')
    bagpipes_filter_dir = bagpipes_dir + 'inputs/filters/'
    

elif computer == 'singularity':
    print('Running in container.')
    bagpipes_filter_dir = 'filters/'


def update_mpl(tex_on=True):
    mpl.rcParams["lines.linewidth"] = 2.
    mpl.rcParams["axes.linewidth"] = 1.5
    mpl.rcParams["axes.labelsize"] = 18.
    mpl.rcParams["xtick.top"] = True
    mpl.rcParams["xtick.labelsize"] = 14
    mpl.rcParams["xtick.direction"] = "in"
    mpl.rcParams["ytick.right"] = True
    mpl.rcParams["ytick.labelsize"] = 14
    mpl.rcParams["ytick.direction"] = "in"
    mpl.rcParams["figure.facecolor"] = '#f7f7f7'
    #mpl.rcParams["figure.edgecolor"] = 'k'
    mpl.rcParams["savefig.bbox"] = 'tight'
    mpl.rcParams["savefig.dpi"] = 300
    mpl.rcParams["figure.dpi"] = 300

    if tex_on:
        #mpl.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
        mpl.rc('text', usetex=True)
        mpl.rcParams["text.usetex"] = True

    else:
        mpl.rcParams["text.usetex"] = False

def scale_fluxes(mag_aper, mag_auto, a, b, theta, kron_radius, psf, zero_point = 28.08, aper_diam = 0.32*u.arcsec, sourcex_factor=6, pix_scale=0.03):
    '''Scale fluxes to total flux using PSF
    Scale mag_aper to mag_auto (ideally measured in LW stack), and then derive PSF correction by placing a elliptical aperture around the PSF.
    '''
    a = sourcex_factor * a
    b = sourcex_factor * b

    area_of_aper = np.pi * (aper_diam/2)**2
    area_of_ellipse = np.pi * a * b
    scale_factor = area_of_aper / area_of_ellipse
    flux_auto = 10**((zero_point - mag_auto) / 2.5)
    flux_aper = 10**((zero_point - mag_aper) / 2.5)

    if scale_factor > 1:
        factor = flux_auto/flux_aper
        factor = np.clip(factor, 1, 1)
    else:
        factor = 1

    flux_aper_corrected = flux_aper * factor
    
    print(f'Corrected aperture magnitude by {factor} mag.')
    # Scale for PSF
    assert type(psf) == np.ndarray, "PSF must be a numpy array"
    assert np.sum(psf) < 1, 'PSF should not be normalised, some flux is outside the footprint.'
    #center = (psf.shape[0] - 1) / 2
    from photutils.centroids import centroid_com
    center = centroid_com(psf)
    # 
    #from photutils import CircularAperture, EllipticalAperture, aperture_photometry
    #circular_aperture = CircularAperture(center, center, r=aper_diam/(2*pixel_scale))
    #circular_aperture_phot = aperture_photometry(psf, circular_aperture)

    if a > psf.shape[0] or b > psf.shape[0]:
        # approximate as a circle
        r = np.sqrt(a*b) * kron_radius
        #encircled_energy = # enclosed_energy in F444W from band
    else:
        elliptical_aperture = EllipticalAperture(center, a=6*a, b=6*b, theta=theta)
        elliptical_aperture_phot = aperture_photometry(psf, elliptical_aperture)
        encircled_energy = elliptical_aperture_phot['aperture_sum'][0]
    
    flux_aper_total = flux_aper_corrected/encircled_energy
    mag_aper_total = -2.5 * np.log10(flux_aper_total) + zero_point
    return mag_aper_total
    
    




def make_EAZY_SED_fit_params_arr(SED_code_arr, templates_arr, lowz_zmax_arr):
    return [{"code": code, "templates": templates, "lowz_zmax": lowz_zmax} \
        for code, templates, lowz_zmaxs in zip(SED_code_arr, templates_arr, lowz_zmax_arr) for lowz_zmax in lowz_zmaxs]

class ResolvedGalaxy:
    internal_bagpipes_cache = {}

    def __init__(self, galaxy_id : int, 
                sky_coord : SkyCoord,
                survey : str, 
                bands, im_paths, im_exts, im_zps, seg_paths,
                rms_err_paths, rms_err_exts, im_pixel_scales, phot_imgs, phot_pix_unit, 
                phot_img_headers, rms_err_imgs, seg_imgs, aperture_dict, galfind_version = 'v11', detection_band = 'F277W+F356W+F444W',
                psf_matched_data = None, psf_matched_rms_err = None, pixedfit_map = None, voronoi_map = None, 
                binned_flux_map = None, binned_flux_err_map = None, photometry_table = None, sed_fitting_table = None,
                rms_background = None, psfs = None, psfs_meta = None, galaxy_region = None, psf_kernels = None,
                cutout_size = 64, photometry_properties = None, photometry_meta_properties = None, total_photometry = None,
                dont_psf_match_bands = [], already_psf_matched = False, auto_photometry = None, flux_map = None, redshift = None,
                unmatched_data = None, unmatched_rms_err = None, unmatched_seg = None, meta_properties = None, det_data = None,
                h5_folder = 'galaxies/', 
                psf_kernel_folder = 'psfs/kernels/', 
                psf_folder = 'psfs/psf_models/',
                psf_type = 'star_stack', overwrite = False):

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

        if photometry_properties is not None:
            self.photometry_property_names = list(photometry_properties.keys())
            for property in photometry_properties:
                setattr(self, property, photometry_properties[property])
                setattr(self, f'{property}_meta', photometry_meta_properties[property])
                #print('Added property', property)
        else:
            self.photometry_property_names = []
        
        self.h5_path = h5_folder + f'{self.survey}_{self.galaxy_id}.h5'

        if os.path.exists(self.h5_path) and overwrite:
            os.remove(self.h5_path)
        
        # Check if dvipng is installed
        if shutil.which('dvipng') is None:
            print('dvipng not found, disabling LaTeX')
            update_mpl(tex_on=False)
        else:
            update_mpl(tex_on=True)
            pass
        # Check sizes of images
        for band in self.bands:
            assert self.phot_imgs[band].shape == (self.cutout_size, self.cutout_size), f"Image shape for {band} is {self.phot_imgs[band].shape}, not {(self.cutout_size, self.cutout_size)}"
            assert self.rms_err_imgs[band].shape == (self.cutout_size, self.cutout_size), f"RMS error image shape for {band} is {self.rms_err_imgs[band].shape}, not {(self.cutout_size, self.cutout_size)}"
            assert self.seg_imgs[band].shape == (self.cutout_size, self.cutout_size), f"Segmentation map shape for {band} is {self.seg_imgs[band].shape}, not {(self.cutout_size, self.cutout_size)}"
        # Bin the pixels
        self.voronoi_map = voronoi_map
        self.pixedfit_map = pixedfit_map
        #print('type', type(pixedfit_map))
        self.psf_matched_data = psf_matched_data
        self.psf_matched_rms_err = psf_matched_rms_err
        self.psfs = psfs
        self.psfs_meta = psfs_meta
        # print(len(psfs_meta))

        self.binned_flux_map = binned_flux_map
        self.binned_flux_err_map = binned_flux_err_map

        self.photometry_table = photometry_table

        self.sed_fitting_table = sed_fitting_table

        self.psf_kernel_folder = psf_kernel_folder
        self.psf_folder = psf_folder
        #self.psf_kernels = {psf_type: {}}
        self.psf_kernels = psf_kernels

        self.gal_region = galaxy_region
        # Assume bands is in wavelength order, and that the largest PSF is in the last band
        self.use_psf_type = psf_type

        if self.psf_matched_data in [None, {}] or self.psf_matched_rms_err in [None, {}] or psf_type not in self.psf_matched_data.keys():

            # If no PSF matched data, then we need to get PSF kernels
            
            if psf_type == 'webbpsf':
                print('Getting WebbPSF')
                self.get_webbpsf()
            elif psf_type == 'star_stack':
                self.get_star_stack_psf()

            #print(f'Assuming {self.bands[-1]} is the band with the largest PSF, and convolving all bands with this PSF kernel.')
    
            print('Convolving images with PSF')
            self.convolve_with_psf(psf_type = psf_type)


        #if self.rms_background is None:
        #    self.estimate_rms_from_background()

        self.dump_to_h5()
         # Save to .h5

    @classmethod
    def init(cls, galaxy_id, survey, version, instruments = ['ACS_WFC', 'NIRCam'], 
                            excl_bands = [], cutout_size = 64, forced_phot_band = ["F277W", "F356W", "F444W"], 
                            aper_diams = [0.32] * u.arcsec, output_flux_unit = u.uJy, h5folder = 'galaxies/',
                            dont_psf_match_bands=[], already_psf_matched = False):
        
        if type(galaxy_id) not in [list, np.ndarray, Column]:
            galaxy_name = f'{survey}_{galaxy_id}'
            if os.path.exists(f'{h5folder}{galaxy_name}.h5'):
                print('Loading from .h5')
                return cls.init_from_h5(galaxy_name, h5_folder = h5folder)
        
        else:
            galaxy_names = [f'{survey}_{gal_id}' for gal_id in galaxy_id]
            found = [os.path.exists(f'{h5folder}{galaxy_name}.h5') for galaxy_name in galaxy_names]
            if all(found):
                print('Loading from .h5')
                return cls.init_multiple_from_h5(galaxy_names, h5_folder = h5folder)

            
        print('Loading from GALFIND')
        return cls.init_from_galfind(galaxy_id, survey, version, instruments = instruments, 
                          excl_bands = excl_bands, cutout_size=cutout_size, forced_phot_band = forced_phot_band, 
                            aper_diams = aper_diams, output_flux_unit = output_flux_unit, h5folder = h5folder,
                            dont_psf_match_bands=dont_psf_match_bands, already_psf_matched = already_psf_matched)

    @classmethod
    def init_from_galfind(cls, galaxy_id, survey, version, instruments = ['ACS_WFC', 'NIRCam'], 
                            excl_bands = [], cutout_size=64, forced_phot_band = ["F277W", "F356W", "F444W"], 
                            aper_diams = [0.32] * u.arcsec, output_flux_unit = u.uJy, h5folder = 'galaxies/',
                            templates_arr = ["fsps_larson"], lowz_zmax_arr = [[4., 6., None]], 
                            dont_psf_match_bands=[], already_psf_matched = False,  psf_type = 'star_stack'):
        # Imports here so only used if needed
        #from galfind import Data
        from galfind import Catalogue, EAZY
        from galfind.Catalogue_Creator import GALFIND_Catalogue_Creator

        if type(galaxy_id) in [list, np.ndarray, Column]:
            load_multiple = True
            crop_by = {'ID': galaxy_id}
        else:
            load_multiple = False
            crop_by = f'ID:{int(galaxy_id)}'
            galaxy_id = [galaxy_id]

        SED_code_arr = [EAZY()]
        SED_fit_params_arr = make_EAZY_SED_fit_params_arr(SED_code_arr, templates_arr, lowz_zmax_arr)
        # Make cat creator
        cat_creator = GALFIND_Catalogue_Creator("loc_depth", aper_diams[0], 10)
        # Load catalogue and populate galaxies
        cat = Catalogue.from_pipeline(survey = survey, version = version, instruments = instruments,
            aper_diams = aper_diams, cat_creator = cat_creator, SED_fit_params_arr = SED_fit_params_arr,
            forced_phot_band = forced_phot_band, excl_bands = excl_bands, loc_depth_min_flux_pc_errs = [10], crop_by=crop_by)
        
        if load_multiple:
            if type(cutout_size) in [list, np.ndarray, Column]:
                assert len(cutout_size) == len(galaxy_id), f"Cutout size must be the same length as galaxy_id, {len(cutout_size)} != {len(galaxy_id)}"

                cutout_size_dict = {galaxy_id: size * cat.data.im_pixel_scales[forced_phot_band[0]].value * u.arcsec for galaxy_id, size in zip(galaxy_id, cutout_size)}

        if not load_multiple or type(cutout_size) not in [list, np.ndarray, Column]:
            cutout_size_dict = cat.data.im_pixel_scales[forced_phot_band[0]]/u.pixel * cutout_size


        band_properties_to_load = ['A_IMAGE', 'B_IMAGE', 'THETA_IMAGE', 'FLUX_RADIUS', 'MAG_AUTO', 'MAGERR_AUTO', 'MAG_BEST', 'MAGERR_BEST', 'MAG_ISO', 'MAGERR_ISO']
        for prop in band_properties_to_load:
            cat.load_band_properties_from_cat(prop, prop)
        properties_to_load = ['ALPHA_J2000', 'DELTA_J2000', 'zbest_16_fsps_larson_zfree', 'zbest_fsps_larson_zfree', 'zbest_84_fsps_larson_zfree', 'chi2_best_fsps_larson_zfree']
        for prop in properties_to_load:
            cat.load_property_from_cat(prop, prop)
        
        # Obtain galaxy object

      
        cat.make_cutouts(galaxy_id, cutout_size = cutout_size_dict)
        objects = []

        for position, gal_id in enumerate(galaxy_id):
            cutout_size_gal = cutout_size_dict[gal_id] if load_multiple else cutout_size_dict
            cutout_size_gal /= cat.data.im_pixel_scales[forced_phot_band[0]].value * u.arcsec

                # Make cutouts - this may not work currently as data.wht_types doesn't appear to be defined.
            
            galaxy = [gal for gal in cat.gals if gal.ID == gal_id]
        
            if len(galaxy) == 0:
                raise Exception(f"Galaxy {gal_id} not found")
            elif len(galaxy) > 1:
                raise Exception(f"Multiple galaxies with ID {gal_id} found")
            else:
                galaxy = galaxy[0]

            #Print all properties of galaxy
            print(galaxy.__dict__.keys())

            cutout_paths = galaxy.cutout_paths
            # Settings things needed for init
            # Get things from Data object
            im_paths = cat.data.im_paths
            im_exts = cat.data.im_exts
            err_paths = cat.data.rms_err_paths
            err_exts = cat.data.rms_err_exts
            seg_paths = cat.data.seg_paths
            im_zps = cat.data.im_zps
            im_pixel_scales = cat.data.im_pixel_scales
            bands = galaxy.phot.instrument.band_names # should be bands just for galaxy!
            #cat.data.instrument.band_names # should be bands just for galaxy!
            # Get things from galaxy object
            galaxy_skycoord = galaxy.sky_coord
            #print('skycoord')
            #print(galaxy.sky_coord)
            bands_mask = galaxy.phot.flux_Jy.mask
            bands = bands[~bands_mask]
            # Get aperture photometry
            flux_aper = galaxy.phot.flux_Jy[~bands_mask]
            flux_err_aper = galaxy.phot.flux_Jy_errs[~bands_mask]
            depths = galaxy.phot.depths[~bands_mask]
            # Get the wavelegnths
            wave = galaxy.phot.wav#[~bands_mask]

            # Get redshift
            SED_fit_params = SED_fit_params_arr[-1]
            print(SED_fit_params["code"].label_from_SED_fit_params(SED_fit_params))
            SED_results = galaxy.phot.SED_results[SED_fit_params["code"].label_from_SED_fit_params(SED_fit_params)]
            redshift = SED_results.z

            print(f'Redshift is {redshift}')
            
            # aperture_dict
            aperture_dict = {str(0.32*u.arcsec): {'flux': flux_aper, 'flux_err': flux_err_aper, 'depths': depths, 'wave': wave}}      
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
                hdu = fits.open(cutout_path)
                assert hdu[1].header['NAXIS1'] == hdu[1].header['NAXIS2'] == cutout_size_gal, f"Cutout size is {hdu[1].header['NAXIS1']}, not {cutout_size_gal}"
                #assert int(hdu[0].header['cutout_size_as'] / cat.data.im_pixel_scales[forced_phot_band[0]]) == cutout_size, f"Cutout size is {hdu[0].header['cutout_size_as'] / cat.data.im_pixel_scales[forced_phot_band[0]]}, not {cutout_size}"
                data = hdu['SCI'].data
                try:
                    rms_data = hdu['RMS_ERR'].data
                except KeyError:
                    weight_data = hdu['WHT'].data
                    rms_data = np.where(weight_data==0, 0, 1/np.sqrt(weight_data))
                unit = u.Unit(hdu['SCI'].header['BUNIT'])
                pix_scale = im_pixel_scales[band]
                # convert to flux_unit
                if unit == u.Unit('MJy/sr'):
                    if output_flux_unit == u.Unit('MJy/sr'):
                        data = data * u.MJy/u.sr
                        rms_data = rms_data * u.MJy/u.sr
                        unit = u.MJy/u.sr
                    else:
                        data = data * unit * pix_scale**2
                        rms_data = rms_data * unit * pix_scale**2

                        if output_flux_unit in [u.Jy, u.mJy, u.uJy, u.nJy]:
                            data = data.to(output_flux_unit)
                            rms_data = rms_data.to(output_flux_unit)
                            unit = output_flux_unit
                        elif output_flux_unit == u.erg/u.s/u.cm**2/u.AA:
                            data = data.to(output_flux_unit, equivalencies=u.spectral_density(wave[band]))
                            rms_data = rms_data.to(output_flux_unit, equivalencies=u.spectral_density(wave[band]))
                            unit = output_flux_unit
                        else:
                            raise Exception("Output flux unit not recognised")
                
                phot_imgs[band] = data
                phot_pix_unit[band] = unit
                rms_err_imgs[band] = rms_data
                seg_imgs[band] = hdu['SEG'].data
                phot_img_headers[band] = str(hdu['SCI'].header)
                hdu.close()

            if already_psf_matched:
                psf_matched_data = {psf_type: {}}
                psf_matched_rms_err = {psf_type: {}}
                for band in bands:
                    psf_matched_data[psf_type][band] = phot_imgs[band]
                    psf_matched_rms_err[psf_type][band] = rms_err_imgs[band]
            else:
                psf_matched_data = None
                psf_matched_rms_err = None        
            
            object = cls(galaxy_id = gal_id, sky_coord = galaxy_skycoord, survey = survey, bands = bands, 
                            im_paths = im_paths, im_exts = im_exts, im_zps = im_zps, seg_paths = seg_paths,
                            detection_band = '+'.join(forced_phot_band), galfind_version = version, 
                            rms_err_paths = err_paths, rms_err_exts = err_exts, im_pixel_scales = im_pixel_scales,
                            phot_imgs = phot_imgs, phot_pix_unit = phot_pix_unit, phot_img_headers = phot_img_headers, 
                            rms_err_imgs = rms_err_imgs, seg_imgs = seg_imgs, aperture_dict = aperture_dict, redshift = redshift,
                            cutout_size = cutout_size_gal, dont_psf_match_bands = dont_psf_match_bands, auto_photometry = auto_photometry,
                            psf_matched_data = psf_matched_data, psf_matched_rms_err = psf_matched_rms_err, meta_properties = meta_properties,
                            already_psf_matched = already_psf_matched, overwrite=True)
            objects.append(object)
       
        if load_multiple:
            return objects
        else:
            return objects[0]

    @classmethod
    def init_multiple_from_h5(cls, h5_names, h5_folder = 'galaxies/'):
        return [cls.init_from_h5(h5_name, h5_folder = h5_folder) for h5_name in h5_names]

    @classmethod
    def init_from_h5(cls, h5_name, h5_folder = 'galaxies/'):
        '''Load a galaxy from an .h5 file'''
        if type(h5_name) == BytesIO:
            hfile = h5.File(h5_name, 'r')
        else:
            h5path = f'{h5_folder}{h5_name}.h5'
            hfile = h5.File(h5path, 'r')
        # Load meta data
        galaxy_id = int(hfile['meta']['galaxy_id'][()].decode('utf-8'))
        survey = hfile['meta']['survey'][()].decode('utf-8')
        sky_coord = hfile['meta']['sky_coord'][()].split(b' ')
        sky_coord = SkyCoord(ra=float(sky_coord[0]), dec=float(sky_coord[1]), unit=(u.deg, u.deg))
        redshift = hfile['meta']['redshift'][()]
        bands = ast.literal_eval(hfile['meta']['bands'][()].decode('utf-8'))
        cutout_size = int(hfile['meta']['cutout_size'][()])
        im_zps = ast.literal_eval(hfile['meta']['zps'][()].decode('utf-8'))
        if 'galfind_version' in hfile['meta'].keys():
            galfind_version = hfile['meta']['galfind_version'][()].decode('utf-8')
        else:
            print('Warning! Assuming galfind version is v11')
            galfind_version = 'v11'
        if 'detection_band' in hfile['meta'].keys():
            detection_band = hfile['meta']['detection_band'][()].decode('utf-8')
        else:
            detection_band = 'F277W+F356W+F444W'
            print('Warning! Assuming detection band is F277W+F356W+F444W')

        im_pixel_scales = ast.literal_eval(hfile['meta']['pixel_scales'][()].decode('utf-8'))
        im_pixel_scales = {band:u.Quantity(scale) for band, scale in im_pixel_scales.items()}
        phot_pix_unit = ast.literal_eval(hfile['meta']['phot_pix_unit'][()].decode('utf-8'))
        phot_pix_unit = {band:u.Unit(unit) for band, unit in phot_pix_unit.items()}
        dont_psf_match_bands = ast.literal_eval(hfile['meta']['dont_psf_match_bands'][()].decode('utf-8'))
        if hfile['meta'].get('already_psf_matched') is not None:
            already_psf_matched = ast.literal_eval(hfile['meta']['already_psf_matched'][()].decode('utf-8'))
        else:
            print('Warning! Assuming already PSF matched')
            already_psf_matched = True
        if hfile['meta'].get('meta_properties') is not None:
            meta_properties = {}
            for prop in hfile['meta']['meta_properties'].keys():
                meta_properties[prop] = ast.literal_eval(hfile['meta']['meta_properties'][prop][()].decode('utf-8'))
        else:
            meta_properties = None
        if 'auto_photometry' in hfile.keys():
            auto_photometry = ast.literal_eval(hfile['auto_photometry']['auto_photometry'][()].decode('utf-8').replace('<Quantity', '').replace('>', ''))
        else:
            auto_photometry = None

        total_photometry = None
        if 'total_photometry' in hfile.keys():
            if hfile['total_photometry'].get('total_photometry') is not None:
                total_photometry = ast.literal_eval(hfile['total_photometry']['total_photometry'][()].decode('utf-8').replace('<Quantity', '').replace('>', ''))
        



        # Load paths and exts
        im_paths = ast.literal_eval(hfile['paths']['im_paths'][()].decode('utf-8'))
        im_exts = ast.literal_eval(hfile['paths']['im_exts'][()].decode('utf-8'))
        seg_paths = ast.literal_eval(hfile['paths']['seg_paths'][()].decode('utf-8'))
        rms_err_paths = ast.literal_eval(hfile['paths']['rms_err_paths'][()].decode('utf-8'))
        rms_err_exts = ast.literal_eval(hfile['paths']['rms_err_exts'][()].decode('utf-8'))
        # Load aperture photometry
        #aperture_dict = ast.literal_eval(hfile['aperture_photometry']['aperture_dict'][()].decode('utf-8'))
        aperture_dict = {}
        for aper in hfile['aperture_photometry'].keys():
            aperture_dict[aper] = {}
            for key in hfile['aperture_photometry'][aper].keys():
                aperture_dict[aper][key] = hfile['aperture_photometry'][aper][key][()]
        # Load raw data
        phot_imgs = {}
        rms_err_imgs = {}
        seg_imgs = {}
        phot_img_headers = {}
        unmatched_data = {}
        unmatched_rms_err = {}
        unmatched_seg = {}

        for band in bands:
            phot_imgs[band] = hfile['raw_data'][f'phot_{band}'][()]
            rms_err_imgs[band] = hfile['raw_data'][f'rms_err_{band}'][()]
            seg_imgs[band] = hfile['raw_data'][f'seg_{band}'][()]
            header = hfile['headers'][band][()].decode('utf-8')
            phot_img_headers[band] = header
            if hfile.get('unmatched_data') is not None:
                if len(hfile['unmatched_data'].keys()) > 0:
                    unmatched_data[band] = hfile['unmatched_data'][f'phot_{band}'][()]
                    unmatched_rms_err[band] = hfile['unmatched_data'][f'rms_err_{band}'][()]
                    unmatched_seg[band] = hfile['unmatched_data'][f'seg_{band}'][()]
            
        if len(unmatched_data) == 0:
            unmatched_data = None
            unmatched_rms_err = None
            unmatched_seg = None

        if hfile.get('psf_matched_data') is not None:
            psf_matched_data = {}
            for psf_type in hfile['psf_matched_data'].keys():
                psf_matched_data[psf_type] = {}
                for band in bands:
                    psf_matched_data[psf_type][band] = hfile['psf_matched_data'][psf_type][band][()]
        else:
            psf_matched_data = None

        if hfile.get('psf_matched_rms_err') is not None:
            psf_matched_rms_err = {}
            for psf_type in hfile['psf_matched_rms_err'].keys():
                psf_matched_rms_err[psf_type] = {}
                for band in bands:
                    psf_matched_rms_err[psf_type][band] = hfile['psf_matched_rms_err'][psf_type][band][()]
        else:
            psf_matched_rms_err = None
        pixedfit_map = None
        voronoi_map = None
        if hfile.get('bin_maps') is not None:
            if hfile['bin_maps'].get('pixedfit') is not None:
                pixedfit_map = hfile['bin_maps']['pixedfit'][()]
            if hfile['bin_maps'].get('voronoi') is not None:
                voronoi_map = hfile['bin_maps']['voronoi'][()]

        binned_flux_map = None
        binned_flux_err_map = None
        if hfile.get('bin_fluxes') is not None:
            if hfile['bin_fluxes'].get('pixedfit') is not None:
                binned_flux_map = hfile['bin_fluxes']['pixedfit'][()] * u.erg/u.s/u.cm**2/u.AA
        
        if hfile.get('bin_flux_err') is not None:
            if hfile['bin_flux_err'].get('pixedfit') is not None:
                binned_flux_err_map = hfile['bin_flux_err']['pixedfit'][()] * u.erg/u.s/u.cm**2/u.AA
        
        possible_phot_keys = []
        possible_sed_keys = []
        photometry_table = {}
        if hfile.get('binned_photometry_table') is not None:
            for psf_type in hfile['binned_photometry_table'].keys():
                photometry_table[psf_type] = {}
                for binmap_type in hfile['binned_photometry_table'][psf_type].keys():
                    if not '__' in binmap_type:
                        photometry_table[psf_type][binmap_type] = None
                        possible_phot_keys.append(f'binned_photometry_table/{psf_type}/{binmap_type}')
        
        sed_fitting_table = {}
        
        if hfile.get('sed_fitting_table') is not None:
            for tool in hfile['sed_fitting_table'].keys():
                sed_fitting_table[tool] = {}
                for run in hfile['sed_fitting_table'][tool].keys():
                    if not '__' in run:
                        sed_fitting_table[tool][run] = None
                        possible_sed_keys.append(f'sed_fitting_table/{tool}/{run}')
        
        rms_background = None
        if hfile.get('meta/rms_background') is not None:
            rms_background = ast.literal_eval(hfile['meta/rms_background'][()].decode('utf-8'))

        # Get PSFs
        psfs = {}
        psfs_meta = {}
        if hfile.get('psfs') is not None:
            for psf_type in hfile['psfs'].keys():
                psfs[psf_type] = {}
                for band in bands:
                    if hfile['psfs'][psf_type].get(band) is not None:
                        psfs[psf_type][band] = hfile['psfs'][psf_type][band][()]

        if hfile.get('psfs_meta') is not None:
            for psf_type in hfile['psfs_meta'].keys():
                psfs_meta[psf_type] = {}
                for band in bands:
                    if hfile['psfs_meta'][psf_type].get(band) is not None:
                        psfs_meta[psf_type][band] = hfile['psfs_meta'][psf_type][band][()].decode('utf-8')

        galaxy_region = {}
        if hfile.get('galaxy_region') is not None:
            for binmap_type in hfile['galaxy_region'].keys():
                galaxy_region[binmap_type] = hfile['galaxy_region'][binmap_type][()]

        flux_map = {}
        if hfile.get('flux_map') is not None:
            for binmap_type in hfile['flux_map'].keys():
                flux_map[binmap_type] = hfile['flux_map'][binmap_type][()]
        
        
        if hfile.get('det_data') is not None:
            det_data = {}
            det_data['phot'] = hfile['det_data'][f'phot'][()]
            det_data['rms_err'] = hfile['det_data'][f'rms_err'][()]
            det_data['seg'] = hfile['det_data'][f'seg'][()]
        else:
            det_data = None
        
        # Read in PSF 
        psf_kernels = {}
        if hfile.get('psf_kernels') is not None:
            for psf_type in hfile['psf_kernels'].keys():
                psf_kernels[psf_type] = {}
                for band in bands:
                    if hfile['psf_kernels'][psf_type].get(band) is not None:
                        psf_kernels[psf_type][band] = hfile['psf_kernels'][psf_type][band][()]

        #hfile.close()
        # Read in photometry table(s)
        if len(possible_phot_keys) > 0:
            for key in possible_phot_keys:
                table = read_table_hdf5(hfile, key)
                psf_type, binmap_type = key.split('/')[1:]
                photometry_table[psf_type][binmap_type] = table

        if len(possible_sed_keys) > 0:
            for key in possible_sed_keys:
                table = read_table_hdf5(hfile, key)
                tool, run = key.split('/')[1:]
                sed_fitting_table[tool][run] = table

        photometry_properties = {}
        photometry_meta_properties = {}
        if hfile.get('photometry_properties') is not None:
            for prop in hfile['photometry_properties'].keys():
                photometry_properties[prop] = hfile['photometry_properties'][prop][()]
                meta = hfile['photometry_properties'][prop].attrs
                #print('meta', meta)
                photometry_meta_properties[prop] = {}
                for key in meta:
                    #print(meta[key])
                    #print(key)
                    #print(meta[key])
                    photometry_meta_properties[prop][key] = meta[key]
                if 'unit' in photometry_meta_properties[prop].keys():
                    unit = u.Unit(photometry_meta_properties[prop]['unit'])
                else:
                    unit = u.dimensionless_unscaled

                #print('unit', unit)

                photometry_properties[prop] = photometry_properties[prop] * unit
                #print(f'Loaded {prop} with shape {photometry_properties[prop].shape}')

        hfile.close()

        return cls(galaxy_id = galaxy_id, sky_coord = sky_coord, survey = survey, bands = bands, im_paths = im_paths,
                    im_zps = im_zps, im_exts = im_exts, seg_paths = seg_paths, rms_err_paths = rms_err_paths,
                    rms_err_exts = rms_err_exts, im_pixel_scales = im_pixel_scales, phot_imgs = phot_imgs,
                    galfind_version = galfind_version, detection_band = detection_band,
                    phot_pix_unit = phot_pix_unit, phot_img_headers = phot_img_headers, rms_err_imgs = rms_err_imgs,
                    seg_imgs = seg_imgs, aperture_dict = aperture_dict, psf_matched_data = psf_matched_data,
                    psf_matched_rms_err = psf_matched_rms_err, pixedfit_map = pixedfit_map, voronoi_map = voronoi_map,
                    binned_flux_map = binned_flux_map, binned_flux_err_map = binned_flux_err_map, photometry_table = photometry_table,
                    sed_fitting_table = sed_fitting_table, rms_background = rms_background, psfs = psfs, psfs_meta = psfs_meta, 
                    auto_photometry = auto_photometry, dont_psf_match_bands = dont_psf_match_bands, galaxy_region = galaxy_region, redshift = redshift,
                    flux_map = flux_map, photometry_properties = photometry_properties, photometry_meta_properties = photometry_meta_properties, already_psf_matched = already_psf_matched,
                    unmatched_data = unmatched_data, unmatched_rms_err = unmatched_rms_err, unmatched_seg = unmatched_seg, meta_properties = meta_properties, det_data = det_data,
                    total_photometry = total_photometry, cutout_size = cutout_size, h5_folder = h5_folder, psf_kernels = psf_kernels)

        '''
        return cls(galaxy_id, sky_coord, survey, bands, im_paths, im_exts, im_zps,
                    seg_paths, rms_err_paths, rms_err_exts, im_pixel_scales, 
                    phot_imgs, phot_pix_unit, phot_img_headers, rms_err_imgs, seg_imgs, 
                    aperture_dict, psf_matched_data, psf_matched_rms_err, pixedfit_map, voronoi_map,
                    binned_flux_map, binned_flux_err_map, photometry_table, sed_fitting_table, rms_background,
                    psfs, psfs_meta, galaxy_region,
                    cutout_size, h5_folder)
        '''

    def get_filter_wavs(self, facilities = {'JWST':['NIRCam'], 'HST':['ACS', 'WFC3_IR']}):
         
        if getattr(self, 'filter_wavs', None) is not None:
            return self.filter_wavs

        from astroquery.svo_fps import SvoFps

        filter_wavs = {}
        filter_instruments = {}
        filter_ranges = {}

        done_band = False
        for facility in facilities:
            if done_band:
                break
            for instrument in facilities[facility]:
                try:
                    svo_table = SvoFps.get_filter_list(facility = facility, instrument = instrument)
                except:
                    continue
                bands_in_table = [i.split('.')[-1] for i in svo_table['filterID']]
                for band in self.bands:
                    if band in bands_in_table:
                        if band in filter_wavs.keys():
                            raise Exception(f'Band {band} found in multiple facilities')
                        else:

                            if instrument == 'ACS':
                                instrument = 'ACS_WFC'
                            filter_instruments[band] = instrument
                            mask = svo_table['filterID'] == f'{facility}/{instrument}.{band}'
                            wav = svo_table[mask]['WavelengthCen']
                            upper = svo_table[mask]["WavelengthCen"] + svo_table[mask]["FWHM"] / 2.
                            lower = svo_table[mask]["WavelengthCen"] - svo_table[mask]["FWHM"] / 2.
                            range = (lower * wav.unit, upper * wav.unit)
                            
                            if len(wav) > 1:
                                raise Exception(f'Multiple profiles found for {band}')
                                
                            filter_wavs[band] = wav[0] * wav.unit

                            filter_ranges[band] = range 
            
        assert len(filter_wavs.keys()) == len(self.bands), f"Not all filters found {filter_wavs.keys()} vs {self.bands}"
        
        self.filter_wavs = filter_wavs
        self.filter_ranges = filter_ranges
        self.filter_instruments = filter_instruments

    def get_star_stack_psf(self, match_band = None):
        '''Get the PSF kernel from a star stack'''

        # Check for kernel

        psf_kernel_folder = f'{self.psf_kernel_folder}/star_stack/{self.survey}/'
        psf_folder = f'{self.psf_folder}/star_stack/{self.survey}/'

        if self.psfs is None:
            self.psfs = {}
        if self.psfs_meta is None:
            self.psfs_meta = {}
        
        self.psfs['star_stack'] = {}
        self.psfs_meta['star_stack'] = {}
        run = False
        for band in self.bands:
            if band in self.dont_psf_match_bands or self.already_psf_matched:
                continue
            path = f'{psf_folder}/{band}_psf.fits'
            if not os.path.exists(path):
                raise Exception(f'No PSF found for {band} in {psf_folder}')
            else:
                psf = fits.getdata(path)
                psf_hdr = str(fits.getheader(path))
                self.psfs['star_stack'][band] = psf
                self.psfs_meta['star_stack'][band] = psf_hdr
            
            self.add_to_h5(psf, 'psfs/star_stack/', band, overwrite=True)
            self.add_to_h5(psf_hdr, 'psfs_meta/star_stack/', band, overwrite=True)
            if match_band is None:
                match_band = self.bands[-1]
            kernel_path = f'{psf_kernel_folder}/kernel_{band}_to_{match_band}.fits'
            if os.path.exists(kernel_path):
                kernel = fits.getdata(kernel_path)
                if self.psf_kernels is None:
                    self.psf_kernels = {'star_stack': {}}
                elif self.psf_kernels.get('star_stack') is None:
                    self.psf_kernels['star_stack'] = {}

                self.psf_kernels['star_stack'][band] = kernel
                self.add_to_h5(kernel, 'psf_kernels/star_stack/', band, overwrite=True)
            else:
                run = True
        if run:
            self.convert_psfs_to_kernels(psf_type = 'star_stack')
                           
    def estimate_rms_from_background(self, cutout_size = 250, object_distance = 20, overwrite=True, plot=False):
        '''Estimate the RMS error from the background'''
        import cv2

        if self.rms_background is None or overwrite:
            self.rms_background = {}
            if plot:
                update_mpl(tex_on=False)
                max_in_row = 4
                fig, axs = plt.subplots(nrows = len(self.bands)//max_in_row + 1, ncols = max_in_row, figsize = (20, 20))
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
                data = hdu[self.im_exts[band]].section[int(y_cent - cutout_size/2):int(y_cent + cutout_size/2), int(x_cent - cutout_size/2):int(x_cent + cutout_size/2)]
                # Open seg
                seg_path = self.seg_paths[band]
                seg_hdu = fits.open(seg_path)
                seg_data = seg_hdu[0].data[int(y_cent - cutout_size/2):int(y_cent + cutout_size/2), int(x_cent - cutout_size/2):int(x_cent + cutout_size/2)]

                # Dilate the segmentation map to be more than 20 pixels from 
                seg_data[seg_data != 0] = 1
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (object_distance, object_distance))
                seg_data = seg_data.astype(np.uint8)
                seg_mask = cv2.dilate(seg_data, kernel, iterations = 1)
                seg_mask = seg_mask.astype(bool)
                # Get RMS of background
                rms = np.sqrt(np.nanmean(data[~seg_mask]**2))
                if plot:
                    # Plot histogram of background
                    # Don't use latex
                    ax = axs[pos]
                    ax.hist(data[~seg_mask].flatten(), bins = 30, histtype = 'step', color = 'k')
                    ax.axvline(rms, color = 'r', linestyle = '--')
                    #ax.set_xlabel('Background')
                    ax.set_title(f'{band} bckg RMS = {rms:.4f} MJy/sr')
                    
                    err_data = hdu[self.rms_err_exts[band]].section[int(y_cent - cutout_size/2):int(y_cent + cutout_size/2), int(x_cent - cutout_size/2):int(x_cent + cutout_size/2)]
                    ax.hist(err_data[~seg_mask].flatten(), bins = 30, histtype = 'step', color = 'b')
                
                self.rms_background[band] = rms

            # In MJy/sr - need to convert?
           
            self.add_to_h5(str(self.rms_background), 'meta', 'rms_background', overwrite=True)
            
        #if plot:
        #    return fig

    def get_number_of_bins(self, binmap_type = 'pixedfit'):
        if binmap_type == 'pixedfit':
            region = self.pixedfit_map
        if region is not None:
                return len(np.unique(region)) - 1
        else:
            print('No galaxy region found')
            return None
        

    def dump_to_h5(self, h5folder='galaxies/', mode='append'):

        '''Dump the galaxy data to an .h5 file'''
        # for strings

        if not os.path.exists(h5folder):
            print('Making directory', h5folder)
            os.makedirs(h5folder)

        str_dt = h5.string_dtype(encoding='utf-8')
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
            append = '_temp'
        else:
            append = ''
        new_h5_path = self.h5_path.replace('.h5', f'{append}.h5')
        hfile = h5.File(new_h5_path, 'w')
        #print('append is', append)

        groups = ['meta', 'paths', 'raw_data', 'aperture_photometry', 'auto_photometry', 'headers', 'bin_maps', 'bin_fluxes', 'bin_flux_err']
        for group in groups:
            hfile.create_group(group) 

        hfile['meta'].create_dataset('galaxy_id', data=str(self.galaxy_id), dtype=str_dt)
        hfile['meta'].create_dataset('survey', data=self.survey, dtype=str_dt)
        hfile['meta'].create_dataset('redshift', data=self.redshift)
        hfile['meta'].create_dataset('sky_coord', data=self.sky_coord.to_string(style='decimal', precision=8), dtype=str_dt)
        hfile['meta'].create_dataset('bands', data=str(list(self.bands)), dtype=str_dt)
        hfile['meta'].create_dataset('cutout_size', data=self.cutout_size)
        hfile['meta'].create_dataset('zps', data=str(self.im_zps), dtype=str_dt)
        hfile['meta'].create_dataset('pixel_scales', data=str({band:str(scale)
            for band, scale in self.im_pixel_scales.items()}), dtype=str_dt)
        hfile['meta'].create_dataset('phot_pix_unit', data=str({band:str(pix_unit) 
            for band, pix_unit in self.phot_pix_unit.items()}), dtype=str_dt)
        hfile['meta'].create_dataset('dont_psf_match_bands', data=str(self.dont_psf_match_bands), dtype=str_dt)
        hfile['meta'].create_dataset('already_psf_matched', data=str(self.already_psf_matched), dtype=str_dt)
        hfile['meta'].create_dataset('detection_band', data=self.detection_band, dtype=str_dt)
        hfile['meta'].create_dataset('galfind_version', data=self.galfind_version, dtype=str_dt)
        if self.meta_properties is not None:
            hfile['meta'].create_group('meta_properties')
            for prop in self.meta_properties.keys():
                hfile['meta']['meta_properties'].create_dataset(prop, data=str(self.meta_properties[prop]), dtype=str_dt)
    
        if self.auto_photometry is not None:
            hfile['auto_photometry'].create_dataset('auto_photometry', data=str(self.auto_photometry), dtype=str_dt)
        if self.total_photometry is not None:
            hfile.create_group('total_photometry')
            hfile['total_photometry'].create_dataset('total_photometry', data=str(self.total_photometry), dtype=str_dt)
        # Save paths and exts
        keys_to_check = ['im_paths', 'seg_paths', 'rms_err_paths', 'im_exts', 'rms_err_exts']

        hfile['paths'].create_dataset('im_paths', data=str(self.im_paths), dtype=str_dt)
        hfile['paths'].create_dataset('seg_paths', data=str(self.seg_paths), dtype=str_dt)
        hfile['paths'].create_dataset('rms_err_paths', data=str(self.rms_err_paths), dtype=str_dt)
        hfile['paths'].create_dataset('im_exts', data=str(self.im_exts), dtype=str_dt)
        hfile['paths'].create_dataset('rms_err_exts', data=str(self.rms_err_exts), dtype=str_dt)

        for aper in self.aperture_dict.keys():
            hfile['aperture_photometry'].create_group(aper)
            for key in self.aperture_dict[aper].keys():
                data = self.aperture_dict[aper][key]    
                hfile['aperture_photometry'][aper].create_dataset(f'{key}', data=data)
        
        # Save raw data
        for band in self.bands:
            hfile['raw_data'].create_dataset(f'phot_{band}', data=self.phot_imgs[band])
            hfile['raw_data'].create_dataset(f'rms_err_{band}', data=self.rms_err_imgs[band])
            hfile['raw_data'].create_dataset(f'seg_{band}', data=self.seg_imgs[band])

        if self.unmatched_data is not None:
            hfile.create_group('unmatched_data')
            for band in self.bands:
                hfile['unmatched_data'].create_dataset(f'phot_{band}', data=self.unmatched_data[band])
                hfile['unmatched_data'].create_dataset(f'rms_err_{band}', data=self.unmatched_rms_err[band])
                hfile['unmatched_data'].create_dataset(f'seg_{band}', data=self.unmatched_seg[band])
        
                  
        if self.det_data is not None:
            hfile.create_group('det_data')
            hfile['det_data'].create_dataset('phot', data=self.det_data['phot'])
            hfile['det_data'].create_dataset('rms_err', data=self.det_data['rms_err'])
            hfile['det_data'].create_dataset('seg', data=self.det_data['seg'])

        # Save headers
        for band in self.bands:
            hfile['headers'].create_dataset(f'{band}', data=str(self.phot_img_headers[band]), dtype=str_dt)

        if self.psf_matched_data is not None:
            hfile.create_group('psf_matched_data')
            for psf_type in self.psf_matched_data.keys():
                hfile['psf_matched_data'].create_group(psf_type)
                for band in self.bands:
                    #print(band)
                    #print(self.psf_matched_data[psf_type])
                    hfile['psf_matched_data'][psf_type].create_dataset(band, data=self.psf_matched_data[psf_type][band])

        if self.psf_matched_rms_err is not None:
            hfile.create_group('psf_matched_rms_err')
            for psf_type in self.psf_matched_rms_err.keys():
                hfile['psf_matched_rms_err'].create_group(psf_type)
                for band in self.bands:
                    hfile['psf_matched_rms_err'][psf_type].create_dataset(band, data=self.psf_matched_rms_err[psf_type][band])
        
        # or here
        # Save galaxy region
         
        # Save binned maps
        if self.voronoi_map is not None:
            hfile['bin_maps'].create_dataset('voronoi', data=self.voronoi_map)
        if self.pixedfit_map is not None: 
            hfile['bin_maps'].create_dataset('pixedfit', data=self.pixedfit_map)
        if self.binned_flux_map is not None:
            hfile['bin_fluxes'].create_dataset('pixedfit', data=self.binned_flux_map)
        if self.binned_flux_err_map is not None:
            hfile['bin_flux_err'].create_dataset('pixedfit', data=self.binned_flux_err_map)
        if self.rms_background is not None:
            hfile.create_dataset('meta/rms_background', data=str(self.rms_background))

        # Small memory leak here - 0.2 MB per save    
    
        # Save PSFs
        if self.psfs is not None and self.psfs != {}:
            hfile.create_group('psfs')
            for psf_type in self.psfs.keys():
                hfile['psfs'].create_group(psf_type)
                for band in self.bands:
                    if self.psfs[psf_type].get(band) is not None:
                        hfile['psfs'][psf_type].create_dataset(band, data=self.psfs[psf_type][band])
        
        if self.psfs_meta is not None and self.psfs_meta != {}:
            hfile.create_group('psfs_meta')
            for psf_type in self.psfs_meta.keys():
                
                hfile['psfs_meta'].create_group(psf_type)
                for band in self.bands:
                    if self.psfs_meta[psf_type].get(band) is not None:
                        data = str(self.psfs_meta[psf_type][band])
                        hfile['psfs_meta'][psf_type].create_dataset(band, data=data, dtype=str_dt)
        
        
        # Add psf_Kernels
        if self.psf_kernels is not None and self.psf_kernels != {}:
            hfile.create_group('psf_kernels')
            for psf_type in self.psf_kernels.keys():
                hfile['psf_kernels'].create_group(psf_type)
                for band in self.bands:
                    if self.psf_kernels[psf_type].get(band) is not None:
                        hfile['psf_kernels'][psf_type].create_dataset(band, data=self.psf_kernels[psf_type][band])

        # Add galaxy region
        if self.gal_region is not None:
            hfile.create_group('galaxy_region')
            for binmap_type in self.gal_region.keys():
                hfile['galaxy_region'].create_dataset(binmap_type, data=self.gal_region[binmap_type])
        # Add flux_map
        if self.flux_map is not None:
            hfile.create_group('flux_map')
            for binmap_type in self.flux_map.keys():
                hfile['flux_map'].create_dataset(binmap_type, data=self.flux_map[binmap_type])

        for pos, property in enumerate(self.photometry_property_names):
            if pos == 0:
                hfile.create_group('photometry_properties')
            hfile['photometry_properties'].create_dataset(property, data=getattr(self, property))
            # Add meta data
            for key in getattr(self, f'{property}_meta').keys():
                hfile['photometry_properties'][property].attrs[key] = getattr(self, f'{property}_meta')[key]

        # Big memory leak here!!  
        hfile.close()
         
        # Write photometry table(s)
        if self.photometry_table is not None:
            for psf_type in self.photometry_table.keys():
                for binmap_type in self.photometry_table[psf_type].keys():
                    write_table_hdf5(self.photometry_table[psf_type][binmap_type], new_h5_path, f'binned_photometry_table/{psf_type}/{binmap_type}', serialize_meta=True, overwrite=True, append=True)
        # Write sed fitting table(s)
        if self.sed_fitting_table is not None:
            for tool in self.sed_fitting_table.keys():
                for run in self.sed_fitting_table[tool].keys():
                    write_table_hdf5(self.sed_fitting_table[tool][run], new_h5_path, f'sed_fitting_table/{tool}/{run}', serialize_meta=True, overwrite=True, append=True)

        
        # Add anything else from the old file to the new file
        if os.path.exists(self.h5_path) and append != '':
            old_hfile = h5.File(self.h5_path, 'r')
            hfile = h5.File(self.h5_path.replace('.h5', f'{append}.h5'), 'a')
            for key in old_hfile.keys():
                if key not in hfile.keys():
                    print('Copying', key)
                    old_hfile.copy(key, hfile)
        
            old_hfile.close()
            hfile.close()
            os.remove(self.h5_path)
            os.rename(self.h5_path.replace('.h5', f'{append}.h5'), self.h5_path)
        

    def convolve_with_psf(self, psf_type = 'webbpsf'):
        '''Convolve the images with the PSF'''
        if getattr(self, 'psf_matched_data', None) in [None, {}] or getattr(self, 'psf_matched_rms_err', None) in [None, {}] or psf_type not in self.psf_matched_data.keys():
            run = False
            # Try and load from .h5
            h5file = h5.File(self.h5_path, 'a')
            if 'psf_matched_data' in h5file.keys():
                if psf_type in h5file['psf_matched_data'].keys():
                    if len(self.bands) != len(list(h5file['psf_matched_data'][psf_type].keys())):
                        run = True
                    else:
                        self.psf_matched_data = {psf_type:{}}
                        for band in self.bands:
                            self.psf_matched_data[psf_type][band] = h5file['psf_matched_data'][psf_type][band][()]
                else:
                    run = True
            else:
                h5file.create_group('psf_matched_data')
                run = True

            if 'psf_matched_rms_err' in h5file.keys():
                if psf_type in h5file['psf_matched_rms_err'].keys():
                    if len(self.bands) != len(list(h5file['psf_matched_rms_err'][psf_type].keys())):
                        run = True
                    else:
                        self.psf_matched_rms_err = {psf_type:{}}
                        for band in self.bands:
                            self.psf_matched_rms_err[psf_type][band] = h5file['psf_matched_rms_err'][psf_type][band][()]
                else:
                    run = True
            else:
                h5file.create_group('psf_matched_rms_err')
                run = True
            

            if run:
                print('Am running')
                # Do the convolution\
                if getattr(self, "psf_matched_data", None) is None:

                    self.psf_matched_data = {psf_type:{}}
                else:
                    self.psf_matched_data[psf_type] = {}
                if getattr(self, "psf_matched_rms_err", None) is None:
                    self.psf_matched_rms_err = {psf_type:{}}
                else:
                    self.psf_matched_rms_err[psf_type] = {}

                for band in self.bands[:-1]:
                    if band in self.dont_psf_match_bands or self.already_psf_matched:
                        psf_matched_img = self.phot_imgs[band]
                        psf_matched_rms_err = self.rms_err_imgs[band]
                    else:
                        kernel = self.psf_kernels[psf_type][band]
                        #kernel = fits.open(kernel_path)[0].data
                        # Convolve the image with the PSF
                        
                        psf_matched_img = convolve_fft(self.phot_imgs[band], kernel, normalize_kernel=True)
                        psf_matched_rms_err = convolve_fft(self.rms_err_imgs[band], kernel, normalize_kernel=True)

                    try:
                        from piXedfit.piXedfit_images.images_utils import remove_naninfzeroneg_image_2dinterpolation
                        psf_matched_rms_err = remove_naninfzeroneg_image_2dinterpolation(psf_matched_rms_err)
                    except:
                        print('Didnt work')
                        pass

                    # Save to psf_matched_data
                    self.psf_matched_data[psf_type][band] = psf_matched_img
                    self.psf_matched_rms_err[psf_type][band] = psf_matched_rms_err

                    if h5file.get(f'psf_matched_data/{psf_type}/{band}') is not None:
                        del h5file[f'psf_matched_data/{psf_type}/{band}']
                    if h5file.get(f'psf_matched_rms_err/{psf_type}/{band}') is not None:
                        del h5file[f'psf_matched_rms_err/{psf_type}/{band}']

                    h5file.create_dataset(f'psf_matched_data/{psf_type}/{band}', data=psf_matched_img)
                    h5file.create_dataset(f'psf_matched_rms_err/{psf_type}/{band}', data=psf_matched_rms_err)

                    #h5file['psf_matched_data'][psf_type] = self.psf_matched_data
                self.psf_matched_data[psf_type][self.bands[-1]] = self.phot_imgs[self.bands[-1]] # No need to convolve the last band
                self.psf_matched_rms_err[psf_type][self.bands[-1]] = self.rms_err_imgs[self.bands[-1]]

                # Deal with last band
                if h5file.get(f'psf_matched_data/{psf_type}/{self.bands[-1]}') is not None:
                    del h5file[f'psf_matched_data/{psf_type}/{self.bands[-1]}']
                if h5file.get(f'psf_matched_rms_err/{psf_type}/{self.bands[-1]}') is not None:
                    del h5file[f'psf_matched_rms_err/{psf_type}/{self.bands[-1]}']
                h5file.create_dataset(f'psf_matched_data/{psf_type}/{self.bands[-1]}', data=self.phot_imgs[self.bands[-1]].value)
                h5file.create_dataset(f'psf_matched_rms_err/{psf_type}/{self.bands[-1]}', data=self.rms_err_imgs[self.bands[-1]].value)
                h5file.close()
            else:
                print('not running')
 
    def __str__(self):
        str = f"Resolved Galaxy {self.galaxy_id} from {self.survey} survey\n"
        str += f"SkyCoord: {self.sky_coord}\n"
        str += f"Bands: {self.bands}\n"
        str += f"Cutout size: {self.cutout_size}\n"
        str += f"Aperture photometry: {self.aperture_dict}\n"
        if hasattr(self, 'pixedfit_map'):
            str += f"Number of bins: {self.get_number_of_bins()}\n"
            # Make an ascii map of the bins
            test_map = copy.copy(self.pixedfit_map)
            unique_vals = np.unique(test_map)
            test_map = test_map.astype(object)
            for val in unique_vals:
                if val in [0, np.nan]:
                    fill_val = ' '
                else:   
                    fill_val = chr(int(val) + 65)

                test_map[test_map == val] = fill_val
            str += f"Pixel-fit map\n"
            for row in test_map:
                str += f"{''.join(row)}\n"


            

        return str

    def __repr__(self):
        return self.__str__()

    def plot_cutouts(self, bands = None, psf_matched = False, save = False, psf_type = None, save_path = None, show = False, facecolor='white'):
        '''Plot the cutouts for the galaxy'''
        if bands is None:
            bands = self.bands

        nrows = len(self.bands)//6 + 1
        fig, axes = plt.subplots(nrows, 6, figsize=(18, 4*nrows), sharex=True, sharey=True, facecolor=facecolor)
        axes = axes.flatten()

        for i in range(len(self.bands), len(axes)):
            fig.delaxes(axes[i])
        

        for i, band in enumerate(bands):
            if psf_matched:
                if psf_type is None:
                    psf_type = self.use_psf_type
                data = self.psf_matched_data[psf_type][band]
            else:
                if self.unmatched_data is not None:
                    data = self.unmatched_data[band]
                else:
                    data = self.phot_imgs[band]
            norm = simple_norm(data, stretch='log', max_percent=99.9)
            axes[i].imshow(data, origin='lower', interpolation='none', norm=norm)
            axes[i].text(0.9, 0.9, band, color='w', transform=axes[i].transAxes, ha='right', va='top', fontsize=12, fontweight='bold', path_effects=[PathEffects.Stroke(linewidth=2, foreground='black'), PathEffects.Normal()])
            #axes[].imshow(self.rms_err_imgs[band], origin='lower', interpolation='none')
            #axes[].set_title(f'{band} RMS Err')
        plt.tight_layout()
        plt.subplots_adjust(hspace=-0.55, wspace=0)
        if save:
            plt.savefig(save_path)
        if show:
            plt.show()
        else:
            return fig

    def get_webbpsf(self, plot=False, overwrite=False, fov=4, og_fov=10, oversample=4, PATH_SW_ENERGY = 'psfs/Encircled_Energy_SW_ETCv2.txt', PATH_LW_ENERGY = 'psfs/Encircled_Energy_LW_ETCv2.txt'):
        skip = False
        if getattr(self, 'psfs', None) not in [None, {}, []]:
            skip = True
        if 'webbpsf' in self.psfs.keys():
            skip = True

        if not skip or overwrite:
            import webbpsf
            self.psfs['webbpsf'] = {}
            self.psfs_meta['webbpsf'] = {}
            psfs = {}
            psf_headers = {}
            for band in self.bands:
                # Get dimensions from header
                header = fits.open(self.im_paths[band])[self.im_exts[band]].header
                header_0 = fits.open(self.im_paths[band])[0].header
                # Get dimensions
                
                xdim = header['NAXIS1']
                ydim = header['NAXIS2']
               
                
                if 10000 < xdim < 10400 & 4200 < ydim < 4400:
                    print('Dimensions consistent with a single NIRCam pointing')

                x_pos, y_pos = WCS(header).all_world2pix(self.sky_coord.ra.deg, self.sky_coord.dec.deg, 0)
                # Calculate which NIRCam detector the galaxy is on
                if float(band[1:-1]) < 240:
                    wav = 'SW'
                    jitter = 0.022
                    # 17 corresponds with 2" radius (i.e. 4" FOV)
                    energy_table = ascii.read(PATH_SW_ENERGY)
                    row = np.argmin(abs(fov/2. - energy_table['aper_radius']))
                    encircled = energy_table[row][filt]
                    norm_fov = energy_table['aper_radius'][row] * 2
                    print(f'Will normalize PSF within {norm_fov}" FOV to {encircled}')

                else:
                    wav = 'LW'
                    jitter = 0.034
                    energy_table = ascii.read(PATH_LW_ENERGY)
                    row = np.argmin(abs(fov/2. - energy_table['aper_radius']))
                    encircled = energy_table[row][filt]
                    norm_fov = energy_table['aper_radius'][row] * 2
                    print(f'Will normalize PSF within {norm_fov}" FOV to {encircled}')

                if wav == 'LW' and  x_pos < 10244/2:
                    det = 'A5'
                else:
                    det = 'B5'
                
                if wav == 'SW':
                    if x_pos < 10244/2:
                        det = 'A'
                        center_rot = (2190, 2190)
                    else:
                        det = 'B'
                        center_rot = (8000, 2250)

                    # Calculate rotation angle from vertical
                    rot = np.arctan((y_pos - center_rot[1])/(x_pos - center_rot[0]))
                    rot = rot * 180/np.pi
                    if 0 < rot < 90:
                        det += '1'
                    elif 90 < rot < 180:
                        det += '3'
                    elif -90 < rot < 0:
                        det += '2'
                    elif -180 < rot < -90:
                        det += '4'
                    
                    print(f'Galaxy at {self.sky_coord.ra.deg} ({x_pos}), {self.sky_coord.dec.deg} ({y_pos}) is on NIRCam {wav} detector {det}')
                
                                
                print(f'{filt} at {fov}" FOV')


                # If consistent with a single NIRCam pointing
                nircam = webbpsf.NIRCam()
                date = header_0['DATE-OBS']
                nircam.load_wss_opd_by_date(date,plot=False)
                #nircam = webbpsf.setup_sim_to_match_file(self.im_paths[band])
                nircam.options['detector'] = f'NRC{det}'
                # Can set nircam.options['source_offset_theta'] = position_angle if not oriented vertical
               
                nircam.filter = band
                nircam.options['output_mode'] = 'detector sampled'
                nircam.options['parity'] = 'odd'
                nircam.options['jitter_sigma'] = jitter
                print('Calculating PSF')
                fov = self.cutout_size*self.im_pixel_scales[band].to(u.arcsec).value
                print(f'Size: {fov} arcsec')

                nircam.pixel_scale = self.im_pixel_scales[band].to(u.arcsec).value
                
                psf = nircam.calc_psf(fov_arcsec=og_fov, normalize='exit_pupil', oversample=4)
                # Drop the first element from the HDU
                
                psf_data =  psf['DET_SAMP'].data

                clip = int((og_fov - fov) / 2 / self.im_pixel_scales[band].to(u.arcsec).value)
                psf_data = psf_data[clip:-clip, clip:-clip]

        
                w, h = np.shape(psf_data)
                Y, X = np.ogrid[:h, :w]
                r = norm_fov / 2. / nc.pixelscale
                center = [w/2., h/2.]
                dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
                psf_data /= np.sum(psf_data[dist_from_center < r])
                psf_data *= encircled # to get the missing flux accounted for
                print(f'Final stamp normalization: {rotated.sum()}')

                psfs[band] = psf_data
                psf_headers[band] = str(psf[1].header)
                if plot:
                    webbpsf.display_psf(psf)
                    #plt.show()
                psf = fits.PrimaryHDU(psf_data)
                
                dir = f'{self.psf_folder}/webbpsf/{self.survey}_{self.galaxy_id}/'
                os.makedirs(dir, exist_ok=True)
                psf.writeto(f'{dir}/webbpsf_{band}.fits', overwrite=True)

                self.add_to_h5(psf.data, 'psfs/webbpsf/', band, overwrite=True)
                self.add_to_h5(str(psf.header), 'psfs_meta/webbpsf/', band, overwrite=True)
            
                self.psfs['webbpsf'][band] = psf.data
                self.psfs_meta['webbpsf'][band] = str(psf.header) # This seems to be large? Maybe just save essentials
     

        else:
            print('Webbpsf PSFs already calculated')
            print('Saving to run pypher')
            for band in self.bands:
                dir = f'{self.psf_folder}/{self.survey}_{self.galaxy_id}/'
                os.makedirs(dir, exist_ok=True)
                hdu = fits.ImageHDU(self.psfs['webbpsf'][band], header=fits.Header.fromstring(self.psfs_meta['webbpsf'][band], sep='\n'))
                hdu.writeto(f'{dir}/webbpsf_{band}.fits', overwrite=True)

        self.convert_psfs_to_kernels(match_band=self.bands[-1], psf_type='webbpsf')


    def convert_psfs_to_kernels(self, match_band=None, psf_type = 'webbpsf', oversample=3):

        if match_band is None:
            match_band = self.bands[-1]

        target_psf = self.psfs[psf_type][match_band]

        dir = f'{self.psf_kernel_folder}/{psf_type}/{self.survey}'
        dir += f'_{self.galaxy_id}/' if psf_type == 'webbpsf' else '/'

        kernel_dir = f'{self.psf_kernel_folder}/{psf_type}/{self.survey}'
        kernel_dir += f'_{self.galaxy_id}/' if psf_type == 'webbpsf' else '/'

        #target_psf = fits.getdata(f'{dir}/{psf_type}_{match_band}.fits')

        if oversample > 1:
            print(f'Oversampling PSF by {oversample}x...')
            target_psf = zoom(target_psf, oversample)

        print(f'Normalizing PSF to unity...')
        target_psf /= target_psf.sum()
        os.makedirs(dir, exist_ok=True)
        fits.writeto(f'{dir}/{psf_type}_a.fits', target_psf, overwrite=True)

        command = ['addpixscl', f'{dir}/{psf_type}_a.fits', f'{self.im_pixel_scales[match_band].to(u.arcsec).value}']
        os.system(' '.join(command))
        print('Computing kernels for PSF matching to ', match_band)
        for band in self.bands[:-1]:
            if band in self.dont_psf_match_bands:
                continue
            #filt_psf = fits.getdata(f'{dir}/{psf_type}_{match_band}.fits')
            filt_psf = self.psfs[psf_type][band]
            if oversample:
                filt_psf = zoom(filt_psf, oversample)

            filt_psf /= filt_psf.sum()

            fits.writeto(f'{dir}/{psf_type}_b.fits', filt_psf, overwrite=True)

            # Need ! pip install pypher first if not installed
            
            command = ['addpixscl', f'{dir}/{psf_type}_b.fits', f'{self.im_pixel_scales[band].to(u.arcsec).value}']
            os.system(' '.join(command))
            try:
                os.remove(f'{dir}/kernel.fits')
            except:
                pass
            command = ['pypher', f'{dir}/{psf_type}_b.fits', f'{dir}/{psf_type}_a.fits', f'{dir}/kernel.fits', '-r', '3e-3']
            print(' '.join(command))
            os.system(' '.join(command))
            kernel = fits.getdata(f'{dir}/kernel.fits')
            
            os.remove(f'{dir}/{psf_type}_b.fits')

            if oversample > 1:
                kernel = block_reduce(kernel,block_size=oversample, func=np.sum)
                kernel /= kernel.sum()
            os.makedirs(kernel_dir, exist_ok=True)
            fits.writeto(f'{kernel_dir}/kernel_{band}_to_{match_band}.fits', kernel, overwrite=True)
            self.psf_kernels[psf_type][band] = kernel 
            #f'{self.psf_kernel_dir}/{psf_type}/kernel_{band}_to_{match_band}.fits'

            os.remove(f'{dir}/kernel.fits')
            os.remove(f'{dir}/kernel.log')

        os.remove(f'{dir}/{psf_type}_a.fits')


      
    def plot_lupton_rgb(self, red = [], green = [], blue = [], q = 1, stretch = 1, use_psf_matched=False, override_psf_type = None, return_array = True, save = False, save_path = None, show = False,):
        '''Plot the galaxy in Lupton RGB'''

        if hasattr(self, 'use_psf_type') and override_psf_type is None:
            psf_type = self.use_psf_type
        else:
            psf_type = override_psf_type
            

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
        
        plt.imshow(rgb, origin='lower')
        self.plot_kron_ellipse(ax=plt.gca())
        if save:
            plt.savefig(save_path)
        if show:
            plt.show()

    def pixedfit_processing(self, use_galfind_seg = True, seg_combine = ['F277W', 'F356W', 'F444W'], gal_region_use = 'pixedfit',
            dir_images = 'galaxies/', override_psf_type = None, use_all_pixels = False, overwrite = False):
        
        if hasattr(self, 'use_psf_type') and override_psf_type is None:
            psf_type = self.use_psf_type
        else:
            psf_type = override_psf_type
            

        from piXedfit.piXedfit_images import images_processing

        if not os.path.exists(dir_images):
            os.makedirs(dir_images)
        
        instruments = ['hst_acs' if band.lower() in ['f435w', 'f606w', 'f775w', 'f814w', 'f850lp'] else 'jwst_nircam' for band in self.bands]
        filters = [f'{instrument}_{band.lower()}' for instrument, band in zip(instruments,self.bands)]
        
        sci_img = {}
        var_img = {}
        img_unit = {}
        scale_factors = {}
        
        for f, band in zip(filters, self.bands):
           
            data = self.psf_matched_data[psf_type][band]
            err = self.psf_matched_rms_err[psf_type][band]
            var = np.square(err)
            header = Header.fromstring(self.phot_img_headers[band], sep='\n')
            if type(data) == u.Quantity:
                data = data.value
            if type(var) == u.Quantity:
                var = var.value

            hdu = fits.PrimaryHDU(data, header=header)
            path = f'{dir_images}/crop_{band}_sci.fits'
            hdu.writeto(path, overwrite=True)
            sci_img[f] = Path(path).name

            err_path = f'{dir_images}/crop_{band}_var.fits'
            hdu = fits.PrimaryHDU(var, header=header)
            hdu.writeto(err_path, overwrite=True)
            var_img[f] = Path(err_path).name
            unit = self.phot_pix_unit[band]
            unit_str = {u.Jy:'Jy', u.MJy/u.sr:'MJy/sr', u.erg/u.s/u.cm**2/u.AA:"erg/s/cm2/A"}
            
            img_unit[f] = unit_str.get(unit, unit)

            scale_factor = (1 * unit) / (1 * u.Jy)
            scale_factor = scale_factor.decompose()
            if scale_factor.unit == u.dimensionless_unscaled:
                scale_factor = scale_factor.value
                img_unit[f] = 'Jy'
            else:
                scale_factor = 1
            
            scale_factors[f] = scale_factor
            # If Unit is some power of 10 of Jy, calculate scale factor to Jy
            
        
        img_pixsizes = {f: float(self.im_pixel_scales[band].to(u.arcsec).value) for f, band in zip(filters, self.bands)}

        gal_ra = self.sky_coord.ra.deg
        gal_dec = self.sky_coord.dec.deg
        gal_z = -1 # PLACEHOLDER -  NOT USED

        flag_psfmatch = True
        flag_reproject = True
        flag_crop = True

        remove_files = True

        print(filters, sci_img)
        
        img_process = images_processing(filters, sci_img, var_img, gal_ra, gal_dec, 
                        dir_images=dir_images, img_unit=img_unit, img_scale=scale_factors, 
                        img_pixsizes=img_pixsizes, run_image_processing=True, stamp_size=(self.cutout_size, self.cutout_size),
                        flag_psfmatch=flag_psfmatch, flag_reproject=flag_reproject, 
                        flag_crop=flag_crop, kernels=None, gal_z=gal_z, remove_files=remove_files)
        
        seg_type = 'galfind'
        # Get galaxy region from segmentation map
        if not use_galfind_seg:
            print('Making segmentation maps')
            img_process.segmentation_sep()
            self.seg_imgs = img_process.seg_maps
            seg_type = 'pixedfit_sep'
            #self.add_to_h5(img_process.segm_maps, 'seg_maps', 'pixedfit', ext='SEG_MAPS')


        if use_all_pixels:
            segm_maps = [np.ones_like(self.seg_imgs[band]) for band in self.bands]
            img_process.segm_maps = segm_maps
            segm_maps_ids = None
            seg_type = 'all_pixels'
        else:
            segm_maps = []
            for band in self.bands:
                segm = self.seg_imgs[band]
                #change to 0 is background, 1 is galaxy
                # Get value in center
                #print(self.cutout_size//2)
                center = int(self.cutout_size//2)
                center = segm[center, center]

                segm[segm == center] = 1
                segm[segm != 1] = 0
                #print(np.count_nonzero(segm))
                segm_maps.append(segm)

            img_process.segm_maps = segm_maps

            if seg_combine is not None:
                segm_maps_ids = np.argwhere(np.array([band in seg_combine for band in self.bands])).flatten()
            else:
                segm_maps_ids = None

        if self.gal_region is None:
            self.gal_region = {}
        galaxy_region = img_process.galaxy_region(segm_maps_ids=segm_maps_ids)
        
        self.gal_region['pixedfit'] = galaxy_region
        
        if self.det_data is not None:
            det_galaxy_region = copy.copy(self.det_data['seg'])
            det_galaxy_region[det_galaxy_region == self.galaxy_id] = 1
            det_galaxy_region[det_galaxy_region != 1] = 0
            self.gal_region['detection'] = det_galaxy_region
            if gal_region_use == 'detection':
                seg_type = 'detection'


        # Difference between gal_region - which I think is one image for all bands. 
        
        # Calculate maps of multiband fluxes
        flux_maps_fits = f"{dir_images}/{self.survey}_{self.galaxy_id}_fluxmap.fits"
        Gal_EBV = 0 # Placeholder

        img_process.flux_map(self.gal_region[gal_region_use], Gal_EBV=Gal_EBV, name_out_fits=flux_maps_fits)
        if remove_files:
            files = glob.glob('*crop_*')
            for file in files:
                os.remove(file)
            files = glob.glob(dir_images+'/crop_*')
            for file in files:
                os.remove(file)
        
        self.flux_map_path = flux_maps_fits
        self.img_process = img_process

        meta_dict = {'stacked_bands':'+'.join(seg_combine), 'seg_type':seg_type}
        
        self.add_to_h5(galaxy_region, 'galaxy_region', 'pixedfit', meta=meta_dict, overwrite=overwrite)
        
        ## What I have named 'galaxy_region' is actually flux_map_fits
        self.add_to_h5(flux_maps_fits, 'flux_map', 'pixedfit', overwrite=overwrite)

        # Copy flux map to h5 file - TODO
        self.dir_images = dir_images

        return img_process

    def plot_voronoi_map(self):
        if self.voronoi_map is None:
            print('No Voronoi map found')
            return

        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        mappable = ax.imshow(self.voronoi_map, origin='lower', interpolation='none', cmap = 'nipy_spectral_r')   
        fig.colorbar(mappable, ax=ax)
        ax.set_title('Voronoi Map')
        #plt.show()

    def plot_snr_map(self, band = 'All', override_psf_type = None, facecolor='white', show = False):
        if hasattr(self, 'use_psf_type') and override_psf_type is None:
            psf_type = self.use_psf_type
        else:
            psf_type = override_psf_type
            
        
        bands = self.bands if band == 'All' else [band]
        nrows = len(bands)//6 + 1
        fig, axes = plt.subplots(nrows, 6, figsize=(18, 4*nrows), facecolor=facecolor)
        axes = axes.flatten()
        # Remove empty axes
        for i in range(len(bands), len(axes)):
            fig.delaxes(axes[i])
        
        for i, band in enumerate(bands):
            snr_map = self.psf_matched_data[psf_type][band] / self.psf_matched_rms_err[psf_type][band]
            mappable = axes[i].imshow(snr_map, origin='lower', interpolation='none')
            cax = make_axes_locatable(axes[i]).append_axes('right', size='5%', pad=0.05)
            fig.colorbar(mappable, ax=axes[i], cax=cax)
            axes[i].set_title(f'{band} SNR Map')
            # Turn of ticklabels
            axes[i].set_xticklabels([])
            axes[i].set_yticklabels([])
        
        if show:
            plt.show()
        else:
            return fig
        #plt.show()

    def voronoi_binning(self, SNR_reqs=10, ref_band='F277W', plot=True, override_psf_type=None):
        if hasattr(self, 'use_psf_type') and override_psf_type is None:
            psf_type = self.use_psf_type
        else:
            psf_type = override_psf_type
            
        from vorbin.voronoi_2d_binning import voronoi_2d_binning

        # x - x coordinates of pixels
        # y - y coordinates of pixels
        # signal - fluxes of pixels
        # noise - 
        # target SN - 

        x = np.arange(self.cutout_size)
        y = np.arange(self.cutout_size)
        x, y = np.meshgrid(x, y)
        x = x.flatten()
        y = y.flatten()
    
        signal = self.psf_matched_data[psf_type][ref_band].flatten()
        noise = self.psf_matched_rms_err[psf_type][ref_band].flatten()
       
        #sn_func = lambda index, flux, flux_err: print(index) #flux[index] / flux_err[index]
        bin_number, x_gen, y_gen, x_bar, y_bar, sn, nPixels, scale = voronoi_2d_binning(
                        x, y, signal, noise, SNR_reqs,
                        #pixelsize = self.im_pixel_scales[ref_band].to(u.arcsec).value, 
                        plot=plot)# sn_func = sn_func)
        
        # Reshape bin_number to 2D
        bin_number = bin_number.reshape(self.cutout_size, self.cutout_size)
        self.voronoi_map = bin_number
        meta_dict = {'ref_band':ref_band, 'SNR_reqs':SNR_reqs}
        self.add_to_h5(bin_number, 'bin_maps', 'voronoi', setattr_gal='voronoi_map', meta=meta_dict)

    def pixedfit_binning(self, SNR_reqs=7, ref_band='F277W', min_band = 'auto', Dmin_bin=7, redc_chi2_limit=5.0, del_r=2.0, overwrite=False, min_snr_wav = 1216 * u.AA, only_snr_instrument = 'NIRCam'):
        '''
        : SNR_reqs: list of SNR requirements for each band
        : ref_band: reference band for pixel binning
        : Dmin_bin: minimum diameter between pixels in binning (should be ~ FWHM of PSF)
        '''
        from piXedfit.piXedfit_bin import pixel_binning     

        if not hasattr(self, 'img_process'):
            raise ValueError('No image processing done. Run pixedfit_processing() first')

       
        #ref_band_pos = np.argwhere(np.array([band == ref_band for band in self.bands])).flatten()[0]
        no_SNR_requirement = []
        # Should calculate SNR requirements intelligently based on redshift
        if min_band is not None:
            if min_band == 'auto':
                self.get_filter_wavs()
                min_snr_wav_obs = min_snr_wav * (1 + self.redshift)
                delta_wav = 1e10 * u.AA
                for band in self.bands:
                    instrument = self.filter_instruments[band]
                    if only_snr_instrument is not None and instrument != only_snr_instrument:
                        print(f'Skipping {band} as it is not in {only_snr_instrument}')
                        no_SNR_requirement.append(band)    
                    
                    if self.filter_ranges[band][1] < min_snr_wav_obs:
                        no_SNR_requirement.append(band)
                    '''
                    if self.filter_ranges[band][1] > min_snr_wav_obs and self.filter_ranges[band][1] - min_snr_wav_obs < delta_wav:
                        min_band = band
                        delta_wav = self.filter_ranges[band][1] - min_snr_wav_obs 
                    '''
                #print(f'Auto-selected minimum band for SNR: {min_band}')

            
        name_out_fits = f'{self.dir_images}/{self.survey}_{self.galaxy_id}_binned.fits'

        header = fits.open(self.flux_map_path)[0].header
        header_order = [header[f'FIL{pos}'].split('_')[-1].upper() for pos in range(0, len(self.bands))]
        ref_band_pos = header_order.index(ref_band)
        
        SNR_reqs = {band: SNR_reqs for band in self.bands if band not in no_SNR_requirement}
        for band in no_SNR_requirement:
              SNR_reqs[band] = 0

        SNR = [SNR_reqs[band] for band in header_order]

        for pos, band in enumerate(header_order):
            print(band, SNR_reqs[band])
            if pos == ref_band_pos:
                print('Reference band.')
            

        pixel_binning(self.flux_map_path, ref_band=ref_band_pos, Dmin_bin=Dmin_bin, SNR=SNR, redc_chi2_limit=redc_chi2_limit, del_r=del_r,  name_out_fits=name_out_fits)
        
        self.pixedfit_binmap_path = name_out_fits
        self.add_to_h5(name_out_fits, 'bin_maps', 'pixedfit', ext='BIN_MAP', setattr_gal='pixedfit_map', overwrite=overwrite)
        self.add_to_h5(name_out_fits, 'bin_fluxes', 'pixedfit', ext='BIN_FLUX', setattr_gal='binned_flux_map', overwrite=overwrite)
        self.add_to_h5(name_out_fits, 'bin_flux_err', 'pixedfit', ext='BIN_FLUXERR', setattr_gal='binned_flux_err_map', overwrite=overwrite)


    def plot_kron_ellipse(self, ax, center, band='detection', color = 'red', return_params = False):
        if band == 'detection':
            if self.total_photometry is None:
                if return_params:
                    return 0, 0, 0
            #kron = self.total_photometry[self.detection_band][f'KRON_RADIUS_{self.detection_band}']
            a = self.total_photometry[self.detection_band][f'a_{self.detection_band}'] # already scaled by kron 
            b = self.total_photometry[self.detection_band][f'b_{self.detection_band}'] # already scaled by kron
            theta = self.total_photometry[self.detection_band][f'theta_{self.detection_band}']
        else:
            kron_radius = self.auto_photometry[band]['KRON_RADIUS']
            a = self.auto_photometry[band]['A_IMAGE'] * kron_radius
            b = self.auto_photometry[band]['B_IMAGE'] * kron_radius
            theta = self.auto_photometry[band]['THETA_IMAGE']
        
        if return_params:
            return a, b, theta

        #center = np.shape(data)[0]/2
        e = Ellipse((center, center), a, b, angle=theta, edgecolor=color, facecolor='none')
        
        ax.add_artist(e)
        e.set_clip_box(ax.bbox)
        e.set_alpha(0.5)

    def plot_image_stamp(self, band, scale = 'log10', save=False, save_path=None, show=False, facecolor='white', sex_factor=6):
        fig, ax = plt.subplots(1, 1, figsize=(6, 6), facecolor=facecolor)
        if scale == 'log10':
            data = np.log10(self.phot_imgs[band])
        elif scale == 'linear':
            data = self.phot_imgs[band]
        else:
            raise ValueError('Scale must be log10 or linear')
        ax.imshow(data, origin='lower', interpolation='none')
        if band in self.auto_photometry.keys():
            self.plot_kron_ellipse(ax, center=self.cutout_size/2, band=band)
        
        re = 15 # pixels
        d_A = cosmo.angular_diameter_distance(self.redshift)
        pix_scal = u.pixel_scale(0.03*u.arcsec/u.pixel)
        re_as = (re * u.pixel).to(u.arcsec, pix_scal)
        re_kpc = (re_as * d_A).to(u.kpc, u.dimensionless_angles())
        
        # First scalebar
        scalebar = AnchoredSizeBar(ax.transData, 0.5 / self.im_pixel_scales[band].value, \
            "0.5\"", 'lower right', pad = 0.3, color='black', frameon=False, size_vertical=1, fontproperties=FontProperties(size=18))
        ax.add_artist(scalebar)
        # Plot scalebar with physical size
        scalebar = AnchoredSizeBar(ax.transData, re, f"{re_kpc:.1f}", \
            'upper left', pad=0.3, color='black', frameon=False, size_vertical=1, fontproperties=FontProperties(size=18))
        scalebar.set(path_effects=[PathEffects.withStroke(linewidth=3, foreground='white')])
        ax.add_artist(scalebar)

        # Add scalebar

        ax.set_title(f'{band} Image')
        if save:
            plt.savefig(save_path)
        if show:
            plt.show()
        

    def plot_image_stamps(self, show = False):
        nrows = len(self.bands)//6 + 1
        fig, axes = plt.subplots(nrows, 6, figsize=(24, 4*nrows))
        axes = axes.flatten()
        # Remove empty axes
        for i in range(len(self.bands), len(axes)):
            fig.delaxes(axes[i])

        for i, band in enumerate(self.bands):
            axes[i].imshow(np.log10(self.phot_imgs[band]), origin='lower', interpolation='none')
            axes[i].set_title(f'{band} Image')
        plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, hspace=0.1, wspace=0.15)
        #return fig
        if show:
            plt.show()
        else:
            return fig

    
    def plot_gal_region(self, bin_type = 'pixedfit', facecolor='white', show=False):
        if self.gal_region is None:
            raise ValueError('No gal_region region found. Run pixedfit_processing() first')
        else:
            if bin_type not in self.gal_region.keys():
                raise ValueError(f'gal_region not found for {bin_type}. Run pixedfit_processing() first')
        gal_region = self.gal_region[bin_type]
        nrows = len(self.bands)//6 + 1
        fig, axes = plt.subplots(nrows, 6, figsize=(18, 4*nrows), facecolor=facecolor)
        axes = axes.flatten()
        # Remove empty axes
        for i in range(len(self.bands), len(axes)):
            fig.delaxes(axes[i])

        for i, band in enumerate(self.bands):
            #rows, cols = np.where(gal_region==0)
            #gal_region[rows,cols] = float('nan')
            axes[i].imshow(np.log10(self.phot_imgs[band]), origin='lower', interpolation='none')
            axes[i].set_title(f'{band} Image')
            axes[i].imshow(gal_region, origin='lower', interpolation='none', alpha=0.5, cmap='copper')
        plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, hspace=0.1, wspace=0.15)
        if show:
            plt.show()
        else:
            return fig
        #return fig

    def add_detection_data(self, detection_instrument = 'NIRCam', galfind_work_dir = '/raid/scratch/work/austind/GALFIND_WORK', overwrite = False):
        
        if self.det_data is not None and not overwrite:
            print('Detection data already loaded')
            return

        im_path = f'{galfind_work_dir}/Stacked_Images/{self.galfind_version}/{detection_instrument}/{self.survey}/{self.survey}_{self.detection_band}_{self.galfind_version}_stack_new.fits'
        seg_path = f'{galfind_work_dir}/SExtractor/{detection_instrument}/{self.galfind_version}/{self.survey}/{self.survey}_{self.detection_band}_{self.detection_band}_sel_cat_{self.galfind_version}_seg.fits'
        hdu_data = fits.open(im_path)
        hdu_seg = fits.open(seg_path)
        det_data = {}

        #print(im_path)
        #print(seg_path)
        output_flux_unit = self.phot_pix_unit['F444W']
        hd = fits.getheader(self.im_paths['F444W'], ext = 1)
        wcs_test = WCS(hd)
        #print(self.sky_coord)
        pix_scale = self.im_pixel_scales['F444W'].to(u.arcsec)
        for ext, hdu, name, flux_unit in zip(['SCI', 'ERR', 0], [hdu_data, hdu_data, hdu_seg], ['phot', 'rms_err', 'seg'], [output_flux_unit, output_flux_unit, None]):
            data = hdu[ext].data
            header = hdu[ext].header
            unit = u.Unit(header['BUNIT']) if flux_unit is not None else False
            wcs = WCS(header)
            #print(wcs.world_to_pixel(self.sky_coord))
            #print(wcs_test.world_to_pixel(self.sky_coord))
            skycoord = SkyCoord(self.meta_properties['ALPHA_J2000'] * u.deg, self.meta_properties['DELTA_J2000'] * u.deg, frame='icrs')
            #print(skycoord, self.sky_coord)
            cutout = Cutout2D(data, skycoord, (self.cutout_size, self.cutout_size), wcs=wcs)
            data = cutout.data 

            if unit:

                if unit == u.Unit('MJy/sr'):
                    if output_flux_unit == u.Unit('MJy/sr'):
                        data = data * u.MJy/u.sr
                        
                    else:
                        data = data * unit * pix_scale**2

                        if output_flux_unit in [u.Jy, u.mJy, u.uJy, u.nJy]:
                            data = data.to(output_flux_unit)
                            unit = output_flux_unit
                        elif output_flux_unit == u.erg/u.s/u.cm**2/u.AA:
                            data = data.to(output_flux_unit, equivalencies=u.spectral_density(wave[band]))
                            unit = output_flux_unit
                        else:
                            raise Exception("Output flux unit not recognised")
                
            self.add_to_h5(data, 'det_data', f'{name}', overwrite=overwrite)
            det_data[name] = data

        hdu_data.close()
        hdu_seg.close()
        self.det_data = det_data


    def add_original_data(self, instruments = ['ACS_WFC', 'NIRCam'], excl_bands = [], 
                            aper_diams = [0.32] * u.arcsec, templates_arr = ["fsps_larson"], overwrite = False,
                            lowz_zmax_arr = [[4., 6., None]], cat = None, return_cat = False, crop_by = 'ID'):

        if self.unmatched_data is not None and self.unmatched_rms_err is not None and self.unmatched_seg is not None and not overwrite:
            print('Unmatched data already loaded')
            return

        if crop_by == 'ID':
            crop_by = f'ID={int(self.galaxy_id)}'
        
        if cat == None:
            from galfind import Catalogue, EAZY
            from galfind.Catalogue_Creator import GALFIND_Catalogue_Creator
            SED_code_arr = [EAZY()]
            SED_fit_params_arr = make_EAZY_SED_fit_params_arr(SED_code_arr, templates_arr, lowz_zmax_arr)
            # Make cat creator
            cat_creator = GALFIND_Catalogue_Creator("loc_depth", aper_diams[0], 10)
            # Load catalogue and populate galaxies
            cat = Catalogue.from_pipeline(survey = self.survey.replace('_psfmatched', ''), version = self.galfind_version, instruments = instruments,
                aper_diams = aper_diams, cat_creator = cat_creator, SED_fit_params_arr = SED_fit_params_arr,
                forced_phot_band = self.detection_band.split('+'), excl_bands = excl_bands, loc_depth_min_flux_pc_errs = [10], crop_by=crop_by)
            # Make cutouts - this may not work currently as data.wht_types doesn't appear to be defined.
        
        cat.make_cutouts([self.galaxy_id], cutout_size = self.cutout_size * self.im_pixel_scales['F444W'].to(u.arcsec))

        # Obtain galaxy object
        galaxy = [gal for gal in cat.gals if gal.ID == self.galaxy_id]
        if len(galaxy) == 0:
            raise ValueError(f'Galaxy with ID {self.galaxy_id} not found')
        elif len(galaxy) > 1:
            for gal in galaxy:
                print(gal)
        
        assert len(galaxy) == 1, f'{len(galaxy)} galaxies found with ID {self.galaxy_id}'
        galaxy = galaxy[0]
       
        cutout_paths = galaxy.cutout_paths
        bands = galaxy.phot.instrument.band_names # should be bands just for galaxy!
        galaxy_skycoord = galaxy.sky_coord
        # Some checks 
        assert [np.round(galaxy_skycoord.ra.degree, 4), np.round(galaxy_skycoord.dec.degree, 4)] == [np.round(self.sky_coord.ra.degree, 4), np.round(self.sky_coord.dec.degree, 4)], f'Galaxy skycoord {galaxy_skycoord} does not match input skycoord {self.sky_coord}'
        
        bands_mask = galaxy.phot.flux_Jy.mask
        bands = bands[~bands_mask]  
        assert len(bands) == len(self.bands)

        phot_imgs = {}
        phot_pix_unit = {}
        rms_err_imgs = {}
        seg_imgs = {}
        phot_img_headers = {}

        hfile = h5.File(self.h5_path, 'a')
        if 'unmatched_data' not in hfile.keys():
            hfile.create_group('unmatched_data')
        output_flux_unit = self.phot_pix_unit['F444W']

        for band in bands:
            cutout_path = cutout_paths[band]
            hdu = fits.open(cutout_path)
            assert hdu['SCI'].header['NAXIS1'] == hdu['SCI'].header['NAXIS2'] == self.cutout_size # Check cutout size
            data = hdu['SCI'].data
            try:
                rms_data = hdu['RMS_ERR'].data
            except KeyError:
                weight_data = hdu['WHT'].data
                rms_data = np.where(weight_data==0, 0, 1/np.sqrt(weight_data))
            unit = u.Unit(hdu['SCI'].header['BUNIT'])
            pix_scale = self.im_pixel_scales[band]
            # convert to flux_unit
            if unit == u.Unit('MJy/sr'):
                if output_flux_unit == u.Unit('MJy/sr'):
                    data = data * u.MJy/u.sr
                    rms_data = rms_data * u.MJy/u.sr
                    unit = u.MJy/u.sr
                else:
                    data = data * unit * pix_scale**2
                    rms_data = rms_data * unit * pix_scale**2

                    if output_flux_unit in [u.Jy, u.mJy, u.uJy, u.nJy]:
                        data = data.to(output_flux_unit)
                        rms_data = rms_data.to(output_flux_unit)
                        unit = output_flux_unit
                    elif output_flux_unit == u.erg/u.s/u.cm**2/u.AA:
                        data = data.to(output_flux_unit, equivalencies=u.spectral_density(wave[band]))
                        rms_data = rms_data.to(output_flux_unit, equivalencies=u.spectral_density(wave[band]))
                        unit = output_flux_unit
                    else:
                        raise Exception("Output flux unit not recognised")
            
            phot_imgs[band] = data
            phot_pix_unit[band] = unit
            rms_err_imgs[band] = rms_data
            seg_imgs[band] = hdu['SEG'].data
            #phot_img_headers[band] = str(hdu['SCI'].header)
            hdu.close()
            
            if f'phot_{band}' in hfile['unmatched_data'].keys():
                hfile['unmatched_data'][f'phot_{band}'][()] = data
                hfile['unmatched_data'][f'rms_err_{band}'][()] = rms_data
                hfile['unmatched_data'][f'seg_{band}'][()] = seg_imgs[band]
            else:
                hfile['unmatched_data'].create_dataset(f'phot_{band}', data=data)
                hfile['unmatched_data'].create_dataset(f'rms_err_{band}', data=rms_data)
                hfile['unmatched_data'].create_dataset(f'seg_{band}', data=seg_imgs[band])
            
        hfile.close()
        self.unmatched_data = phot_imgs
        self.unmatched_rms_err = rms_err_imgs
        self.unmatched_seg = seg_imgs

        if return_cat:
            return cat
        
            
        '''
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
            err_data = hdu[self.err_exts[band]].data
            err_header = hdu[self.err_exts[band]].header
            hdu.close()
            wcs = WCS(err_header)
            cutout = Cutout2D(fits.open(path)[self.err_exts[band]].data, self.sky_coord, size=self.cutout_size*u.pixel, wcs=wcs)
            cutout_err_data = cutout.data
            cutout_err_header = cutout.wcs.to_header()
        '''



            

    def add_to_h5(self, original_data, group, name, ext=0, setattr_gal=None, overwrite=False, meta=None, setattr_gal_meta=None):
        
        if type(original_data) in [u.Quantity, u.Magnitude]:
            data = original_data.value
        else:
            data = original_data


        if type(data) in [dict]:
            data = str(data)

        if type(data) == str:
            if data.endswith('.fits'):
                data = fits.open(data)[ext].data
                original_data = data


        hfile = h5.File(self.h5_path, 'a')
        if group not in hfile.keys():
            hfile.create_group(group)
        if name in hfile[group].keys():
            if overwrite:
                del hfile[group][name]
            else:
                print(f'{name} already exists in {group} group and overwrite is set to False')
                return

        hfile[group].create_dataset(name, data=data)
        if meta is not None:
            for key in meta.keys():
                print('Setting meta', key, str(meta[key]))
                hfile[group][name].attrs[key] = str(meta[key])
            if setattr_gal_meta is not None:
                setattr(self, setattr_gal_meta, meta)


        print('added to ', hfile, group, name)
        hfile.close()
        if setattr_gal is not None:
            print(f'Setting {setattr_gal} attribute.')
            setattr(self, setattr_gal, original_data)

    def plot_err_stamps(self):
        fig, axes = plt.subplots(1, len(self.bands), figsize=(4*len(self.bands), 4))
        for i, band in enumerate(self.bands):
            axes[i].imshow(np.log10(self.rms_err_imgs[band]), origin='lower', interpolation='none')
            axes[i].set_title(f'{band} Error')

    def plot_seg_stamps(self, show_pixedfit=False):
        # Split ax over rows - max 6 per row
        nrows = len(self.bands)//6 + 1
        fig, axes = plt.subplots(nrows, 6, figsize=(24, 4*nrows))
        axes = axes.flatten()
        # Remove empty axes
        for i in range(len(self.bands), len(axes)):
            fig.delaxes(axes[i])

        for i, band in enumerate(self.bands):
            if show_pixedfit:
                mappable = axes[i].imshow(self.img_process.segm_maps[i], origin='lower', interpolation='none')

            mappable = axes[i].imshow(self.seg_imgs[band], origin='lower', interpolation='none')
            fig.colorbar(mappable, ax=axes[i])
            axes[i].set_title(f'{band} Segmentation')
        return fig

    def pixedfit_plot_map_fluxes(self):
        from piXedfit.piXedfit_images import plot_maps_fluxes
        if not hasattr(self, 'flux_map_path'):
            raise ValueError('No flux map found. Run pixedfit_processing() first')

        plot_maps_fluxes(self.flux_map_path, ncols=8, savefig=False)

    def pixedfit_plot_radial_SNR(self):
        from piXedfit.piXedfit_images import plot_SNR_radial_profile
        if not hasattr(self, 'flux_map_path'):
            raise ValueError('No flux map found. Run pixedfit_processing() first')
        plot_SNR_radial_profile(self.flux_map_path, savefig=False)

    def pixedfit_plot_image_stamps(self):
        if not hasattr(self, 'img_process'):
            raise Exception("Need to run pixedfit_processing first")
        fig = self.img_process.plot_image_stamps(savefig=False)
        return fig


    def pixedfit_plot_galaxy_region(self):
        if not hasattr(self, 'gal_region'):
            raise Exception("Need to run pixedfit_processing first")
        self.img_process.plot_gal_region(self.gal_region, savefig=False)
    
    def pixedfit_plot_segm_maps(self):
        if not hasattr(self, 'img_process'):
            raise Exception("Need to run pixedfit_processing first")
        self.img_process.plot_segm_maps(savefig=False)

    def pixedfit_plot_binmap(self):
        if not hasattr(self, 'pixedfit_binmap_path'):
            raise Exception("Need to run pixedfit_binning first")
        from piXedfit.piXedfit_bin import plot_binmap
        plot_binmap(self.pixedfit_binmap_path, plot_binmap_spec=False, savefig=False)

    def measure_flux_in_bins(self, override_psf_type=None, binmap_type='pixedfit', overwrite = False):

        if hasattr(self, 'use_psf_type') and override_psf_type is None:
            psf_type = self.use_psf_type
        else:
            psf_type = override_psf_type

        if not hasattr(self, f'{binmap_type}_map'):
            raise Exception(f"Need to run {binmap_type}_binning first")
        # Sum fluxes in each bins and produce a table

        if hasattr(self, f'photometry_table') and self.photometry_table is not None:
            if psf_type in self.photometry_table.keys() and binmap_type in self.photometry_table[psf_type].keys() and not overwrite:
                print(f'Photometry table already exists for {psf_type} and {binmap_type}')
                return self.photometry_table[psf_type][binmap_type]

        binmap = getattr(self, f'{binmap_type}_map')
        table = QTable()
        table['ID'] = [str(i) for i in range(1, int(np.max(binmap))+1)]
        table['type'] = 'bin'
        for pos, band in enumerate(self.bands):
            fluxes = []
            flux_errs = []
            for i in range(1, int(np.max(binmap))+1):
                flux = np.sum(self.psf_matched_data[psf_type][band][binmap == i])
                flux_err = np.sqrt(np.sum(self.psf_matched_rms_err[psf_type][band][binmap == i]**2))
                if type(flux) in [u.Quantity, u.Magnitude]:
                    assert flux.unit == self.phot_pix_unit[band]
                    assert flux_err.unit == self.phot_pix_unit[band]
                    flux = flux.value
                    
                fluxes.append(flux)
                flux_errs.append(flux_err)
            
            table[band] = u.Quantity(np.array(fluxes), unit = self.phot_pix_unit[band])
            table[f'{band}_err'] = u.Quantity(np.array(flux_errs), unit = self.phot_pix_unit[band])
        
        # Add sum of all rows
        row = ['TOTAL_BIN', 'TOTAL_BIN']
        for pos, band in enumerate(self.bands):
            row.append(np.sum(table[band]))
            row.append(np.sqrt(np.sum(table[f'{band}_err']**2)))
        table.add_row(row)

        # Add MAG_AUTO and MAGERR_AUTO
        for i in ['MAG_AUTO', 'MAG_ISO', 'MAG_BEST']:
            row = [i, i]
            self.get_filter_wavs()
            for pos, band in enumerate(self.bands):
                
                try:
                    mag = self.auto_photometry[band][i] * u.ABmag
                    try:
                        mag_err = self.auto_photometry[band][i.replace('_', 'ERR_')]
                    except:
                        mag_err = 0.1 
                        print(f'No {i} error found for {band}, setting to 0.1 mag')
                    flux = mag.to(u.uJy, equivalencies=u.spectral_density(self.filter_wavs[band]))
                    flux_err = 0.4 * np.log(10) * flux * mag_err

                except KeyError as e:
                    print(e)
                    print(f'WARNING! No {i} found for {band}, falling back to aperture photometry!! Should switch to SEP. ')
                    flux = self.aperture_dict[str(0.32*u.arcsec)]['flux'][pos] * u.Jy
                    flux_err = self.aperture_dict[str(0.32*u.arcsec)]['flux_err'][pos] * u.Jy

                flux = flux.to(u.uJy)
                flux_err = flux_err.to(u.uJy)
                row.append(flux)
                row.append(flux_err)
            table.add_row(row)
        # Add MAG_APER
        for aper in self.aperture_dict.keys():
            row = [f'MAG_APER_{aper.value}', f'MAG_APER_{aper.value}']
            for pos, band in enumerate(self.bands):
                flux = self.aperture_dict[aper]['flux'][pos] * u.Jy
                flux_err = self.aperture_dict[aper]['flux_err'][pos] * u.Jy
                flux = flux.to(u.uJy)
                flux_err = flux_err.to(u.uJy)
                row.append(flux)
                row.append(flux_err)
            table.add_row(row)

        # Add total aperture fluxes
        if getattr(self, 'total_photometry', None) is not None:
            print('Adding total aperture fluxes.')
            row = ['MAG_APER_TOTAL', 'MAG_APER_TOTAL']
           
            for pos, band in enumerate(self.bands):
                flux = self.total_photometry[band]['flux']
                flux_err = self.total_photometry[band]['flux_err']
                if type(flux) != u.Quantity:
                    flux_unit = u.Unit(self.total_photometry[band]['flux_unit'])
                    flux *= flux_unit
                    flux_err *= flux_unit
                row.append(flux)
                row.append(flux_err)

            table.add_row(row)


        if not (hasattr(self, 'photometry_table') and self.photometry_table is None):
            self.photometry_table = {psf_type: {binmap_type: table}}
        else:
            self.photometry_table[psf_type][binmap_type] = table
        
        
        # Write table to our existing h5 file
        write_table_hdf5(table, self.h5_path, f'binned_photometry_table/{psf_type}/{binmap_type}', serialize_meta = True, overwrite=True, append=True)

        return table

    def provide_bagpipes_phot(self, id):
        '''Provide the fluxes in the correct format for bagpipes'''
        if not hasattr(self, 'photometry_table'):
            raise Exception("Need to run measure_flux_in_bins first")
        if hasattr(self, 'use_psf_type'):
            psf_type = self.use_psf_type
        else:
            psf_type = 'webbpsf'

        if hasattr(self, 'use_binmap_type'):
            binmap_type = self.use_binmap_type
        else:
            binmap_type = 'pixedfit'

        if hasattr(self, 'sed_min_percentage_err'):
            min_percentage_error = self.sed_min_percentage_err
        else:
            min_percentage_error = 5

        if id == 1:
            print(f'Using {psf_type} PSF and {binmap_type} binning derived fluxes')        
        flux_table = self.photometry_table[psf_type][binmap_type]
        
        if flux_table[flux_table.colnames[1]].unit != u.uJy:
            for col in flux_table.colnames[2:]:
                flux_table[col] = flux_table[col].to(u.uJy)
        
        for band in self.bands:
            flux_col_name = band
            fluxerr_col_name = f'{band}_err'
            # Where the error is less than 10% of the flux, set the error to 10% of the flux, if the flux is greater than 0
            mask = (flux_table[fluxerr_col_name]/flux_table[flux_col_name] < min_percentage_error/100)  & (flux_table[flux_col_name]>0)
            flux_table[fluxerr_col_name][mask] = min_percentage_error/100 * flux_table[flux_col_name][mask]
        
        row = flux_table[flux_table['ID'] == id]

        if len(row) == 0:
            raise Exception(f'ID {id} not found in flux table')
        elif len(row) > 1:
            raise Exception(f'More than one ID {id} found in flux table')


        row = Table(row)

        order = list(np.ndarray.flatten(np.array([[f'{band}',f'{band}_err'] for pos, band in enumerate(self.bands)])))
        
        table_order = row[order]

        flux, err = [], []
        for pos, item in enumerate(table_order[0]):	
            if pos % 2 == 0:
                flux.append(item)
            else:	
                err.append(item)
        final = np.vstack((np.array(flux), np.array(err))).T

        return final

    def plot_photometry_bins(self, binmap_type='pixedfit', bins_to_show='all', 
                            wav_unit = u.um, flux_unit = u.uJy, marker_colors='black',
                            save=False, save_path=None, show=True, facecolor='white'):

        self.get_filter_wavs()
        #from galfind import Filter
       
        # Get photometry table
        if not hasattr(self, 'photometry_table'):
            raise Exception("Need to run measure_flux_in_bins first")
        if hasattr(self, 'use_psf_type'):
            psf_type = self.use_psf_type
        else:
            psf_type = 'webbpsf'

        table = self.photometry_table[psf_type][binmap_type]
        
        fig = plt.figure(figsize=(8, 6), facecolor=facecolor, constrained_layout=True)
        ax = fig.add_subplot(111)
        
        if bins_to_show == 'all':
            bins_to_show = np.unique(table['ID'])

        if type(marker_colors) == str:
            marker_colors = [marker_colors for i in range(len(bins_to_show))]
        colorss = []
        for i, bin in enumerate(bins_to_show):
            mask = table['ID'] == bin
            name = table['type'][mask][0]
            if name == 'TOTAL_BIN':
                color = 'red'
            if name == 'MAG_AUTO':
                color = 'blue'
            if name  == 'MAG_ISO':
                color = 'green'
            if name  == 'MAG_BEST':
                color = 'orange'
            if name == 'MAG_APER_TOTAL':
                color = 'aqua'
            if name.startswith('MAG_APER'):
                color = 'purple'
            if name  == 'bin':
                color = marker_colors[i]
            
            for j, band in enumerate(self.bands):
                flux = table[mask][band]
                flux_err = table[mask][f'{band}_err']
                wav = self.filter_wavs[band]
                #print(wav, flux, flux_err)
                ax.errorbar(wav.to(wav_unit), flux.to(flux_unit), yerr=flux_err.to(flux_unit), fmt='o', color=color, label = name if color not in colorss else '')
                colorss.append(color)
            # Plot the Bagpipes input
            tab = self.provide_bagpipes_phot(bin)
            for row, band in zip(tab, self.bands):
                

                flux, err = row * u.uJy
                wav = self.filter_wavs[band]
                #ax.errorbar(wav.to(wav_unit)+0.05*u.um, flux.to(flux_unit), yerr=err.to(flux_unit), fmt='x', color=color)

        ax.set_xlabel(f'Wavelength ({wav_unit})')
        ax.set_ylabel(f'Flux ({flux_unit})')
        ax.legend()

    def run_dense_basis(self, db_atlas_name, db_dir=db_dir, overwrite=False, use_emcee=False, emcee_samples=10000, plot = False):
        

        import dense_basis as db
        import glob


        if not hasattr(self, 'photometry_table'):
            raise Exception("Need to run measure_flux_in_bins first")
        if hasattr(self, 'use_psf_type'):
            psf_type = self.use_psf_type
        else:
            psf_type = 'webbpsf'

        if hasattr(self, 'use_binmap_type'):
            binmap_type = self.use_binmap_type
        else:
            binmap_type = 'pixedfit'

        if hasattr(self, 'sed_fitting_table'):
            if 'dense_basis' in self.sed_fitting_table.keys():
                if db_atlas_name in self.sed_fitting_table['dense_basis'].keys():
                    if not overwrite:
                        print(f'Run {db_atlas_name} already exists')
                        return

        flux_table = self.photometry_table[psf_type][binmap_type]


        path = glob.glob(f'{db_dir}/{db_atlas_name}*.dbatlas')[0]
        N_param = int(atlas_path.split('N_param_')[1].split('.dbatlas')[0])
        N_pregrid = int(atlas_path.split('N_pregrid_')[1].split('_N_param')[0])
        atlas = db.load_atlas(atlas_path, N_pregrid = N_pregrid, N_param = N_param, path = db_dir)

        # Need to generate obs_sed, obs_err, and fit_mask based on the input filter files
       
        if use_emcee:
            import emcee
            sampler = db.run_emceesampler(obs_sed, obs_err, atlas, epochs=emcee_samples, plot_posteriors=plot)


        else:
             # pass the atlas and the observed SED + uncertainties into the fitter,
            sedfit = db.SedFit(obs_sed, obs_err, atlas, fit_mask=[])

            # evaluate_likelihood returns the likelihood for each SED in the atlas and the norm value to
            # best match the observed SED with the atlas.
            sedfit.evaluate_likelihood()

            # evaluate_posterior_percentiles calculates the 16,50,84th percentiles for
            # the physical parameters - stellar mass, SFR, tx, dust, metallicity and redshift
            sedfit.evaluate_posterior_percentiles()
    
    def run_bagpipes(self, bagpipes_config, filt_dir = bagpipes_filter_dir, fit_photometry = 'all',
    run_dir = f'pipes/', overwrite=False):
        #meta - run_name, use_bpass, redshift (override)
        assert type(bagpipes_config) == dict, "Bagpipes config must be a dictionary" # Could load from a file as well
        meta = bagpipes_config.get('meta', {})
        # Override fit_photometry if it is in the meta
        if 'fit_photometry' in meta.keys():
            fit_photometry = meta['fit_photometry']

        print(f'Fitting photometry: {fit_photometry}')

        fit_instructions = bagpipes_config.get('fit_instructions', {})

        use_bpass = meta.get('use_bpass', False)
        os.environ['use_bpass'] = str(int(use_bpass))
        run_name = meta.get('run_name', 'default')
        redshift = meta.get('redshift', self.redshift) # PLACEHOLDER for self.redshift
        if type(redshift) == str:
            # logic to choose a redshift to fit at
            if redshift == 'eazy':
                print('Using EAZY redshift.')
                redshift = self.meta_properties['zbest_fsps_larson_zfree']
            elif redshift in self.sed_fitting_table['bagpipes'].keys():
                table = self.sed_fitting_table['bagpipes'][redshift]
                redshift_id = meta.get('redshift_id', table['ID'][0])
                row_index = table['ID'] == redshift_id
                table = table[row_index]

                if 'redshift_50' in table.colnames:
                    redshift = table['redshift_50'][redshift_id]
                elif 'input_redshift' in table.colnames:
                    redshift = table['input_redshift'][redshift_id]
            

        redshift_sigma = meta.get('redshift_sigma', 0)
        if redshift_sigma == 'eazy':
            z50 = self.meta_properties['zbest_fsps_larson_zfree']
            z16 = self.meta_properties['zbest_16_fsps_larson_zfree']
            z84 = self.meta_properties['zbest_84_fsps_larson_zfree']
            if type(z84) in [list, np.ndarray]: z84 = z84[0]
            if type(z16) in [list, np.ndarray]: z16 = z16[0]
            redshift_sigma = np.mean([z84 - z50, z50 - z16])      

        min_redshift_sigma = meta.get('min_redshift_sigma', 0)
        sampler = meta.get('sampler', 'multinest')
        if redshift_sigma != 0:
            redshift_sigma = max(redshift_sigma, min_redshift_sigma)  
        else:
            redshift_sigma = None


        if type(redshift) == float:
            print(f'Fitting Redshift = {redshift}, Redshift sigma = {redshift_sigma}, allowed range = ({redshift - 3*redshift_sigma}, {redshift + 3*redshift_sigma})')
        elif type(redshift) in [list, np.ndarray, tuple]:
            assert len(redshift) == 2, "Redshift must be a float or a list of length 2"
            print(f'Allowing free redshift: {redshift[0]} < z < {redshift[1]}')
            fit_instructions['redshift'] = (redshift[0], redshift[1])
            redshift = None

        if not hasattr(self, 'photometry_table'):
            raise Exception("Need to run measure_flux_in_bins first")
        if hasattr(self, 'use_psf_type'):
            psf_type = self.use_psf_type
        else:
            psf_type = 'webbpsf'

        if hasattr(self, 'use_binmap_type'):
            binmap_type = self.use_binmap_type
        else:
            binmap_type = 'pixedfit'

        if hasattr(self, 'sed_fitting_table'):
            if 'bagpipes' in self.sed_fitting_table.keys():
                if run_name in self.sed_fitting_table['bagpipes'].keys():
                    if not overwrite:
                        print(f'Run {run_name} already exists')
                        return

        flux_table = self.photometry_table[psf_type][binmap_type]
        if fit_photometry == 'all':
            mask = np.ones(len(flux_table), dtype=bool)
        elif fit_photometry == 'bin':
            mask = flux_table['type'] == 'bin'
        elif fit_photometry == 'all_total':
            mask = (flux_table['type'] != 'bin') 
        elif fit_photometry == 'MAG_AUTO':
            mask = flux_table['type'] == 'MAG_AUTO'
        elif fit_photometry == 'MAG_ISO':
            mask = flux_table['type'] == 'MAG_ISO'
        elif fit_photometry == 'MAG_BEST':
            mask = flux_table['type'] == 'MAG_BEST'
        elif fit_photometry == 'TOTAL_BIN':
            mask = flux_table['type'] == 'TOTAL_BIN'
        elif fit_photometry == 'MAG_APER_TOTAL':
            mask = flux_table['type'] == 'MAG_APER_TOTAL'
        elif fit_photometry == 'MAG':
            mask = (flux_table['type'] == 'MAG_AUTO') & (flux_table['type'] == 'MAG_ISO') & (flux_table['type'] == 'MAG_BEST')
        elif fit_photometry.startswith('MAG_APER'):
            mask = flux_table['type'] == 'MAG_APER'
        elif fit_photometry == 'TOTAL_BIN+MAG_APER_TOTAL':
            mask = (flux_table['type'] == 'MAG_APER_TOTAL') | (flux_table['type'] == 'TOTAL_BIN')
        else:
            raise ValueError('fit_photometry must be one of: all, bin, MAG_AUTO, MAG_ISO, MAG_BEST, TOTAL_BIN, MAG or MAG_APER')
       
        flux_table = flux_table[mask]
        print(f'Fitting only {fit_photometry} fluxes, which is {len(flux_table)} sources')
            
        ids = list(flux_table['ID'])
        print(ids)
        
        nircam_filts = [f'{filt_dir}/{band}_LePhare.txt' for band in self.bands]
        
        if redshift is None:
            print('Allowing free redshift.')
            redshifts = None
        else:
            redshifts = np.ones(len(flux_table)) * redshift

        import bagpipes as pipes
        # out_subdir (moving files)

        out_subdir = f'{run_name}/{self.survey}/{self.galaxy_id}'
        # make a temp directory
        # hash out_subdir to get a unique shorter name
        import hashlib
        out_subdir_encoded = hashlib.md5(out_subdir.encode()).hexdigest()
        

        existing_files = []
        for i in [out_subdir, out_subdir_encoded]:
            for j in ['posterior', 'plots', 'seds']:
                path = f'{run_dir}/{j}/{i}'
                os.makedirs(path, exist_ok=True)
                os.chmod(path, 0o777)
                existing_files += glob.glob(f'{path}/*')    

        os.makedirs(f'{run_dir}/cats/{out_subdir_encoded}', exist_ok=True)

        existing_filenames = [os.path.basename(file) for file in existing_files]  
        
        #path_fits = f'{run_dir}/cats/{run_name}/{self.survey}/' # Filename is galaxy ID rather than being a folder
        
        #for path in [path_post, path_plots, path_sed, path_fits]:
        #    os.makedirs(path, exist_ok=True)
        #    os.chmod(path, 0o777)

        #existing_files = glob.glob(f'{path_post}/*') + glob.glob(f'{path_plots}/*') + glob.glob(f'{path_sed}/*') + glob.glob(f'{path_fits}/*')
        # Copy any .h5 in out_subdir_encoded to out_subdir
        for file in existing_files:
            if out_subdir_encoded in file:
                new_file = file.replace(out_subdir_encoded, out_subdir)
                if not os.path.exists(new_file):
                    os.rename(file, new_file)
                else:
                    if overwrite:
                        os.remove(new_file)
                        os.rename(file, new_file)
                    else:
                        os.remove(file)

        print(existing_filenames)

        if overwrite:
            for file in existing_files:
                os.remove(file)
        else: # Check if already run
            mask = np.zeros(len(ids))
            for pos, id in enumerate(ids):
                print(f'{id}.h5')
                if f'{id}.h5' in existing_filenames:
                    print(f'{id}.h5 already exists')
                    mask[pos] = 1
            
            if np.all(mask == 1):
                print('All files already exist')
                self.load_bagpipes_results(run_name)
                return

        if np.any(mask == 1):
            # Rename the out_subdir to out_subdir_encoded
            for i in ['posterior', 'plots', 'seds']:
                path = f'{run_dir}/{i}/{out_subdir}'
                new_path = f'{run_dir}/{i}/{out_subdir_encoded}'
                os.rename(path, new_path)
            # Move catalogue
            
            path = f'{run_dir}/cats/{run_name}/{self.survey}/{self.galaxy_id}.fits'
            run_name = f'{run_dir}/cats/{out_subdir_encoded}/{self.galaxy_id}.fits'
            if os.path.exists(path) and not os.path.exists(run_name):
                os.rename(path, run_name)
            
        # Default pipes install for now

        fit_cat = pipes.fit_catalogue(ids, fit_instructions, self.provide_bagpipes_phot,
                                    spectrum_exists=False, photometry_exists=True, run=out_subdir_encoded,
                                    make_plots=False, cat_filt_list=nircam_filts, redshifts = redshifts,
                                    redshift_sigma = redshift_sigma,
                                    full_catalogue=True) #analysis_function=custom_plotting,
        print('Beginning fit')
        print(fit_instructions)
        # Run this with MPI

        fit_cat.fit(verbose=False, mpi_serial = True, sampler = sampler)

        # Move files to the correct location
        for i in ['posterior', 'plots', 'seds']:
            path = f'{run_dir}/{i}/{out_subdir}'
            new_path = f'{run_dir}/{i}/{out_subdir_encoded}'
            os.rename(path, new_path)
        # Move catalogue
        path = f'{run_dir}/cats/{run_name}/{self.survey}/{self.galaxy_id}.fits'
        old_name = f'{run_dir}/cats/{out_subdir_encoded}/{self.galaxy_id}.fits'
        os.rename(old_name, path)

        '''
        json_file = json.dumps(fit_instructions)
        f = open(f'{path_overall}/posterior/{out_subdir}/config.json',"w")
        f.write(json_file)
        f.close()
        '''


        self.load_bagpipes_results(run_name)

    def load_bagpipes_results(self, run_name, run_dir = f'pipes/'):
        
        catalog_path = f'{run_dir}/cats/{run_name}/{self.survey}/{self.galaxy_id}.fits'
        try:
            table = Table.read(catalog_path)
        except:
            raise Exception(f'Catalog {catalog_path} not found')

        # SFR map
        # Av map (dust)
        # Stellar Mass map
        # Metallicity map
        # Age map
        if getattr(self, 'sed_fitting_table', None) is None:
            self.sed_fitting_table = {'bagpipes': {run_name: table}}
        else:
            if 'bagpipes' not in self.sed_fitting_table.keys():
                self.sed_fitting_table['bagpipes'] = {run_name: table}
            else:
                self.sed_fitting_table['bagpipes'][run_name] = table
        
        write_table_hdf5(self.sed_fitting_table['bagpipes'][run_name], self.h5_path, f'sed_fitting_table/bagpipes/{run_name}', serialize_meta=True, overwrite=True, append=True)

    def convert_table_to_map(self, table, id_col, value_col, map, remove_log10=False):
        changed = np.zeros(map.shape)
        return_map = copy.deepcopy(map)
        for row in table:
            id = row[id_col]
            try:
                id = float(id)
            except:
                continue
            value = row[value_col]
            return_map[map == float(id)] = value if not remove_log10 else 10**value
            changed[map == float(id)] = 1
        #print(f'changed {np.sum(changed)} pixels out of {len(map.flatten())} pixels')
        return_map[changed == 0] = np.nan
        return return_map

    def param_unit(self, param):
        zsun = u.def_unit('Zsun', format={'latex':r'Z_{\odot}', 'html': 'Z<sub>sun</sub>'})
        param_dict = {
            'stellar_mass':u.Msun,
            'stellar_mass_density':u.Msun/u.kpc**2,
            'formed_mass':u.Msun,
            'sfr':u.Msun/u.yr,
            'sfr_10myr':u.Msun/u.yr,
            'sfr_density':u.Msun/u.yr/u.kpc**2,
            'dust:Av':u.mag,
            'UV_colour':u.mag,
            'VJ_colour':u.mag,
            'ssfr':u.Gyr**-1,
            'ssfr_10myr':u.Gyr**-1,
            'nsfr_10myr':u.dimensionless_unscaled,
            'nsfr':u.dimensionless_unscaled,
            'mass_weighted_zmet':zsun,
            'm_UV':u.mag,
            'M_UV':u.mag,
            'Ha_EWrest':u.AA,
            'xi_ion_caseB':u.Hz/u.erg,
            'beta_C94':u.dimensionless_unscaled,
            'tform':u.Gyr,
            'tquench':u.Gyr,
            'age_min':u.Gyr,
            'age_max':u.Gyr,
            'massformed':u.Msun,
            'mass_weighted_age':u.Gyr,
            'chisq_phot-': u.dimensionless_unscaled,
            'metallicity': zsun,

        }
        return param_dict.get(param, u.dimensionless_unscaled)

    def plot_bagpipes_corner(self, run_name=None, bins_to_show = 'all', save=False, corner_bins=25, facecolor='white', colors='black', cache=None,
                            plotpipes_dir='pipes_scripts/',run_dir = f'pipes/'):
        
        if run_name is None:
            run_name = list(self.sed_fitting_table['bagpipes'].keys())
            if len(run_name) > 1:
                raise Exception("Multiple runs found, please specify run_name")
            else:
                run_name = run_name[0]

        table = self.sed_fitting_table['bagpipes'][run_name]
        
        if bins_to_show == 'all':
            bins_to_show = np.unique(table['#ID'])

        if type(colors) == str:
            colors = [colors for i in range(len(bins_to_show))]

        if cache is None:
            cache = {}
        
        fig = None
        x_lims = []
        y_lims = []

        for bin, color, in zip(bins_to_show, colors):
            h5_path = f'{run_dir}/posterior/{run_name}/{self.survey}/{self.galaxy_id}/{bin}.h5'

            
            pipes_obj = self.load_pipes_object(run_name, bin, run_dir = run_dir, cache=cache, plotpipes_dir=plotpipes_dir)
           
            fig_xlim, fig_ylim = [], []
            fig = pipes_obj.plot_corner_plot(show=False, save=save, bins=corner_bins, type="fit_params", fig=fig, color=color, facecolor=facecolor)
            for ax in fig.get_axes():
                fig_xlim.append(ax.get_xlim())
                fig_ylim.append(ax.get_ylim())
            x_lims.append(fig_xlim)
            y_lims.append(fig_ylim)
        
        if fig is not None:
            for pos, ax in enumerate(fig.get_axes()):
                all_xlim = [x_lims[i][pos] for i in range(len(x_lims))]
                all_ylim = [y_lims[i][pos] for i in range(len(y_lims))]
                ax.set_xlim(np.min(all_xlim), np.max(all_xlim))
                ax.set_ylim(np.min(all_ylim), np.max(all_ylim))
                #print(ax.get_xlabel(), np.min(all_xlim), np.max(all_xlim))
                #print(ax.get_ylabel(), np.min(all_ylim), np.max(all_ylim))
                if len(x_lims) > 1:
                    ax.set_title('')

        return fig, cache

    def plot_bagpipes_fit(self, run_name=None, axes = None, fig=None, bins_to_show = 'all', save=False, 
                        facecolor='white', marker_colors='black', wav_units=u.um, plotpipes_dir='pipes_scripts/',
                        flux_units=u.ABmag, lw=1,fill_uncertainty=False,zorder=5, run_dir = f'pipes/', cache=None):
        sys.path.insert(1, plotpipes_dir)
        
        if run_name is None:
            run_name = list(self.sed_fitting_table['bagpipes'].keys())
            if len(run_name) > 1:
                raise Exception("Multiple runs found, please specify run_name")
            else:
                run_name = run_name[0]
        if not hasattr(self, 'sed_fitting_table') or 'bagpipes' not in self.sed_fitting_table.keys() or run_name not in self.sed_fitting_table['bagpipes'].keys():
            self.load_bagpipes_results(run_name)
        table = self.sed_fitting_table['bagpipes'][run_name]

        if axes is None:
            fig, axes = plt.subplots(1, 1, figsize=(6, 3), constrained_layout=True, facecolor=facecolor)
        if bins_to_show == 'all':
            bins_to_show = np.unique(table['bin'])

        if type(marker_colors) == str:
            marker_colors = [marker_colors for i in range(len(bins_to_show))]

        if cache is None:
            cache = {}
        for bin, color, in zip(bins_to_show, marker_colors):
            h5_path = f'{run_dir}/posterior/{run_name}/{self.survey}/{self.galaxy_id}/{bin}.h5'

            pipes_obj = self.load_pipes_object(run_name, bin, run_dir = run_dir, cache=cache, plotpipes_dir=plotpipes_dir)

            pipes_obj.plot_best_fit(axes, color, wav_units=wav_units, flux_units=flux_units, lw=lw, fill_uncertainty=fill_uncertainty, zorder=zorder)

        #cbar.set_label('Age (Gyr)', labelpad=10)
        #cbar.ax.xaxis.set_ticks_position('top')
        #cbar.ax.xaxis.set_label_position('top')
        #cbar.ax.tick_params(labelsize=8)
        #cbar.ax.xaxis.set_major_formatter(ScalarFormatter())
        return fig, cache

    def plot_bagpipes_sfh(self, run_name=None, bins_to_show = 'all', save=False, 
                        facecolor='white', marker_colors='black', time_unit='Gyr', cmap = 'viridis',
                        plotpipes_dir='pipes_scripts/', 
                        run_dir = f'pipes/', cache=None):
        sys.path.insert(1, plotpipes_dir)

        if run_name is None:
            run_name = list(self.sed_fitting_table['bagpipes'].keys())
            if len(run_name) > 1:
                raise Exception("Multiple runs found, please specify run_name")
            else:
                run_name = run_name[0]
        if not hasattr(self, 'sed_fitting_table') or 'bagpipes' not in self.sed_fitting_table.keys() or run_name not in self.sed_fitting_table['bagpipes'].keys():
            self.load_bagpipes_results(run_name)
        table = self.sed_fitting_table['bagpipes'][run_name]

        fig, axes = plt.subplots(1, 1, figsize=(6, 3), constrained_layout=True, facecolor=facecolor)
        if bins_to_show == 'all':
            bins_to_show = np.unique(table['bin'])

        if type(marker_colors) == str and len(bins_to_show) > 1:
            cmap = plt.get_cmap(cmap)
            marker_colors = cmap(np.linspace(0, 1, len(bins_to_show)))
            #marker_colors = [marker_colors for i in range(len(bins_to_show))]

        if cache is None:
            cache = {}
        for bin, color, in zip(bins_to_show, marker_colors):
            h5_path = f'{run_dir}/posterior/{run_name}/{self.survey}/{self.galaxy_id}/{bin}.h5'
            if bin == 'RESOLVED':
                ''' Special case where we sum the SFH of all the bins'''
                dummy_fig, dummy_ax = plt.subplots(1, 1)
                set = False
                for pos, tbin in enumerate(np.unique(table['#ID'])):
                    
                    try:
                        float(tbin)
                        pipes_obj = self.load_pipes_object(run_name, tbin, run_dir = run_dir, cache=cache, plotpipes_dir=plotpipes_dir)
                        tx, tsfh = pipes_obj.plot_sfh(dummy_ax, color, timescale=time_unit, plottype='lookback', logify=False, cosmo=None, label = bin, return_sfh=True)
                        if pos == 0:
                            x_all = tx
                            y_all = tsfh
                            set = True
                        else:
                            assert np.all(x_all == tx), "Time scales do not match"
                            y_all = np.sum([y_all, tsfh], axis=0)
                            set = True
                        
                    except ValueError: 
                        pass
                    
                if set:
                    axes.plot(x_all, y_all[:, 1], color='tomato', label='RESOLVED', lw = 2)
                    axes.fill_between(x_all, y_all[:, 0], y_all[:, 2], color='tomato', alpha=0.5)
                
                plt.close(dummy_fig)
            else:
                pipes_obj = self.load_pipes_object(run_name, bin, run_dir = run_dir, cache=cache, plotpipes_dir=plotpipes_dir)
                pipes_obj.plot_sfh(axes, color, modify_ax = True, add_zaxis=True, timescale=time_unit, plottype='lookback', logify=False, cosmo=None, label = bin)
        
        axes.legend(fontsize=8)

        #cbar.set_label('Age (Gyr)', labelpad=10)
        #cbar.ax.xaxis.set_ticks_position('top')
        #cbar.ax.xaxis.set_label_position('top')
        #cbar.ax.tick_params(labelsize=8)
        #cbar.ax.xaxis.set_major_formatter(ScalarFormatter())
        if len(bins_to_show) == 0:
            fig = plt.figure(facecolor=facecolor)

        return fig, cache

    def add_flux_aper_total(self, catalogue_path, id_column = 'NUMBER', overwrite = False, columns_to_add = ['KRON_RADIUS_', 'a_', 'b_', 'theta_', 'area_', 'flux_ratio_', 'EE_correction_', 'total_correction_', 'FLUX_APER_band_TOTAL_Jy', 'FLUXERR_APER_band_TOTAL_Jy'], stacked_band = 'F277W+F356W+F444W'):
        
        if getattr(self, 'total_photometry', None) is not None and not overwrite:
            print('Total photometry already exists')
            return

        table = Table.read(catalogue_path)
        table = table[table[id_column] == self.galaxy_id]
        if len(table) == 0:
            raise Exception(f'ID {self.galaxy_id} not found in {table}.')

        total_photometry = {}
        band_cols = [col for col in columns_to_add if 'band' in col]
        other_cols = [col for col in columns_to_add if col.endswith('_')]
        
        for band in self.bands:
            zp = self.im_zps[band]
            total_photometry[band] = {}
            for col in band_cols:
                if 'band' in col:
                    col = col.replace('band', band)
                    total_photometry[band][col] = table[col][0]
                    if 'FLUX_' in col:
                        # Convert to uJy
                        flux = table[col][0] * u.Jy #10**(-zp/2.5) * 3631 * u.Jy
                        flux = flux.to(u.uJy)
                        total_photometry[band]['flux'] = flux.value
                        total_photometry[band]['flux_unit'] = str(flux.unit)

                    elif 'FLUXERR_' in col:
                        err = table[col][0] * u.Jy #10**(-zp/2.5) * 3631 * u.Jy 
                        total_photometry[band]['flux_err'] = err.to(u.uJy).value

       

        total_photometry[stacked_band] = {}
        for col in other_cols:
            data = table[f'{col}{stacked_band}'][0]
            if type(data) in [u.Quantity, u.Magnitude]:
                data = data.value

            total_photometry[stacked_band][f'{col}{stacked_band}'] = data
    
        self.add_to_h5(total_photometry, 'total_photometry', 'total_photometry', overwrite=True, setattr_gal = 'total_photometry')
        
    
    def plot_bagpipes_map_gif(self,  parameter = 'dust:Av', run_name = None, lazy = False, facecolor='white', cache = None, n_samples = 100, weight_mass_sfr = True, logmap = False, savetype = 'temp', path = 'temp', cmap = 'magma'):
        if not hasattr(self, 'sed_fitting_table') or 'bagpipes' not in self.sed_fitting_table.keys():
            print('No bagpipes results found.')
            return None

        if run_name is None:
            run_name = list(self.sed_fitting_table['bagpipes'].keys())
            if len(run_name) > 1:
                raise Exception("Multiple runs found, please specify run_name")
            else:
                run_name = run_name[0]
        
        map = self.pixedfit_map

        values = list(np.unique(map[~np.isnan(map)]))
        values = [int(val) for val in values if val != 0.0]

        map_3d = np.zeros((n_samples, map.shape[0], map.shape[1]))

        objects = self.load_pipes_object(run_name, values, cache=cache, get_advanced_quantities=False)
        if not lazy:
            for pos, bin in enumerate(values):
                obj = objects[pos]
                samples = obj.plot_pdf(None, parameter, return_samples=True)
                draws = np.random.choice(samples, n_samples)
                # For all pixels in map that are equal to bin, set the 1st dimension of map_3d to the draw
                mask = (map == bin)
                mask_x, mask_y = np.where(mask)
                # Generate random draws
                draws = np.random.choice(samples, n_samples)
                for pos, draw in enumerate(draws):
                    map_3d[pos, mask_x, mask_y] = draw
                    

        map_3d[:, map == 0] = np.nan
        map_3d[:, map == 0.0] = np.nan
        

        if parameter in ['stellar_mass', 'sfr']:
            density_map = True
            density = '$kpc^{-2}$'
            ref_band = {'stellar_mass':'F444W', 'sfr':'1500A'}
            if weight_mass_sfr:
                weight_by_band = ref_band[parameter]
            else:
                weight_by_band = False
            if parameter  == 'stellar_mass':
                map_3d = 10**map_3d
                #print(map_3d[0, 32, 32])
        else:
            density_map = False
            density = ''
            weight_by_band = False

        if logmap:
            log = '$log_{10}$'
        else:
            log = ''
        unit = self.param_unit(parameter.split(':')[-1])
        if unit != u.dimensionless_unscaled:
            unit = f'[{log}{unit:latex}{density}]'
        else:
            unit = ''

        cbar_label = f'{parameter.replace("_", " ")} {unit}'
        #print('density_map', density_map)

        gif = self.make_animation(map_3d, draw_random = False, facecolor=facecolor, n_draws = n_samples, html = True if savetype == 'html' else False, save=True if savetype == 'gif' else False, 
                                cbar_label = cbar_label, density_map = density_map, logmap = logmap, weight_by_band = weight_by_band, cmap = cmap)

        if path == 'temp':
            import tempfile
            path = tempfile.mktemp(suffix='.gif')
            gif.save(path, writer='pillow', fps=60)
            return path

        return gif

    def plot_bagpipes_results(self, run_name=None, parameters=['bin_map', 'stellar_mass', 'sfr', 'dust:Av', 'chisq_phot-', 'UV_colour'], reload_from_cat=False, save=False, facecolor='white', max_on_row=4, weight_mass_sfr = True, norm = 'linear', total_params = []):
        
        if not hasattr(self, 'sed_fitting_table') or 'bagpipes' not in self.sed_fitting_table.keys():
            print('No bagpipes results found.')
            return None

        if run_name is None:
            run_name = list(self.sed_fitting_table['bagpipes'].keys())
            if len(run_name) > 1:
                raise Exception("Multiple runs found, please specify run_name")
            else:
                run_name = run_name[0]
        cmaps = ['magma','RdYlBu', 'cmr.ember', 'cmr.cosmic', 'cmr.lilac', 'cmr.eclipse', 'cmr.sapphire', 'cmr.dusk', 'cmr.emerald']
        if not hasattr(self, 'sed_fitting_table') or 'bagpipes' not in self.sed_fitting_table.keys() or run_name not in self.sed_fitting_table['bagpipes'].keys() or reload_from_cat:
            self.load_bagpipes_results(run_name)
            
        if not hasattr(self, 'pixedfit_map'):
            raise Exception("Need to run pixedfit_binning first")
        # If it still isn't there, return None
        if not hasattr(self, 'sed_fitting_table') or 'bagpipes' not in self.sed_fitting_table.keys() or run_name not in self.sed_fitting_table['bagpipes'].keys():
            return None

        table = self.sed_fitting_table['bagpipes'][run_name]

        #fig, axes = plt.subplots(1, len(parameters), figsize=(4*len(parameters), 4), constrained_layout=True, facecolor=facecolor)
        fig, axes = plt.subplots(len(parameters)//max_on_row + 1, max_on_row, figsize=(2.5*max_on_row, 2.5*(len(parameters)//max_on_row + 1)), constrained_layout=True, facecolor=facecolor, sharex=True, sharey=True)
        # add gap between rows using get_layout_engine
        fig.get_layout_engine().set( h_pad=4 / 72, hspace=0.2)

        axes = axes.flatten()
        # Remove empty axes
        for i in range(len(parameters), len(axes)):
            fig.delaxes(axes[i])

        redshift = self.sed_fitting_table['bagpipes'][run_name]['input_redshift'][0]
        
        for i, param in enumerate(parameters):
            ax_divider = make_axes_locatable(axes[i])
            cax = ax_divider.append_axes('top', size='5%', pad='2%')

            if param == 'bin_map':
                map = copy.copy(self.pixedfit_map)
                map[map == 0] = np.nan
                log = ''
            else:
                param_name = f'{param[:-1]}' if param.endswith('-') else f'{param}_50'
                map = self.convert_table_to_map(table, '#ID', param_name, self.pixedfit_map, remove_log10=param.startswith('stellar_mass'))
                log = ''
                
            if param in ['stellar_mass', 'sfr']:
                ref_band = {'stellar_mass':'F444W', 'sfr':'1500A'}
                if weight_mass_sfr:
                    weight = ref_band[param]
                else:
                    weight = False
                

                map = self.map_to_density_map(map, redshift = redshift, weight_by_band=weight, logmap = True)
                #map = self.map_to_density_map(map, redshift = redshift, logmap = True) 
                log = '$\log_{10}$ '
                param = f'{param}_density'
 

            if norm == 'log':
                norm = LogNorm(vmin=np.nanmin(map), vmax=np.nanmax(map))
            else:
                norm = Normalize(vmin=np.nanmin(map), vmax=np.nanmax(map))

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

            # Create actual normalisation
            

            mappable = axes[i].imshow(map, origin='lower', interpolation='none', cmap=cmaps[i], norm=norm)
            cbar = fig.colorbar(mappable, cax=cax, orientation='horizontal')
            
            # ensure cbar is using ScalarFormatter
            cbar.ax.xaxis.set_major_formatter(ScalarFormatter())
            cbar.ax.xaxis.set_minor_formatter(ScalarFormatter())
            
            
            cbar.set_label(rf'$\rm{{{param_str}}}${unit}', labelpad=10, fontsize=8)
            cbar.ax.xaxis.set_ticks_position('top')
            cbar.ax.xaxis.set_tick_params(labelsize=6, which='both')
            cbar.ax.xaxis.set_label_position('top')
            #disable xtick labels
            #cax.set_xticklabels([])
            # Fix colorbar tick size and positioning if log
            if norm == 'log':
                # Generate reasonable ticks
                ticks = np.logspace(np.log10(np.nanmin(map)), np.log10(np.nanmax(map)), num=5)
                cbar.set_ticks(ticks)
                cbar.ax.xaxis.set_major_formatter(ScalarFormatter())
                cbar.ax.xaxis.set_minor_formatter(ScalarFormatter())
                cbar.ax.xaxis.set_tick_params(labelsize=6, which='both')
                cbar.ax.xaxis.set_label_position('top')
                cbar.ax.xaxis.set_ticks_position('top')
                cbar.update_ticks()
                
            


        if save:
            fig.savefig(f'galaxies/{run_name}_maps.png', dpi=300, bbox_inches='tight')
        return fig

    def map_to_density_map(self, map, cosmo = FlatLambdaCDM(H0=70, Om0=0.3), redshift = None, logmap = False, weight_by_band = False, psf_type = 'star_stack', binmap_type = 'pixedfit'):
        pixel_scale = self.im_pixel_scales[self.bands[0]]
        density_map = copy.copy(map)
        if binmap_type == 'pixedfit':
            pixel_map = self.pixedfit_map
        else:
            raise Exception('Only pixedfit binning is supported for now')
        
        for id in np.unique(pixel_map):
            #print(id)
            if id == 0:
                continue
            mask = pixel_map == id
        
            if redshift is None:
                z = self.redshift
            else:
                z = redshift

            re_as = pixel_scale 
            d_A = cosmo.angular_diameter_distance(z) #Angular diameter distance in Mpc
            pix_kpc = (re_as * d_A).to(u.kpc, u.dimensionless_angles()) # re of galaxy in kpc
            pix_area = np.sum(mask) * pix_kpc**2 #Area of pixels with id in kpc^2
            # Check if all values in map[mask] are equal
            if len(np.unique(map[mask])) > 1:
                raise Exception(f'This was supposed to be equal {id}')
            
            if weight_by_band:

                if weight_by_band.endswith('A'):
                    # Calculate which band is closest to the wavelength
                    wav = float(weight_by_band[:-1]) * u.AA
                    # Convert to observed frame
                    obs_wav = wav * (1 + z)

                    self.get_filter_wavs()
                    
                    fmask = [self.filter_wavs[band].to(u.AA).value > obs_wav.value for band in self.bands]
                    fmask = np.array(fmask, dtype=bool)
    
                    pos = np.argwhere(fmask == True)[0][0]
                    band = self.bands[pos]
                    #print(f'Using band {band} at wavelength {self.filter_wavs[band].to(u.AA)} for weighting at {obs_wav} (rest frame {wav})')
                else:
                    band = weight_by_band
                if self.use_psf_type:
                    psf_type = self.use_psf_type
                
                data_band = self.psf_matched_data[psf_type][band]
                norm = np.sum(data_band[mask])

                x, y = np.where(mask)
                for xi, yi in zip(x, y):
                    value = map[xi, yi]
                    #print(mask[xi, yi], data_band[xi, yi] / norm)
                    weighted_val = value * data_band[xi, yi] / norm
                    weighted_val_area = weighted_val / pix_kpc**2
                    
                    if logmap:
                        weighted_val_area = np.log10(weighted_val_area.value)
                    if type(weighted_val_area) in [u.Quantity, u.Magnitude]:
                        weighted_val_area = weighted_val_area.value
                    density_map[xi, yi] = weighted_val_area
                #assert np.sum(density_map[mask]) == value, f'overall sum should still correspond, {np.sum(density_map[mask])} != {value}'
            else:
                value = np.unique(map[mask]) / pix_area
                if type(value) in [u.Quantity, u.Magnitude]:
                    value = value.value

                if logmap:
                    value = np.log10(value)

                density_map[mask] = value

        return density_map
    
    def load_pipes_object(self, run_name, bin, run_dir = f'pipes/', bagpipes_filter_dir=bagpipes_filter_dir, cache=None, plotpipes_dir='pipes_scripts/', get_advanced_quantities = True, n_cores = 1):
        plotpipes_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), plotpipes_dir)
        sys.path.insert(1, plotpipes_dir)
        #print(plotpipes_dir)

        if type(bin) not in [list, np.ndarray]:
            single = True
            bin = [bin]
        else:
            single = False

        output = {}
        from plotpipes import PipesFit
        found = False
        if cache is not None:
            for b in bin:
                if b in cache.keys():
                    output[b] = cache[b]
                    

        if run_name in self.internal_bagpipes_cache.keys():
            for b in bin:
                if b in self.internal_bagpipes_cache[run_name].keys():    
                    output[b] = self.internal_bagpipes_cache[run_name][b]
        else:
            self.internal_bagpipes_cache[run_name] = {}
        
        still_to_load = [b for b in bin if b not in output.keys()]
        if len(still_to_load) > 0:
            params_dict = {'galaxy_id':None, 'field':self.survey, 'h5_path':None, 'pipes_path':run_dir, 'catalog':None, 'overall_field':None,
                'load_spectrum':False, 'filter_path':bagpipes_filter_dir, 'ID_col':'NUMBER', 'field_col':'field', 'catalogue_flux_unit':u.MJy/u.sr,
                 'bands':self.bands, 'data_func':self.provide_bagpipes_phot, 'get_advanced_quantities': get_advanced_quantities}
            
            params_b = []
            for b in still_to_load:
                param = copy.copy(params_dict)
                param['galaxy_id'] = b
                param['h5_path'] = f'{run_dir}/posterior/{run_name}/{self.survey}/{self.galaxy_id}/{b}.h5'
                params_b.append(param)

            from joblib import Parallel, delayed
            pipes_objs = Parallel(n_jobs=n_cores)(delayed(PipesFit)(**param) for param in params_b)

            for obj, b in zip(pipes_objs, still_to_load):
                output[b] = obj
                if cache is not None:
                        cache[b] = obj
            
                self.internal_bagpipes_cache[run_name][b] = obj
        
        # Set output as list in order of input
        pipes_obj = [output[b] for b in bin]
        if get_advanced_quantities:
            for obj in pipes_obj:
                if not obj.has_advanced_quantities:
                    print('Adding advanced quantities')
                    obj.add_advanced_quantities()

        if single:
            pipes_obj = pipes_obj[0]

        return pipes_obj


    def plot_bagpipes_component_comparison(self, parameter = 'stellar_mass', run_name=None, 
                                        bins_to_show = 'all', save=False, run_dir = f'pipes/', 
                                        facecolor='white', plotpipes_dir='pipes_scripts/', 
                                        bagpipes_filter_dir=bagpipes_filter_dir, n_draws = 10000,
                                        cache = None, colors = None):
        
        plotpipes_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), plotpipes_dir)
        #print(plotpipes_dir)
        sys.path.insert(1, plotpipes_dir)

        from plotpipes import hist1d
        if run_name is None:
            run_name = list(self.sed_fitting_table['bagpipes'].keys())
            if len(run_name) > 1:
                raise Exception("Multiple runs found, please specify run_name")
            else:
                run_name = run_name[0]
        
        if not hasattr(self, 'sed_fitting_table') or 'bagpipes' not in self.sed_fitting_table.keys() or run_name not in self.sed_fitting_table['bagpipes'].keys():
            self.load_bagpipes_results(run_name)

        table = self.sed_fitting_table['bagpipes'][run_name]

        bins = []
        if bins_to_show == 'all':
            show_all = True
            bins_to_show = table['#ID']
            for bin in bins_to_show:
                try:
                    bin = float(bin)
                except:
                    bins.append(bin)

        else:
            show_all = False
            bins = bins_to_show
        
        fig, ax_samples = plt.subplots(1, 1, figsize=(3, 3), constrained_layout=True, facecolor=facecolor)

        if cache is None:
            cache = {}

        all_samples = []
        if colors is None:
            colors = mcm.get_cmap('cmr.guppy', len(bins))
            colors = {bin:colors(i) for i, bin in enumerate(bins)}
        if type(colors) == list:
            assert len(colors) == len(bins), "Need to provide a color for each bin"
            colors = {bin:colors[i] for i, bin in enumerate(bins)}
        
        
        for bin in bins_to_show:
            pipes_obj = self.load_pipes_object(run_name, bin, run_dir = run_dir, cache=cache, plotpipes_dir=plotpipes_dir)
            
            bin_number = True
            try:
                bin = float(bin)
            except:
                bin_number = False
            # This will need to change huh. 
            if show_all:
                datta = ax_samples if not bin_number else None
                collors = colors[bin] if not bin_number else 'black'
                labbels = str(bin) if not bin_number else None
                ret_samples = True
            else:
                datta = ax_samples
                collors = colors[bin]
                labbels = str(bin)
                ret_samples = False
            #print('parameter', parameter)
            samples = pipes_obj.plot_pdf(datta, parameter,
            return_samples = ret_samples, linelabel = labbels, 
            colour=collors, norm_height=True)

            if ret_samples:
                all_samples.append(samples)
        # Sum all samples
        
        if 'all' in bins_to_show:
            all_samples = np.array(all_samples, dtype=object)

            #all_samples = all_samples[~np.isnan(all_samples)]
            print(f'Combining {len(all_samples)} samples for {parameter}')
            #all_samples = all_samples.T
            new_samples = np.zeros((n_draws, len(all_samples)))
            for i, samples in enumerate(all_samples):
                #samples = samples[~np.isnan(samples)]
                new_samples[:, i] = np.random.choice(samples, size=n_draws)

            sum_samples = np.log10(np.sum(10**new_samples, axis=1))
            #print(len(sum_samples))
            # Normalize height of histogram

            hist1d(sum_samples, ax_samples,
                    smooth=True, color='black', percentiles=False, lw=1, alpha=1, fill_between=False, norm_height=True, label='$\Sigma$ Resolved')

        # Fix xticks
        #ax_samples.set_xticks(np.arange(ax_samples.get_xlim()[0], ax_samples.get_xlim()[1], 0.5))
        ax_samples.legend(fontsize=6)

        gunit = self.param_unit(parameter.split(':')[-1])
        unit = f' ({gunit:latex})' if gunit != u.dimensionless_unscaled else ''
        param_str = parameter.replace("_", r" ")

        ax_samples.set_xlabel(f'{param_str} {unit}')
        print(f'{param_str} {unit}')
        return fig, cache


    def make_animation(self, map_3d, draw_random = True, n_draws = 10, facecolor='white', scale = 'linear', save=False, html=False, cbar_label = None, density_map = False, logmap = False, weight_by_band = False, path = 'temp', cmap = 'magma'): 
        
        map_3d = copy.copy(map_3d)

        # Check map has 3 dimensions
        if len(map_3d.shape) != 3:
            raise Exception('Map must have 3 dimensions')
        
        if draw_random:
            # Make n_draws maps of random indices of the same shape as the 2nd and 3rd dimensions of map_3d

            N, M, _ = map_3d.shape

            # Create an array of random indices for each pixel
            random_indices = np.random.randint(0, N, size=(n_draws, M, M))
            
            # Create meshgrid for the M x M coordinates
            i, j = np.meshgrid(range(M), range(M), indexing='ij')
            
            # Use advanced indexing to select random pixels
            map = map_3d[random_indices, i, j]

            if density_map:
                #print(weight_by_band)
                map = self.map_to_density_map(map, redshift = self.redshift, logmap = logmap, weight_by_band = weight_by_band)
            else:
                if logmap:
                    map = np.log10(map)

            new_map = map


        else:
            n_draws = map_3d.shape[0]
            if density_map:
                temp_map = np.zeros((n_draws, map_3d.shape[1], map_3d.shape[2]))
                for i in range(n_draws):
                    new_map = self.map_to_density_map(map_3d[i], redshift = self.redshift, logmap = logmap, weight_by_band = weight_by_band)
                    if type(new_map) in [u.Quantity, u.Magnitude]:
                        new_map = new_map.value
                    temp_map[i] = new_map
                new_map = temp_map
            else:
                new_map = map_3d

        # Set all 0 values to nan
        new_map[new_map == 0] = np.nan
        new_map[new_map == 0.0] = np.nan

        if scale == 'log':
            scale = LogNorm(vmin=np.nanmin(new_map), vmax=np.nanmax(new_map))
        elif scale == 'linear':
            scale = Normalize(vmin=np.nanmin(new_map), vmax=np.nanmax(new_map))
        else:
            raise Exception('Scale must be log or linear')

        fig, ax = plt.subplots(1, 1, figsize=(4, 4), facecolor=facecolor, layout='tight')

          # Create the image plot once
        im = ax.imshow(new_map[0], origin='lower', interpolation='none', cmap=cmap, norm=scale, animated=True)
        cax = make_axes_locatable(ax).append_axes('top', size='5%', pad='2%')
        cbar = fig.colorbar(im, cax=cax, orientation='horizontal')
        cbar.ax.xaxis.set_ticks_position('top')

        if cbar_label is not None:
            cbar.set_label(cbar_label, labelpad=10, fontsize=8)
            cbar.ax.xaxis.set_label_position('top')

        
        def animate(i):
            im.set_array(new_map[i])
            return [im]
        '''
        def animate(i):
            ax.clear()
            ax.imshow(new_map[i], origin='lower', interpolation='none', cmap=cmap, norm=scale)
        '''
        ani = FuncAnimation(fig, animate, frames=n_draws, interval=100, repeat=True, blit=True)

        # Stop colorbar getting cut off


        if save:
            ani.save('galaxies/animation.gif', writer='pillow', fps=60)
        
        plt.close(fig)

        if html:
            return ani.to_jshtml()

        return ani

    def scale_map_by_band(self, map, band):
        # Unused
        pix_map_band = self.self.psf_matched_data[binmap_type][band]

        bins = np.unique(self.pixedfit_map)
        for bin in bins:
            mask = self.pixedfit_map == bin
            map[mask] = map[mask] * pix_map_band[mask]
        

    def plot_bagpipes_sed(self, run_name, run_dir = f'pipes/', bins_to_show='all', plotpipes_dir = 'pipes_scripts/'):
        if not hasattr(self, 'sed_fitting_table') or 'bagpipes' not in self.sed_fitting_table.keys():
            raise Exception("Need to run bagpipes first")

        table = self.sed_fitting_table['bagpipes'][run_name]
    
        
        # Plot map next to SED plot

        fig = plt.figure(constrained_layout=True, figsize=(8, 4))
        gs = fig.add_gridspec(2, 3)
        ax_map = fig.add_subplot(gs[:, 0])
        ax_sed = fig.add_subplot(gs[:, 1:])

        ax_divider = make_axes_locatable(ax_map)
        cax = ax_divider.append_axes('top', size='5%', pad='2%')
        
        # Make cmap for map
        count = []
        other_count = []
        for i, id in enumerate(table['#ID']):
            try:
                id = float(id)
                count.append(i)
            except:
                other_count.append(i)
                pass

        cmap = plt.cm.get_cmap('cmr.cosmic', len(count))
        color = {table['#ID'][i]:cmap(pos) for pos, i in enumerate(count)}
        entire_cmap = plt.cm.get_cmap('cmr.pepper', len(other_count))
        color.update({table['#ID'][i]:entire_cmap(pos) for pos, i in enumerate(other_count)})
        map = copy.copy(self.pixedfit_map)
        map[map == 0] = np.nan
        
        mappable = ax_map.imshow(map, origin='lower', interpolation='none', cmap='cmr.cosmic')
        cbar = fig.colorbar(mappable, cax=cax, orientation='horizontal')
        cbar.ax.xaxis.set_ticks_position('top')
        cbar.ax.xaxis.set_label_position('top')
        cbar.ax.tick_params(labelsize=8)
        cbar.ax.xaxis.set_major_formatter(ScalarFormatter())

        
        if bins_to_show == 'all':
            bins_to_show = table['#ID']
        else:
            bins_to_show = bins_to_show

        for pos, bin in enumerate(bins_to_show):
            pipes_obj = self.load_pipes_object(run_name, bin, run_dir = run_dir, cache=None, plotpipes_dir=plotpipes_dir)

            # This plots the observed SED
            #pipes_obj.plot_sed(ax=ax_sed, colour=color[bin], wav_units=u.um, flux_units=u.ABmag, x_ticks=None, zorder=4, ptsize=40,
            #                y_scale=None, lw=1., skip_no_obs=False, fcolour='blue',
            #                label=None,  marker="o", rerun_fluxes=False)
            # This plots the best fit SED
            #print(color)
            pipes_obj.plot_best_fit(ax_sed, colour=color[str(bin)],  wav_units=u.um, flux_units=u.ABmag, lw=1, fill_uncertainty=False, zorder=5, linestyle = '-.' if pos in other_count else 'solid', label = bin if pos in other_count else "")
            # Plot photometry
            pipes_obj.plot_sed(ax_sed, colour=color[str(bin)], wav_units=u.um, flux_units=u.ABmag, zorder = 6, fcolour=color[str(bin)], ptsize=15)
            try:
                float(bin)
                # find CoM of pixels in map
                y, x = np.where(map == bin)
                y = np.mean(y)
                x = np.mean(x)
                ax_map.text(x, y, bin, fontsize=8, color='black', path_effects=[PathEffects.withStroke(linewidth=1, foreground='white')])
            except ValueError:
                pass
        # Set x-axis limits
        ax_sed.set_xlim(0.5, 5)
        # Set y-axis limits
        ax_sed.set_ylim(31, 25)

        # Set x-axis label
        ax_sed.set_xlabel('Wavelength ($\mu$m)')
        # Set y-axis label
        ax_sed.set_ylabel('AB Mag')
        ax_sed.legend(fontsize=6,loc='upper left')
    
    def init_galfind_phot(self, inst = 'ACS_WFC+NIRCam', psf_type = 'star_stack', binmap_type = 'pixedfit'):

        from galfind import Photometry_obs, Photometry, Photometry_rest, Instrument, Combined_Instrument, PDF

        if not hasattr(self, 'photometry_table'):
            raise Exception("Need to run measure_flux_in_bins first")
        if hasattr(self, 'use_psf_type'):
            psf_type = self.use_psf_type
            print(f'Using PSF type {psf_type}')
        else:
            print(f'Using PSF type from argument {psf_type}')
        
        if hasattr(self, 'use_binmap_type'):
            binmap_type = self.use_binmap_type
            print(f'Using binmap type {binmap_type}')
        else:
            print(f'Using binmap type from argument {binmap_type}')
        
        if psf_type not in self.photometry_table.keys():
            raise ValueError(f'PSF type {psf_type} not found in photometry table')
        if binmap_type not in self.photometry_table[psf_type].keys():
            raise ValueError(f'Binmap type {binmap_type} not found in photometry table')
        

        table = self.photometry_table[psf_type][binmap_type]
        self.galfind_photometry_rest = {}

        start_instrument = Combined_Instrument.from_name(inst)

        for row in tqdm(table):
            flux_Jy, flux_Jy_errs = [], []
            instrument = copy.deepcopy(start_instrument)
            for band in instrument.band_names:
                if band in self.bands:
                    flux = row[band].to(u.Jy).value
                    flux_err = row[f'{band}_err'].to(u.Jy).value
                    flux_Jy.append(flux)
                    flux_Jy_errs.append(flux_err)
                else:
                    instrument.remove_band(band)
            
            flux_Jy = np.array(flux_Jy) * u.Jy
            flux_Jy_errs = np.array(flux_Jy_errs) * u.Jy
            self.galfind_photometry_rest[str(row['ID'])] = Photometry_rest(instrument, flux_Jy, flux_Jy_errs, depths = np.ones(len(flux_Jy)), z = self.redshift)
        print('Finished building galfind photometry')


    def galfind_phot_property_map(self, property, iters = 100, load_in = True, density = False, logmap = False, plot = True, ax = None, facecolor = 'white', cmap='viridis', **kwargs):
        '''
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
        '''
        if not hasattr(self, 'galfind_photometry_rest'):
            print('Warning: galfind photometry not initialized, initializing with default values')
            self.init_galfind_phot()

        from galfind import Photometry_obs, Photometry, Photometry_rest, Instrument, Combined_Instrument, PDF


        required_kwargs = {'beta_phot':['rest_UV_wav_lims',  'iters'],
                            'AUV_from_beta_phot':['rest_UV_wav_lims', 'dust_author_year',  'iters', 'ref_wav'],
                            'mUV_phot':['rest_UV_wav_lims', 'ref_wav', 'iters'],
                            'MUV_phot':['rest_UV_wav_lims', 'ref_wav', 'iters'],
                            'LUV_phot':['rest_UV_wav_lims', 'ref_wav', 'iters', 'frame'],
                            'SFR_UV_phot':['rest_UV_wav_lims', 'ref_wav', 'dust_author_year', 'kappa_UV_conv_author_year', 'frame',  'iters'],
                            'fesc_from_beta_phot':['rest_UV_wav_lims', 'conv_author_year', 'iters'],
                            'EW_rest_optical':['strong_line_names'],
                            'line_flux_rest_optical':['strong_line_names', 'iters'],
                            'xi_ion':['iters'],
                            }
        map = copy.deepcopy(self.pixedfit_map)
        map[map == 0] = np.nan
        #PDFs = []

        kwargs['iters'] = iters

        test_id = np.unique(map)[1]
        func = getattr(self.galfind_photometry_rest[str(int(test_id))],  f'calc_{property}')
        
        #arguments = signature(func).parameters
        #func_args = arguments['args']
        #func_kwargs = arguments['kwargs']
        #print(func_args, func_kwargs)
        #print(func_args.default, func_kwargs.default)
        #print(func)
        
        
        # match kwargs to arguments
        #for arg in arguments:
        #    if arg not in kwargs.keys():
        #        raise ValueError(f'Missing required argument {arg} for {property}')

        used_kwargs = {arg:kwargs[arg] for arg in kwargs if arg in required_kwargs[property]}
        
        property_name = func(extract_property_name = True, **used_kwargs)     
        num_pdfs = len(property_name)
        PDFs = [[] for i in range(num_pdfs)]       

        for pos, id in enumerate(tqdm(np.unique(map))):
            if str(id) in ['0', 'nan']:
                value = np.nan
                continue
            
            phot = copy.deepcopy(self.galfind_photometry_rest[str(int(id))])
            loaded = False
            for prop_name in property_name:
                #print('checking', f'{prop_name}')

                if hasattr(self, f'{prop_name}'):
                    skip = False
                    ppdf = getattr(self, f'{prop_name}')[pos]
                    #print(ppdf.unit)
                    
                    if not skip and load_in:
                        #print('Updating property PDFs from .h5')
                        #pdf = phot.property_PDFs[property_name][pos]
                        #pdf = getattr(self, f'{property_name}_PDFs')[pos]
                        
                        saved_kwargs = getattr(self, f'{prop_name}_kwargs', {})
                        
                        phot.property_PDFs[prop_name] = PDF.from_1D_arr(prop_name, ppdf, kwargs = saved_kwargs)
                        
                        phot._update_properties_from_PDF(prop_name)
                        loaded = True
                        if len(ppdf) != iters:
                        # print(np.shape(pdf), [iters, len(np.unique(map))])
                            print(f'PDFs for {prop_name} do not match shape of map, recalculating, {len(ppdf)} != {iters}')
                            loaded = False

            func_name = f'calc_{property}'
            if hasattr(phot, func_name):
                func = getattr(phot, func_name)
                _, param_names = phot._calc_property(SED_rest_property_function = func, **used_kwargs)
                
                #if type(param_name) in [tuple, list]:
                #    param_name = param_name
            else:
                raise ValueError(f'Function calc_{property} not found in galfind photometry object')
            
            for pos_param, param_name in enumerate(param_names):
                print(phot.properties.keys())
                value = phot.properties[param_name]
                #print(param_name, value)
                if type(value) in [u.Quantity, u.Magnitude]:
                    unit = value.unit
                
            
                if phot.property_PDFs[param_name] is not None:
                    out_kwargs = phot.property_PDFs[param_name].kwargs
                    PDFs[pos_param].append(phot.property_PDFs[param_name].input_arr)

                else:
                    out_kwargs = {}
                    PDFs[pos_param].append([])

            
                if pos_param == len(param_names) - 1:
                    pname = param_name
                    map[map == id] = value
        if density:
            map = self.map_to_density_map(map, redshift = self.redshift, logmap = logmap)
            
            unit = unit/u.kpc**2
    
        label = f'{pname} ({unit:latex})'
        
        if not loaded:
            for pos, param_name in enumerate(param_names):
                unit = phot.properties[param_name].unit
                out_kwargs['unit'] = unit
                all_PDFs = np.array(PDFs[pos]) * unit
                out_kwargs = phot.property_PDFs[param_name].kwargs
                self.add_to_h5(all_PDFs, 'photometry_properties', param_name, setattr_gal = f'{param_name}', overwrite=True, meta = out_kwargs, setattr_gal_meta = f'{param_name}_kwargs')

        #def add_to_h5(self, data, group, name, ext=0, setattr_gal=None, overwrite=False, meta=None):

        if np.all(np.isnan(map)):
            print(f'Calculation not possible for {param_name}')
            return None, None

        if plot:
            if ax is not None:
                fig = ax.get_figure()
            else:
                fig, ax = plt.subplots(1, 1, figsize=(5, 5), constrained_layout=True, facecolor=facecolor)
            
            ax_divider = make_axes_locatable(ax)
            cax = ax_divider.append_axes('top', size='5%', pad='2%')

            i = ax.imshow(map, origin='lower', interpolation='none', cmap=cmap)
            cbar = fig.colorbar(i, cax=cax, orientation='horizontal')
            # Label top of axis
            cbar.set_label(label, labelpad = 10, fontsize=10)
            cbar.ax.xaxis.set_ticks_position('top')
            cbar.ax.xaxis.set_label_position('top')
            cbar.ax.tick_params(labelsize=8)
            cbar.ax.xaxis.set_major_formatter(ScalarFormatter())
            if property == 'EW_rest_optical':
                line = param_name.split('_')[2]

                line_band = out_kwargs[f'{line}_emission_band']
                cont_band = out_kwargs[f'{line}_cont_band']
                ax.text(0.05, 0.95, f'{line_band} - {cont_band}', transform=ax.transAxes, fontsize=10, horizontalalignment = 'left', verticalalignment='top', color='black')
            ax.set_title(property)
            return fig
        
        return map, out_kwargs

    # calc_beta_phot(
    @property
    def available_em_lines(self):
        from galfind import Emission_lines
        return Emission_lines.strong_optical_lines #Emission_lines.line_diagnostics.keys()

    def plot_ew_figure(self, save = False, facecolor = 'white', max_col = 5, **kwargs):

        to_plot = {}
        for em_line in self.available_em_lines:
            map, out_kwargs = self.galfind_phot_property_map(f'EW_rest_optical', plot = False, facecolor = facecolor, strong_line_names = [em_line], rest_optical_wavs =[3_700., 10_000.] * u.AA, **kwargs)
            if type(map) != type(None):
                to_plot[em_line] = (map, out_kwargs)
        
        num_rows = len(to_plot) // max_col + 1
        fig, axes = plt.subplots(num_rows, max_col, figsize=(2.5*max_col, 2.5*num_rows), constrained_layout=True, facecolor=facecolor, sharex=True, sharey=True)
        fig.get_layout_engine().set( h_pad=4 / 72, hspace=0.2)

        axes = axes.flatten()
        for pos, line in enumerate(to_plot.keys()):
            cax = make_axes_locatable(axes[pos]).append_axes('top', size='5%', pad='2%')
            map, out_kwargs = to_plot[line]
            i = axes[pos].imshow(map, origin='lower', interpolation='none', cmap='viridis')
            cbar = fig.colorbar(i, cax=cax, orientation='horizontal')
            line_band = out_kwargs[f'{line}_emission_band']
            cont_band = out_kwargs[f'{line}_cont_band']
            axes[pos].text(0.05, 0.95, f'{line_band} - {cont_band}', transform=axes[pos].transAxes, fontsize=10, horizontalalignment = 'left', verticalalignment='top', color='black')
            cbar.set_label(line, labelpad = 10)
            cbar.ax.xaxis.set_ticks_position('top')
            cbar.ax.xaxis.set_label_position('top')
            cbar.ax.tick_params(labelsize=8)
            cbar.ax.xaxis.set_major_formatter(ScalarFormatter())
            #axes[pos].set_title(line)
        # Remove empty axes
        for i in range(len(to_plot), len(axes)):
            fig.delaxes(axes[i])
        return fig


class MockResolvedGalaxy(ResolvedGalaxy):
    def __init__(self, galaxy_id, survey, version, instruments = ['ACS_WFC', 'NIRCam'], 
                            excl_bands = [], cutout_size = 64, forced_phot_band = ["F277W", "F356W", "F444W"], 
                            aper_diams = [0.32] * u.arcsec, output_flux_unit = u.uJy, h5folder = 'galaxies/',
                            dont_psf_match_bands=[], already_psf_matched = False):
        

        
        super().__init__(galaxy_id, survey, version, instruments = instruments, 
                            excl_bands = excl_bands, cutout_size=cutout_size, forced_phot_band = forced_phot_band, 
                            aper_diams = aper_diams, output_flux_unit = output_flux_unit, h5folder = h5folder,
                            dont_psf_match_bands=dont_psf_match_bands, already_psf_matched = already_psf_matched)

    @classmethod
    def init_mock_from_synthesizer(cls, redshift_index, galaxy_index, cutout_size = 65,
                                    mock_survey = 'JOF', mock_version = 'v11', 
                                    mock_instrument = 'ACS_WFC+NIRCam', 
                                    mock_forced_phot_band = ["F277W", "F356W", "F444W"],
                                    mock_aper_diams = [0.32] * u.arcsec,
                                    psf_type = '',
                                    grid_name = "bpass-2.2.1-bin_chabrier03-0.1,300.0_cloudy-c17.03",
                                    grid_dir = "./grids/"):
    
        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
            from galfind import Catalogue, EAZY
            from galfind.Catalogue_Creator import GALFIND_Catalogue_Creator

        SED_code_arr = [EAZY()]
        SED_fit_params_arr = make_EAZY_SED_fit_params_arr(SED_code_arr, templates_arr, lowz_zmax_arr)
        # Make cat creator
        cat_creator = GALFIND_Catalogue_Creator("loc_depth", aper_diams[0], 10)
        # Load catalogue and populate galaxies
        cat = Catalogue.from_pipeline(survey = mock_survey, version = mock_version, instruments = mock_instrument,
            aper_diams = mock_aper_diams, cat_creator = cat_creator, SED_fit_params_arr = SED_fit_params_arr,
            forced_phot_band = mock_forced_phot_band, excl_bands = [], loc_depth_min_flux_pc_errs = [10])

        im_paths = cat.data.im_paths
        im_exts = cat.data.im_exts
        err_paths = cat.data.rms_err_paths
        err_exts = cat.data.rms_err_exts
        seg_paths = cat.data.seg_paths
        im_zps = cat.data.im_zps
        im_pixel_scales = cat.data.im_pixel_scales
        bands = galaxy.phot.instrument.band_names # should be bands just for galaxy!
        translate = {'NIRCam':'JWST/NIRCam', 'ACS_WFC':'HST/ACS_WFC'}
        instrument = [translate(cat.data.instrument.instrument_from_band(band).name) for band in bands]

        
        try:
            from scipy import signal
            from synthesizer.emission_models import IncidentEmission
            from synthesizer.filters import FilterCollection as Filters
            from synthesizer.grid import Grid
            from synthesizer.kernel_functions import Kernel
            from synthesizer.particle.galaxy import Galaxy
            from unyt import Hz, erg, Mpc, kpc, Msun, yr, s, arcsecond, nJy
            from synthesizer.emission_models.attenuation.igm import Inoue14, Madau96
        except ImportError:
            raise Exception('Synthesizer not installed. Please clone from Github and install using pip install .')

        grid = Grid(grid_name, grid_dir=grid_dir)

        tag = tags[redshift_index]
        region = regions[redshift_index][galaxy_index]
        id = ids[redshift_index][galaxy_index] - 1

        zed = float(tag[5:].replace("p", "."))

        with h5py.File('./flares_flags_balmer_project.hdf5','r') as hf:  #opening the hdf5 file
            # coordinates of the stellar particles 
            coordinates = np.array(hf[f"{tag}/{region}/{id}"].get('coordinates'), dtype=np.float64) * Mpc
            # initial masses of the stellar particles
            initial_masses = np.array(hf[f"{tag}/{region}/{id}"].get('initial_masses'), dtype=np.float64) * Msun
            # current masses of the stellar particles, mass change due to mass loss as stars age
            current_masses = np.array(hf[f"{tag}/{region}/{id}"].get('initial_masses'), dtype=np.float64) * Msun
            # ages of stars in log10, which is in units of years
            log10ages = np.array(hf[f"{tag}/{region}/{id}"].get('log10ages'), dtype=np.float64)
            # metallicities of the stars in log10
            log10metallicities = np.array(hf[f"{tag}/{region}/{id}"].get('log10metallicities'), dtype=np.float64)
            # optical depth in the V-band, it is the same prescription as shown in Vijayan+21, which
            # includes the V-band dust optical depth due to the diffuse ISM dust and birth cloud dust
            tau_v = np.array(hf[f"{tag}/{region}/{id}"].get('tau_v'), dtype=np.float64)
            # the smoothing lengths of star particles
            smoothing_lengths = np.array(hf[f"{tag}/{region}/{id}"].get('smoothing_lengths'), dtype=np.float64) * Mpc

        gal = Galaxy(redshift=zed)
        gal.load_stars(
            initial_masses=initial_masses,
            ages=10**log10ages * yr,
            metallicities=10**log10metallicities,
            coordinates=coordinates,
            current_masses=current_masses,
            smoothing_lengths=smoothing_lengths,
            centre=np.zeros(3)*Mpc
        )
        # Unsure about this
        gal.tau_v = tau_v

        # Get incident emission
        model = IncidentEmission(grid)
        # Generate incident particle spectra from the galaxy
        sed = gal.stars.get_particle_spectra(model)

        # Calculate the observed SED in nJy, applying Inoue14 IGM absorption
        sed.get_fnu(cosmo, gal.redshift, igm=Inoue14)

        # Need to get bands from GALFIND
        filter_codes = [f'{obsevatory_instrument}.{f}' for f in (bands, instruments)]

        filters = Filters(filter_codes, new_lam=grid.lam)

        fluxes = gal.stars.particle_spectra["incident"].get_photo_fnu(filters)

        re = 1
        d_A = cosmo.angular_diameter_distance(gal.redshift)
        pix_scal = u.pixel_scale(0.03*u.arcsec/u.pixel)
        re_as = (re * u.pixel).to(u.arcsec, pix_scal)
        re_kpc = (re_as * d_A).to(u.kpc, u.dimensionless_angles())

        # Getting the resolution and width of the cutout in kpc (at the redshift of the galaxy)
        resolution = re_kpc.value * kpc
        width = cutout_size * resolution

        # Wavelength and SED 
        wav = sed.obslam
        phot =  -2.5* np.log10(np.sum(sed.fnu, axis=0)) + 31.40
        
        # Wavelength of filters and photometry 
        wav_phot = filters.pivot_lams.ndarray_view()
        flux_phot = -2.5*np.log10(np.sum(sed.photo_fnu.photo_fnu.to(nJy), axis=1)) + 31.40
     
        sph_kernel = Kernel()
        kernel_data = sph_kernel.get_kernel()

        # Get the image - histogram
        hist_imgs = gal.get_images_flux(
            resolution,
            fov=width,
            img_type="hist",
            stellar_photometry="incident",
            blackhole_photometry=None,
        )

        # Get the image - smoothed by SPH kernel
        smooth_imgs = gal.get_images_flux(
            resolution,
            fov=width,
            img_type="smoothed",
            stellar_photometry="incident",
            blackhole_photometry=None,
            kernel=kernel_data,
            kernel_threshold=1,
        )
        # Data is of form smooth_imgs[fcode].arr

        # Then we need PSFs

        # Apply the PSFs
        psf_imgs = smooth_imgs.apply_psfs(psfs)

        # Then we need to apply a noise model
        # Maybe place the galaxy in a blank region of the image in each band.

        # How to approximate WHT and RMS error maps?
        # 

        # Then we need to get segmentation maps

        # Use sep.extract to get segmentation maps and measure equivalent fluxes. 

        # Would also be good to get the property maps



class MultipleResolvedGalaxy:
    def __init__(self, list_of_galaxies):
        self.galaxies = list_of_galaxies

    @classmethod
    def init(cls, list_of_ids, survey, version, **kwargs):
        list_of_galaxies = []
        for id in list_of_ids:
            if 'mock' in survey:
                galaxy = MockResolvedGalaxy(id, survey, version, **kwargs)
            else:
                galaxy = ResolvedGalaxy(id, survey, version, **kwargs)
            list_of_galaxies.append(galaxy)

        return cls(list_of_galaxies)

    def __getitem__(self, key):
        return self.galaxies[key]
    
    def __len__(self):
        return len(self.galaxies)
    
    def __iter__(self):
        return iter(self.galaxies)

    def __next__(self):
        return next(self.galaxies)

    def total_number_of_bins(self):
        return sum([galaxy.get_number_of_bins() for galaxy in self.galaxies])

    def scatter_plot(self, x_attr, y_attr, c_attr = None, facecolor = 'white', cmap = 'viridis', **kwargs):

        fig, ax = plt.subplots(1, 1, figsize=(5, 5), facecolor=facecolor)
        
        if x_label in kwargs:
            ax.set_xlabel(kwargs[x_label])
            kwargs.pop(x_label)
        
        if y_label in kwargs:
            ax.set_ylabel(kwargs[y_label])
            kwargs.pop(y_label)
        
        if c_label in kwargs:
            cbar_label = kwargs[c_label]
            kwargs.pop(c_label)
        else:
            cbar_label = None
        
        if c_attr is not None:
            cbar = make_axes_locatable(ax).append_axes('top', size='5%', pad='2%')


        for galaxy in self.galaxies:
            x = getattr(galaxy, x_attr)
            y = getattr(galaxy, y_attr)
            if c_attr is not None:
                c = getattr(galaxy, c_attr)
                assert len(x) == len(y) == len(c), "x, y and c must have same length"
            else:
                assert len(x) == len(y), "x and y must have same length"

            ax.scatter(x, y, c=c, cmap=cmap, **kwargs)

        return fig

    
    # Run any function on all galaxies
    def run_function(self, function, *args, **kwargs):
        for galaxy in self.galaxies:
            if type(function) == str:
                function = getattr(galaxy, function)
            function(*args, **kwargs)


if __name__ == "__main__":
    if computer == 'morgan':
        catalog_path_selected = '/nvme/scratch/work/tharvey/resolved_sedfitting/catalogs/JOF_psfmatched_MASTER_Sel-F277W+F356W+F444W_v11_total_selected.fits'
        cat_selected = Table.read(catalog_path_selected)
        ids = cat_selected['NUMBER']
        cutout_size = cat_selected['CUTOUT_SIZE'][:10]
        ids = ids[:10] # For testing
        overwrite = False
        h5folder = 'galaxies/'
    elif computer == 'singularity':
        ids = [16, 36, 370, 478, 531, 575, 778, 801, 806, 830]
        overwrite = False
        h5folder = '/mnt/galaxies/'
        cutout_size = None # Not used when loading from h5

    galaxies = ResolvedGalaxy.init(list(ids), 'JOF_psfmatched', 'v11', already_psf_matched = True, cutout_size = cutout_size, h5folder = h5folder)
    # Should speed it up?
    
    # Initial model for photo-z 
    sfh = {} # double-power-law
    '''
    sfh_type = 'dblplaw'
                   
    sfh["tau"] = (0., 15.)                # Vary the time of peak star-formation between
                                            # the Big Bang at 0 Gyr and 15 Gyr later. In 
                                            # practice the code automatically stops this
                                            # exceeding the age of the universe at the 
                                            # observed redshift.

    sfh["tau_prior"] = 'uniform'         # Impose a prior which is uniform in log_10 of the
    if sfh["tau"][0] == 0 and sfh["tau_prior"] == 'log_10':
        sfh["tau"][0] = 0.01

    sfh["alpha"] = (0.01, 1000.)          # Vary the falling power law slope from 0.01 to 1000.
    sfh["beta"] = (0.01, 1000.)           # Vary the rising power law slope from 0.01 to 1000.
    sfh["alpha_prior"] = "log_10"         # Impose a prior which is uniform in log_10 of the 
    sfh["beta_prior"] = "log_10"          # parameter between the limits which have been set 
                                            # above as in Carnall et al. (2017).
    sfh["massformed"] = (5., 12.)
    sfh['metallicity'] = (0, 3)
    '''
    sfh_type = 'delayed'
    sfh["tau"] = (0.01, 15) # `Gyr`
	sfh["massformed"] = (5., 12.)   # Log_10 total stellar mass formed: M_Solar
	
	sfh["age"] = (0.001, 15) # Gyr
	sfh['age_prior'] = 'uniform'
	sfh['metallicity_prior'] = 'uniform'
	if metallicity_prior == 'log_10':
		sfh["metallicity"] = (1e-03, 3)
	elif metallicity_prior == 'uniform':
		sfh['metallicity'] = (0, 3)

    
    dust = {}
    dust["type"] = "Calzetti"
    dust["Av"] = (0, 5.0)

    nebular = {}
    nebular["logU"] = (-3.0, -1.0)

    fit_instructions = {"t_bc":0.01,
                    sfh_type:sfh,
                    "nebular":nebular,
                    "dust":dust}

    meta = {'run_name':'photoz_delayed', 'redshift':'eazy', 'redshift_sigma':'eazy',
            'min_redshift_sigma':0.5, 'fit_photometry':'TOTAL_BIN+MAG_APER_TOTAL',
            'sampler':'multinest'}

    overall_dict = {'meta': meta, 'fit_instructions': fit_instructions}

    dicts = [copy.deepcopy(overall_dict) for i in range(len(galaxies))]

    resolved_sfh = {
    'age_max': (0.01, 2.5), # Gyr 
    'age_min': (0, 2.5), # Gyr
    'metallicity': (1e-3, 2.5), # solar
    'massformed': (4, 12), # log mstar/msun
    }

    dust = {}
    dust["type"] = "Calzetti"
    dust["Av"] = (0, 5.0)

    fit_instructions = {"t_bc":0.01,
                    "constant":sfh,
                    "nebular":nebular,
                    "dust":dust,  
                    }
    # This means that we are fixing the photo-z to the results from the 'photoz_DPL' run,
    # specifically the 'MAG_APER_TOTAL' photometry
    # We are fitting only the resolved photometry in the 'TOTAL_BIN' bins
    meta = {'run_name':'CNST_SFH_RESOLVED', 'redshift':'photoz_delayed', 'redshift_id':'MAG_APER_TOTAL',
            'fit_photometry':'bin'}
    resolved_dict = {'meta': meta, 'fit_instructions': fit_instructions}
    resolved_dicts = [copy.deepcopy(resolved_dict) for i in range(len(galaxies))]


    cat = None
    num_of_bins = 0

    if rank == 0:
        for posi, galaxy in enumerate(galaxies):

            # Add original imaging back
            print('Adding original imaging.')

            cat = galaxy.add_original_data(cat = cat, return_cat = True, overwrite = overwrite, crop_by = None)
            # Add total fluxes
            galaxy.add_flux_aper_total(catalogue_path = '/raid/scratch/work/austind/GALFIND_WORK/Catalogues/v11/ACS_WFC+NIRCam/JOF_psfmatched/JOF_psfmatched_MASTER_Sel-F277W+F356W+F444W_v11_total.fits', overwrite = overwrite)

            # Add Detection data
            galaxy.add_detection_data(overwrite = overwrite)

            # Plot segmentation stamps
            fig = galaxy.plot_seg_stamps()
            fig.savefig(f'{h5folder}/diagnostic_plots/{galaxy.galaxy_id}_seg_stamps.png', dpi=300, bbox_inches='tight')
            plt.close()
            # Currently set to use segmentation map from detection image. May change this in future. 
            if galaxy.gal_region is None or overwrite:
                galaxy.pixedfit_processing(gal_region_use = 'detection', overwrite = overwrite) # Maybe seg map should be from detection image?

                fig = galaxy.plot_gal_region()
                fig.savefig(f'{h5folder}/diagnostic_plots/{galaxy.galaxy_id}_gal_region.png', dpi=300, bbox_inches='tight')
                plt.close()
            if galaxy.pixedfit_map is None or overwrite:
                galaxy.pixedfit_binning(overwrite = overwrite)

            if galaxy.photometry_table is None or overwrite:
                galaxy.measure_flux_in_bins(overwrite = overwrite)

            num_of_bins += galaxy.get_number_of_bins()

        print(f'Total number of bins to fit: {num_of_bins}')
        # Run Bagpipes in parallel

    if size > 1:
        # This option for running by with mpirun/mpiexec
        n_jobs = 1
    else:
        if computer == 'morgan':
            n_jobs = 6
            backend = 'loky'
        elif computer == 'singularity':
            n_jobs = np.min([len(galaxies)+1, 6])
            backend = 'multiprocessing'
        
    if n_jobs == 1:
        for i in range(len(galaxies)):
            galaxies[i].run_bagpipes(dicts[i])
    else:
        # This options runs multiple galaxies in parallel with joblib
        if computer == 'singularity':
            from joblib import parallel_config
            with parallel_config(backend=backend, n_jobs=n_jobs):
                Parallel()(delayed(galaxies[i].run_bagpipes)(dicts[i]) for i in range(len(galaxies)))
        else:
            Parallel(n_jobs=n_jobs)(delayed(galaxies[i].run_bagpipes)(dicts[i]) for i in range(len(galaxies)))
        


        
        




    # Test the Galaxy class
    #galaxy = ResolvedGalaxy.init_from_galfind(645, 'NGDEEP2', 'v11', excl_bands = ['F435W', 'F775W', 'F850LP'])
    #galaxy2 = ResolvedGalaxy.init_from_h5('NGDEEP2_645')

    # Simple test Bagpipes fit_instructions
    '''
    sfh = {
        'age_max': (0.03, 0.5), # Gyr 
        'age_min': (0, 0.5), # Gyr
        'metallicity': (1e-3, 2.5), # solar
        'massformed': (4, 12), # log mstar/msun
        }

    nebular = {}
    nebular["logU"] = -2.0 

    dust = {}
    dust["type"] = "Calzetti"
    dust["Av"] = (0, 5.0)

    fit_instructions = {"t_bc":0.01,
                    "constant":sfh,
                    "nebular":nebular,
                    "dust":dust,  
                    }
    meta = {'run_name':'initial_test_cnst_sfh', 'redshift':0.48466974}

    overall_dict = {'meta': meta, 'fit_instructions': fit_instructions}

    galaxy2.run_bagpipes(overall_dict)

    galaxy2.plot_bagpipes_results('initial_test_cnst_sfh')
    #galaxy2.pixedfit_processing()
    
    '''
    
          
'''
TODO:
    1. Don't run Bagpipes if already run
    2. Convert SFR, stellar mass to surface density (/kpc^2)
    3. Plotting SEDs, PDFs, Corner plots?

'''

