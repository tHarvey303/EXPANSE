from astropy.io import fits
from astropy.io.fits import Header
import numpy as np
import matplotlib.pyplot as plt
import h5py as h5
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.nddata import Cutout2D
import ast
from astropy.convolution import convolve_fft
import glob
import shutil
from pathlib import Path
import os
import typing
from astropy.utils.exceptions import AstropyWarning
import warnings
from astropy.utils.masked import Masked
from astropy.wcs import WCS
# can write astropy to h5

from piXedfit.piXedfit_bin import pixel_binning


warnings.simplefilter('ignore', category=AstropyWarning)

# This class is designed to hold all the data for a galaxy, including the cutouts, segmentation maps, and RMS error maps.

'''TODO:
        1. Store aperture photometry, auto photometry from catalogue. 
        2. Pixel binning.
        3. Does ERR map need to be convolved with PSF?
'''


class ResolvedGalaxy:
    def __init__(self, galaxy_id : int, 
                sky_coord : SkyCoord,
                survey : str, 
                bands, im_paths, im_exts, im_zps, seg_paths,
                rms_err_paths, rms_err_exts, im_pixel_scales, phot_imgs, phot_pix_unit, 
                phot_img_headers, rms_err_imgs, seg_imgs, aperture_dict, 
                psf_matched_data = None, psf_matched_rms_err = None, pixedfit_map = None, voronoi_map = None, 
                binned_flux_map = None, binned_flux_err_map = None, cutout_size=64, 
                h5_folder = '/nvme/scratch/work/tharvey/resolved_sedfitting/galaxies/',
                psf_kernel_folder = '/nvme/scratch/work/tharvey/resolved_sedfitting/psfs/', 
                psf_type = 'webbpsf', overwrite = False):

        self.galaxy_id = galaxy_id
        self.sky_coord = sky_coord
        self.survey = survey
        self.bands = bands
        self.im_paths = im_paths
        self.im_exts = im_exts
        self.im_zps = im_zps
        self.seg_paths = seg_paths
        self.rms_err_paths = rms_err_paths
        self.rms_err_exts = rms_err_exts
        self.im_pixel_scales = im_pixel_scales
        self.cutout_size = cutout_size
        self.aperture_dict = aperture_dict

        # Actual cutouts
        self.phot_imgs = phot_imgs
        self.phot_pix_unit = phot_pix_unit
        self.phot_img_headers = phot_img_headers
        self.rms_err_imgs = rms_err_imgs
        self.seg_imgs = seg_imgs
        
        self.h5_path = h5_folder + f'{self.galaxy_id}.h5'

        if os.path.exists(self.h5_path) and overwrite:
            os.remove(self.h5_path)
        
        # Check sizes of images
        for band in self.bands:
            assert self.phot_imgs[band].shape == (self.cutout_size, self.cutout_size), f"Image shape for {band} is {self.phot_imgs[band].shape}, not {(self.cutout_size, self.cutout_size)}"
            assert self.rms_err_imgs[band].shape == (self.cutout_size, self.cutout_size), f"RMS error image shape for {band} is {self.rms_err_imgs[band].shape}, not {(self.cutout_size, self.cutout_size)}"
            assert self.seg_imgs[band].shape == (self.cutout_size, self.cutout_size), f"Segmentation map shape for {band} is {self.seg_imgs[band].shape}, not {(self.cutout_size, self.cutout_size)}"
        # Bin the pixels
        self.voronoi_map = voronoi_map
        self.pixedfit_map = pixedfit_map
        self.psf_matched_data = psf_matched_data
        self.psf_matched_rms_err = psf_matched_rms_err

        self.binned_flux_map = binned_flux_map
        self.binned_flux_err_map = binned_flux_err_map

        self.psf_kernels = {psf_type: {}}
        # Assume bands is in wavelength order, and that the largest PSF is in the last band
        print(f'Assuming {self.bands[-1]} is the band with the largest PSF, and convolving all bands with this PSF kernel.')
        for band in self.bands[:-1]:
            files = glob.glob(f'{psf_kernel_folder}/*{band}*{self.bands[-1]}.fits')
            if len(files) == 0:
                raise Exception(f"No PSF kernel found between {band} and {bands[-1]}")
            elif len(files) > 1:
                raise Exception(f"Multiple PSF kernels found between {band} and {bands[-1]}")
            else:
                self.psf_kernels[psf_type][band] = files[0]
        print(self.psf_matched_rms_err)
        if self.psf_matched_data in [None, {}] or self.psf_matched_rms_err in [None, {}]:  
            print('Convolving images with PSF')
            self.convolve_with_psf(psf_type = psf_type)

        self.dump_to_h5()
         # Save to .h5

    @classmethod
    def init(cls, galaxy_id, survey, version, instruments = ['NIRCam'], 
                            excl_bands = [], cutout_size=64, forced_phot_band = ["F277W", "F356W", "F444W"], 
                            aper_diams = [0.32] * u.arcsec, output_flux_unit = u.uJy, h5folder = '/nvme/scratch/work/tharvey/resolved_sedfitting/galaxies/'):
        if os.path.exists(f'{h5folder}{galaxy_id}.h5'):
            return cls.init_from_h5(galaxy_id, h5_folder = h5folder)
        else:
            return cls.init_from_galfind(galaxy_id, survey, version, instruments = instruments, 
                            excl_bands = excl_bands, cutout_size=cutout_size, forced_phot_band = forced_phot_band, 
                            aper_diams = aper_diams, output_flux_unit = output_flux_unit, h5folder = h5folder)

    @classmethod
    def init_from_galfind(cls, galaxy_id, survey, version, instruments = ['NIRCam'], 
                            excl_bands = [], cutout_size=64, forced_phot_band = ["F277W", "F356W", "F444W"], 
                            aper_diams = [0.32] * u.arcsec, output_flux_unit = u.uJy, h5folder = '/nvme/scratch/work/tharvey/resolved_sedfitting/galaxies/'):
        # Imports here so only used if needed
        from galfind import Data
        from galfind import Catalogue
        from galfind.Catalogue_Creator import GALFIND_Catalogue_Creator

        # Make cat creator
        cat_creator = GALFIND_Catalogue_Creator("loc_depth", aper_diams[0], 10)
        # Load catalogue and populate galaxies
        cat = Catalogue.from_pipeline(survey = survey, version = version, instruments = instruments,
            aper_diams = aper_diams, cat_creator = cat_creator, code_names = [], lowz_zmax = [], 
            forced_phot_band = forced_phot_band, excl_bands = excl_bands, loc_depth_min_flux_pc_errs = [10], 
            templates_arr = [])
        # Make cutouts - this may not work currently as data.wht_types doesn't appear to be defined.
        cat.make_cutouts(galaxy_id, cutout_size = cutout_size)
    
        # Obtain galaxy object
        galaxy = [gal for gal in cat.gals if gal.ID == galaxy_id]
        
        if len(galaxy) == 0:
            raise Exception(f"Galaxy {galaxy_id} not found")
        elif len(galaxy) > 1:
            raise Exception(f"Multiple galaxies with ID {galaxy_id} found")
        else:
            galaxy = galaxy[0]

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
        bands_mask = galaxy.phot.flux_Jy.mask
        bands = bands[~bands_mask]
        # Get aperture photometry
        flux_aper = galaxy.phot.flux_Jy[~bands_mask]
        flux_err_aper = galaxy.phot.flux_Jy_errs[~bands_mask]
        depths = galaxy.phot.depths[~bands_mask]
        # Get the wavelegnths
        wave = galaxy.phot.wav#[~bands_mask]
        # aperture_dict
        aperture_dict = {str(0.32*u.arcsec): {'flux': flux_aper, 'flux_err': flux_err_aper, 'depths': depths, 'wave': wave}}
        
        phot_imgs = {}
        phot_pix_unit = {}
        rms_err_imgs = {}
        seg_imgs = {}
        phot_img_headers = {}

        for band in bands:
            cutout_path = cutout_paths[band]
            hdu = fits.open(cutout_path)
            assert hdu[0].header['SIZE'] == cutout_size # Check cutout size
            data = hdu['SCI'].data
            rms_data = hdu['RMS_ERR'].data
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

        return cls(galaxy_id, galaxy_skycoord, survey, bands, im_paths, im_exts,
                    im_zps, seg_paths, err_paths, err_exts, im_pixel_scales,
                    phot_imgs,phot_pix_unit, phot_img_headers, rms_err_imgs, seg_imgs, aperture_dict, cutout_size, overwrite=True)
    
    @classmethod
    def init_from_h5(cls, galaxy_id, h5_folder = '/nvme/scratch/work/tharvey/resolved_sedfitting/galaxies/'):
        '''Load a galaxy from an .h5 file'''
        h5path = f'{h5_folder}{galaxy_id}.h5'
        hfile = h5.File(h5path, 'r')
        # Load meta data
        galaxy_id = int(hfile['meta']['galaxy_id'][()].decode('utf-8'))
        survey = hfile['meta']['survey'][()].decode('utf-8')
        sky_coord = hfile['meta']['sky_coord'][()].split(b' ')
        sky_coord = SkyCoord(ra=float(sky_coord[0]), dec=float(sky_coord[1]), unit=(u.deg, u.deg))
        bands = ast.literal_eval(hfile['meta']['bands'][()].decode('utf-8'))
        cutout_size = int(hfile['meta']['cutout_size'][()])
        im_zps = ast.literal_eval(hfile['meta']['zps'][()].decode('utf-8'))
        im_pixel_scales = ast.literal_eval(hfile['meta']['pixel_scales'][()].decode('utf-8'))
        im_pixel_scales = {band:u.Quantity(scale) for band, scale in im_pixel_scales.items()}
        phot_pix_unit = ast.literal_eval(hfile['meta']['phot_pix_unit'][()].decode('utf-8'))
        phot_pix_unit = {band:u.Unit(unit) for band, unit in phot_pix_unit.items()}
        
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
        for band in bands:
            phot_imgs[band] = hfile['raw_data'][f'phot_{band}'][()]
            rms_err_imgs[band] = hfile['raw_data'][f'rms_err_{band}'][()]
            seg_imgs[band] = hfile['raw_data'][f'seg_{band}'][()]
            header = hfile['headers'][band][()].decode('utf-8')
            phot_img_headers[band] = header
        
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
        

        hfile.close()

        return cls(galaxy_id, sky_coord, survey, bands, im_paths, im_exts, im_zps,
                    seg_paths, rms_err_paths, rms_err_exts, im_pixel_scales, 
                    phot_imgs, phot_pix_unit, phot_img_headers, rms_err_imgs, seg_imgs, 
                    aperture_dict, psf_matched_data, psf_matched_rms_err, pixedfit_map, voronoi_map,
                    binned_flux_map, binned_flux_err_map,
                    cutout_size, h5_folder)


    def dump_to_h5(self, h5folder='/nvme/scratch/work/tharvey/resolved_sedfitting/galaxies/'):
        '''Dump the galaxy data to an .h5 file'''
        # for strings

        if not os.path.exists(h5folder):
            print('Making directory', h5folder)
            os.makedirs(h5folder)

        str_dt = h5.string_dtype(encoding='utf-8')
        # Convert most dictionaries to strings
        # 'meta' - galaxy ID, survey, sky_coord, version, instruments, excl_bands, cutout_size, zps, pixel_scales, phot_pix_unit
        # 'paths' - im_paths, seg_paths, rms_err_paths
        # 'raw_data' - phot_imgs, rms_err_imgs, seg_imgs
        # 'headers' - phot_img_headers
        # 'bin_maps' - voronoi_map, pixedfit_map
        # 'psf_matched_data'
        # 'aperture_photometry'
        # 'auto_photometry'
        # 'sed_fitting'[tool] =
        hfile = h5.File(self.h5_path, 'w')

        # Create groups
        hfile.create_group('meta')
        hfile.create_group('paths')
        hfile.create_group('raw_data')
        hfile.create_group('aperture_photometry')
        hfile.create_group('headers')
        hfile.create_group('bin_maps')
        hfile.create_group('bin_fluxes')
        hfile.create_group('bin_flux_err')

        # Save meta data
        hfile['meta'].create_dataset('galaxy_id', data=str(self.galaxy_id), dtype=str_dt)
        hfile['meta'].create_dataset('survey', data=self.survey, dtype=str_dt)
        hfile['meta'].create_dataset('sky_coord', data=self.sky_coord.to_string(), dtype=str_dt)
        hfile['meta'].create_dataset('bands', data=str(list(self.bands)), dtype=str_dt)
        hfile['meta'].create_dataset('cutout_size', data=self.cutout_size)
        hfile['meta'].create_dataset('zps', data=str(self.im_zps), dtype=str_dt)
        hfile['meta'].create_dataset('pixel_scales', data=str({band:str(scale)
            for band, scale in self.im_pixel_scales.items()}), dtype=str_dt)
        hfile['meta'].create_dataset('phot_pix_unit', data=str({band:str(pix_unit) 
            for band, pix_unit in self.phot_pix_unit.items()}), dtype=str_dt)

        # Save paths and exts
        hfile['paths'].create_dataset('im_paths', data=str(self.im_paths), dtype=str_dt)
        hfile['paths'].create_dataset('seg_paths', data=str(self.seg_paths), dtype=str_dt)
        hfile['paths'].create_dataset('rms_err_paths', data=str(self.rms_err_paths), dtype=str_dt)
        hfile['paths'].create_dataset('im_exts', data=str(self.im_exts), dtype=str_dt)
        hfile['paths'].create_dataset('rms_err_exts', data=str(self.rms_err_exts), dtype=str_dt)

        for aper in self.aperture_dict.keys():
            hfile['aperture_photometry'].create_group(aper)
            for key in self.aperture_dict[aper].keys():
                data = self.aperture_dict[aper][key]
                #if type(data) == np.ma.core.MaskedArray:
                #    data = data.data[~data.mask]
                #if type(data) == type(Masked(u.Quantity(1))):
                    #data = data.value[~data.mask]
                    
                hfile['aperture_photometry'][aper].create_dataset(f'{key}', data=data)
        
        # Save raw data
        for band in self.bands:
            hfile['raw_data'].create_dataset(f'phot_{band}', data=self.phot_imgs[band])
            hfile['raw_data'].create_dataset(f'rms_err_{band}', data=self.rms_err_imgs[band])
            hfile['raw_data'].create_dataset(f'seg_{band}', data=self.seg_imgs[band])

        # Save headers
        for band in self.bands:
            hfile['headers'].create_dataset(f'{band}', data=str(self.phot_img_headers[band]), dtype=str_dt)

        if self.psf_matched_data is not None:
            hfile.create_group('psf_matched_data')
            for psf_type in self.psf_matched_data.keys():
                hfile['psf_matched_data'].create_group(psf_type)
                for band in self.bands:
                    hfile['psf_matched_data'][psf_type].create_dataset(band, data=self.psf_matched_data[psf_type][band])

        if self.psf_matched_rms_err is not None:
            hfile.create_group('psf_matched_rms_err')
            for psf_type in self.psf_matched_rms_err.keys():
                hfile['psf_matched_rms_err'].create_group(psf_type)
                for band in self.bands:
                    hfile['psf_matched_rms_err'][psf_type].create_dataset(band, data=self.psf_matched_rms_err[psf_type][band])
        # Save binned maps

        if self.voronoi_map is not None:
            hfile['bin_maps'].create_dataset('voronoi', data=self.voronoi_map)
        if self.pixedfit_map is not None:   
            hfile['bin_maps'].create_dataset('pixedfit', data=self.pixedfit_map)
        if self.binned_flux_map is not None:
            hfile['bin_fluxes'].create_dataset('pixedfit', data=self.binned_flux_map)
        if self.binned_flux_err_map is not None:
            hfile['bin_flux_err'].create_dataset('pixedfit', data=self.binned_flux_err_map)


    def convolve_with_psf(self, psf_type = 'webbpsf'):
        '''Convolve the images with the PSF'''
        if getattr(self, 'psf_matched_data', None) in [None, {}] or getattr(self, 'psf_matched_rms_err', None) in [None, {}]:
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
                # Do the convolution
                self.psf_matched_data = {psf_type:{}}
                self.psf_matched_rms_err = {psf_type:{}}
                for band in self.bands[:-1]:
                    kernel_path = self.psf_kernels[psf_type][band]
                    kernel = fits.open(kernel_path)[0].data
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
                print(f'psf_matched_data/{psf_type}/{self.bands[-1]}')
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
        return str

    def __repr__(self):
        return self.__str__()

    def plot_cutouts(self, bands = None, save = False, save_path = None):
        '''Plot the cutouts for the galaxy'''
        if bands is None:
            bands = self.bands
        fig, axes = plt.subplots(2, len(bands), figsize=(4*len(bands), 8))
        for i, band in enumerate(bands):
            axes[0, i].imshow(self.phot_imgs[band], origin='lower', interpolation='none')
            axes[0, i].set_title(f'{band} Phot')
            axes[1, i].imshow(self.rms_err_imgs[band], origin='lower', interpolation='none')
            axes[1, i].set_title(f'{band} RMS Err')
        plt.tight_layout()
        if save:
            plt.savefig(save_path)
        else:
            plt.show()

    def pixedfit_processing(self, use_galfind_seg = True, seg_combine = None,
            dir_images = '/nvme/scratch/work/tharvey/resolved_sedfitting/galaxies/', psf_type = 'webbpsf'):
        from piXedfit.piXedfit_images import images_processing

        if not os.path.exists(dir_images):
            os.makedirs(dir_images)

        filters = [f'jwst_nircam_{band.lower()}' for band in self.bands]
        sci_img = {}
        var_img = {}
        img_unit = {}
        scale_factors = {}
        
        for f, band in zip(filters, self.bands):
           
            data = self.psf_matched_data[psf_type][band]
            err = self.psf_matched_rms_err[psf_type][band]
            var = np.square(err)
            header = Header.fromstring(self.phot_img_headers[band], sep='\n')
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
        
        img_process = images_processing(filters, sci_img, var_img, gal_ra, gal_dec, 
                        dir_images=dir_images, img_unit=img_unit, img_scale=scale_factors, 
                        img_pixsizes=img_pixsizes, run_image_processing=True, stamp_size=(self.cutout_size, self.cutout_size),
                        flag_psfmatch=flag_psfmatch, flag_reproject=flag_reproject, 
                        flag_crop=flag_crop, kernels=None, gal_z=gal_z, remove_files=remove_files)
        
        
        # Get galaxy region from segmentation map
        if not use_galfind_seg:
            img_process.segmentation_sep()
        else:
            segm_maps = []
            for band in self.bands:
                segm = self.seg_imgs[band]
                #change to 0 is background, 1 is galaxy
                # Get value in center
                center = segm[self.cutout_size//2, self.cutout_size//2]

                segm[segm == center] = 1
                segm[segm != 1] = 0
                segm_maps.append(segm)

            img_process.segm_maps = segm_maps


        if seg_combine is not None:
            segm_maps_ids = np.argwhere(np.array([band in seg_combine for band in self.bands])).flatten()
        else:
            segm_maps_ids = None

        self.gal_region = img_process.galaxy_region(segm_maps_ids=segm_maps_ids)

        # Calculate maps of multiband fluxes
        flux_maps_fits = f"{dir_images}/{self.galaxy_id}_fluxmap.fits"
        Gal_EBV = 0 # Placeholder

        img_process.flux_map(self.gal_region, Gal_EBV=Gal_EBV, name_out_fits=flux_maps_fits)
        
        files = glob.glob('*crop_*')
        for file in files:
            os.remove(file)
        files = glob.glob(dir_images+'/*crop_*')
        for file in files:
            os.remove(file)

        self.flux_map_path = flux_maps_fits
        self.img_process = img_process

        self.add_to_h5(flux_maps_fits, 'galaxy_region', 'pixedfit',  ext = 'GALAXY_REGION')

        # Copy flux map to h5 file - TODO
        self.dir_images = dir_images

        return img_process


    def pixedfit_binning(self, SNR_reqs=10, ref_band='F277W', Dmin_bin=5, redc_chi2_limit=5.0, del_r=2.0):
        '''
        : SNR_reqs: list of SNR requirements for each band
        : ref_band: reference band for pixel binning
        : Dmin_bin: minimum diameter between pixels in binning (should be ~ FWHM of PSF)
        '''
        from piXedfit.piXedfit_bin import pixel_binning     
        SNR = np.zeros(len(self.bands))
        ref_band_pos = np.argwhere(np.array([band == ref_band for band in self.bands])).flatten()[0]
        # Should calculate SNR requirements intelligently based on redshift
        SNR[2:] = SNR_reqs
        name_out_fits = f'{self.dir_images}/{self.galaxy_id}_binned.fits'
        
        pixel_binning(self.flux_map_path, ref_band=ref_band_pos, Dmin_bin=Dmin_bin, SNR=SNR, redc_chi2_limit=redc_chi2_limit, del_r=del_r,  name_out_fits=name_out_fits)
        
        self.pixedfit_binmap_path = name_out_fits
        print('Adding to h5')
        self.add_to_h5(name_out_fits, 'bin_maps', 'pixedfit', ext='BIN_MAP', setattr_gal='pixedfit_map')
        self.add_to_h5(name_out_fits, 'bin_fluxes', 'pixedfit', ext='BIN_FLUX', setattr_gal='binned_flux_map')
        self.add_to_h5(name_out_fits, 'bin_flux_err', 'pixedfit', ext='BIN_FLUXERR', setattr_gal='binned_flux_err_map')


    def plot_image_stamps(self):
        fig, axes = plt.subplots(1, len(self.bands), figsize=(4*len(self.bands), 4))
        for i, band in enumerate(self.bands):
            axes[i].imshow(np.log10(self.phot_imgs[band]), origin='lower', interpolation='none')
            axes[i].set_title(f'{band} Image')
        plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, hspace=0.1, wspace=0.15)

    def plot_gal_region(self):
        gal_region = self.gal_region
        fig, axes = plt.subplots(1, len(self.bands), figsize=(4*len(self.bands), 4))
        for i, band in enumerate(self.bands):
            rows, cols = np.where(gal_region==0)
            gal_region[rows,cols] = float('nan')
            axes[i].imshow(np.log10(self.phot_imgs[band]), origin='lower', interpolation='none')
            axes[i].set_title(f'{band} Image')
            axes[i].imshow(gal_region, origin='lower', interpolation='none', alpha=0.5, cmap='copper')
        plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, hspace=0.1, wspace=0.15)

    def add_to_h5(self, data, group, name, ext=0, setattr_gal=None, overwrite=False):
        

        if type(data) == str:
            if data.endswith('.fits'):
                data = fits.open(data)[ext].data
        
        hfile = h5.File(self.h5_path, 'a')
        if group not in hfile.keys():
            hfile.create_group(group)
        if name in hfile[group].keys():
            if overwrite:
                del hfile[group][name]
            else:
                return

        hfile[group].create_dataset(name, data=data)
        print('added to ', hfile, group, name)
        hfile.close()
        if setattr_gal is not None:
            setattr(self, setattr_gal, data)


    def plot_err_stamps(self):
        fig, axes = plt.subplots(1, len(self.bands), figsize=(4*len(self.bands), 4))
        for i, band in enumerate(self.bands):
            axes[i].imshow(np.log10(self.rms_err_imgs[band]), origin='lower', interpolation='none')
            axes[i].set_title(f'{band} Error')

    def pixedfit_plot_map_fluxes(self):
        from piXedfit.piXedfit_images import plot_maps_fluxes
        plot_maps_fluxes(self.flux_map_path, ncols=8, savefig=False)

    def pixedfit_plot_radial_SNR(self):
        from piXedfit.piXedfit_images import plot_SNR_radial_profile
        plot_SNR_radial_profile(self.flux_map_path, savefig=False)

    def pixedfit_plot_image_stamps(self):
        if not hasattr(self, 'img_process'):
            raise Exception("Need to run pixedfit_processing first")
        self.img_process.plot_image_stamps(savefig=False)

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

if __name__ == "__main__":
    # Test the Galaxy class
    galaxy = ResolvedGalaxy.init_from_galfind(645, 'NGDEEP2', 'v11', excl_bands = ['F435W', 'F775W', 'F850LP'])
    galaxy2 = ResolvedGalaxy.init_from_h5(645)
    galaxy2.pixedfit_processing()
    
          
    