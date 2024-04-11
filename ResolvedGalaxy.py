from astropy.io import fits
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
import os
import typing
from astropy.utils.exceptions import AstropyWarning
import warnings
from astropy.utils.masked import Masked
from astropy.wcs import WCS
# can write astropy to h5


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
                phot_img_headers, rms_err_imgs, seg_imgs, aperture_dict, cutout_size=64, 
                h5_folder = '/nvme/scratch/work/tharvey/resolved_sedfitting/galaxies/',
                 psf_kernel_folder = '/nvme/scratch/work/tharvey/resolved_sedfitting/psfs/', 
                 psf_type = 'webbpsf'):

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
        
        # Check sizes of images
        for band in self.bands:
            assert self.phot_imgs[band].shape == (self.cutout_size, self.cutout_size), f"Image shape for {band} is {self.phot_imgs[band].shape}, not {(self.cutout_size, self.cutout_size)}"
            assert self.rms_err_imgs[band].shape == (self.cutout_size, self.cutout_size), f"RMS error image shape for {band} is {self.rms_err_imgs[band].shape}, not {(self.cutout_size, self.cutout_size)}"
            assert self.seg_imgs[band].shape == (self.cutout_size, self.cutout_size), f"Segmentation map shape for {band} is {self.seg_imgs[band].shape}, not {(self.cutout_size, self.cutout_size)}"
        # Bin the pixels
        self.voronoi_map = None
        self.pixedfit_map = None


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
       
        self.dump_to_h5()

        self.convolve_with_psf(psf_type = psf_type)
         # Save to .h5
        
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
                    phot_imgs,phot_pix_unit, phot_img_headers, rms_err_imgs, seg_imgs, aperture_dict, cutout_size)
    
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

        hfile.close()

        return cls(galaxy_id, sky_coord, survey, bands, im_paths, im_exts, im_zps,
                    seg_paths, rms_err_paths, rms_err_exts, im_pixel_scales, 
                    phot_imgs, phot_pix_unit, phot_img_headers, rms_err_imgs, seg_imgs, aperture_dict, cutout_size, h5_folder)


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

    def convolve_with_psf(self, psf_type = 'webbpsf'):
        if getattr(self, 'psf_matched_data', None) in [None, {}]:
            # Try and load from .h5
            h5file = h5.File(self.h5_path, 'a')
            if 'psf_matched_data' in h5file.keys():
                self.psf_matched_data = {}
                for band in self.bands:
                    self.psf_matched_data[band] = h5file['psf_matched_data'][psf_type][band][()]
            else:
                h5file.create_group('psf_matched_data')
                # Do the convolution
                self.psf_matched_data = {}
                for band in self.bands[:-1]:
                    kernel_path = self.psf_kernels[psf_type][band]
                    kernel = fits.open(kernel_path)[0].data
                    # Convolve the image with the PSF
                    psf_matched_img = convolve_fft(self.phot_imgs[band], kernel, normalize_kernel=True, boundary='wrap')

                    # Save to psf_matched_data
                    self.psf_matched_data[band] = psf_matched_img
                    h5file.create_dataset(f'psf_matched_data/{psf_type}/{band}', data=psf_matched_img)
                    #h5file['psf_matched_data'][psf_type] = self.psf_matched_data
                self.psf_matched_data[self.bands[-1]] = self.phot_imgs[self.bands[-1]] # No need to convolve the last band
                h5file.create_dataset(f'psf_matched_data/{psf_type}/{self.bands[-1]}', data=self.phot_imgs[self.bands[-1]])

                h5file.close()
    
    def __str__(self):
        str = f"Resolved Galaxy {self.galaxy_id} from {self.survey} survey\n"
        str += f"SkyCoord: {self.sky_coord}\n"
        str += f"Bands: {self.bands}\n"
        str += f"Cutout size: {self.cutout_size}\n"
        str += f"Aperture photometry: {self.aperture_dict}\n"
        return str

if __name__ == "__main__":
    # Test the Galaxy class
    galaxy = ResolvedGalaxy.init_from_galfind(645, 'NGDEEP2', 'v11', excl_bands = ['F435W', 'F775W', 'F850LP'])
    galaxy2 = ResolvedGalaxy.init_from_h5(645)
    print(galaxy2.galaxy_id)
          
    