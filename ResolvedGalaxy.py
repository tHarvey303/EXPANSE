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
# This class is designed to hold all the data for a galaxy, including the cutouts, segmentation maps, and RMS error maps.

'''TODO:
        1. Store aperture photometry, auto photometry from catalogue. 
        2. Pixel binning.
        3. Does ERR map need to be convolved with PSF?
'''


class ResolvedGalaxy:
    def __init__(self, galaxy_id, sky_coord, survey, bands, im_paths, im_exts, im_zps, seg_paths, seg_exts,
                rms_err_paths, rms_err_exts, im_pixel_scales, phot_imgs, phot_img_headers, rms_err_imgs, seg_imgs, aperture_dict,
                cutout_size=64, h5_folder = '/nvme/scratch/work/tharvey/resolved_sedfitting/galaxies/', psf_kernel_folder = '/nvme/scratch/work/tharvey/resolved_sedfitting/psf_kernels/', psf_type = 'webbpsf'):
        self.galaxy_id = galaxy_id
        self.sky_coord = sky_coord
        self.survey = survey
        self.bands = bands
        self.im_paths = img_paths
        self.im_exts = im_exts
        self.im_zps = im_zps
        self.seg_paths = seg_paths
        self.seg_exts = seg_exts
        self.rms_err_paths = rms_err_paths
        self.rms_err_exts = rms_err_exts
        self.im_pixel_scales = im_pixel_scales
        self.cutout_size = cutout_size
        self.aperture_dict = aperture_dict

        # Actual cutouts
        self.phot_imgs = phot_imgs
        self.phot_img_headers = phot_img_headers
        self.rms_err_imgs = rms_err_imgs
        self.seg_imgs = seg_imgs

        self.h5_path = h5_folder + f'{self.galaxy_id}.h5'

        # Check all images are the same dimensions
        if not all([img.shape == self.phot_imgs[0].shape for img in self.phot_imgs]):
            raise Exception("Images are not the same size")
        if not all([img.shape == self.rms_err_imgs[0].shape for img in self.rms_err_imgs]):
            raise Exception("RMS error images are not the same size")
        if not all([img.shape == self.seg_imgs[0].shape for img in self.seg_imgs]):
            raise Exception("Segmentation maps are not the same size")
        
        # Bin the pixels
        self.voronoi_map = None
        self.pixedfit_map = None

        self.psf_kernels = {}
        # Assume bands is in wavelength order, and that the largest PSF is in the last band
        print(f'Assuming {bands[-1]} is the band with the largest PSF, and convolving all bands with this PSF kernel.')
        for band in bands[:-1]:
            files = glob.glob(f'{psf_kernel_folder}{band}*{bands[-1]}.fits')
            if len(files) == 0:
                raise Exception(f"No PSF kernel found between {band} and {bands[-1]}")
            elif len(files) > 1:
                raise Exception(f"Multiple PSF kernels found between {band} and {bands[-1]}")
            else:
                self.psf_kernels[band] = files[0]
        # Save to .h5
        self.dump_to_h5()
        
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
        galaxy = [gal for gal in cat.galaxies if gal.galaxy_id == galaxy_id]
        
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
        err_paths = cat.data.err_paths
        err_exts = cat.data.err_exts
        seg_paths = cat.data.seg_paths
        seg_exts = cat.data.seg_exts
        im_zps = cat.data.im_zps
        im_pixel_scales = cat.data.instrument.pixel_size
        bands = cat.data.instrument.bands # should be bands just for galaxy!

        # Get things from galaxy object
        galaxy_skycoord = galaxy.sky_coord
        bands_mask = galaxy.phot.flux_Jy.mask
        bands = bands[~bands_mask]
        # Get aperture photometry
        flux_aper = galaxy.phot.flux_Jy[~bands_mask]
        flux_err_aper = galaxy.phot.flux_err_Jy[~bands_mask]
        depths = galaxy.phot.depths[~bands_mask]
        # Get the wavelegnths
        wave = galaxy.phot.band_wavelengths[~bands_mask]
        # aperture_dict
        aperture_dict = {0.32*u.arcsec: {'flux': flux_aper, 'flux_err': flux_err_aper, 'depths': depths, 'wave': wave}}
        
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
            rms_data = hdu['RMS'].data
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
            phot_img_headers[band] = hdu['SCI'].header

        return cls(galaxy_id, galaxy_skycoord, survey, bands, im_paths, im_exts,
                    im_zps, seg_paths, seg_exts, err_paths, err_exts, im_pixel_scales,
                    phot_imgs, phot_img_headers, rms_err_imgs, seg_imgs, cutout_size)
    
    @classmethod
    def init_from_h5(cls, galaxy_id, h5_folder = '/nvme/scratch/work/tharvey/resolved_sedfitting/galaxies/'):
        '''Load a galaxy from an .h5 file'''
        h5path = f'{h5_folder}{id}.h5'
        hfile = h5.File(h5path, 'r')
        # Load meta data
        galaxy_id = hfile['meta']['galaxy_id'][()]
        survey = hfile['meta']['survey'][()]
        sky_coord = SkyCoord(hfile['meta']['sky_coord'][()])
        bands = ast.literal_eval(hfile['meta']['bands'][()])
        cutout_size = hfile['meta']['cutout_size'][()]
        im_zps = ast.literal_eval(hfile['meta']['zps'][()])
        im_pixel_scales = ast.literal_eval(hfile['meta']['pixel_scales'][()])
        phot_pix_unit = ast.literal_eval(hfile['meta']['phot_pix_unit'][()])
        # Load paths and exts
        im_paths = ast.literal_eval(hfile['paths']['im_paths'][()])
        im_exts = ast.literal_eval(hfile['paths']['im_exts'][()])
        seg_paths = ast.literal_eval(hfile['paths']['seg_paths'][()])
        seg_exts = ast.literal_eval(hfile['paths']['seg_exts'][()])
        rms_err_paths = ast.literal_eval(hfile['paths']['rms_err_paths'][()])
        rms_err_exts = ast.literal_eval(hfile['paths']['rms_err_exts'][()])
        # Load aperture photometry
        aperture_dict = ast.literal_eval(hfile['aperture_photometry']['aperture_dict'][()])
        # Load raw data
        phot_imgs = {}
        rms_err_imgs = {}
        seg_imgs = {}
        phot_img_headers = {}
        for band in bands:
            phot_imgs[band] = hfile['raw_data'][f'phot_{band}'][()]
            rms_err_imgs[band] = hfile['raw_data'][f'rms_err_{band}'][()]
            seg_imgs[band] = hfile['raw_data'][f'seg_{band}'][()]
            phot_img_headers[band] = ast.literal_eval(hfile['headers'][band][()])

        hfile.close()

        return cls(galaxy_id, sky_coord, survey, bands, im_paths, im_exts, im_zps,
                    seg_paths, seg_exts, rms_err_paths, rms_err_exts, im_pixel_scales, 
                    phot_imgs, phot_img_headers, rms_err_imgs, seg_imgs, cutout_size, h5_folder)


    def dump_to_h5(self, h5folder='/nvme/scratch/work/tharvey/resolved_sedfitting/galaxies/'):
        '''Dump the galaxy data to an .h5 file'''
        # for strings

        if not os.path.exists(h5folder):
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
        hfile.create_group('headers')
        hfile.create_group('bin_maps')

        # Save meta data
        hfile['meta'].create_dataset('galaxy_id', data=self.galaxy_id, dtype=str_dt)
        hfile['meta'].create_dataset('survey', data=self.survey, dtype=str_dt)
        hfile['meta'].create_dataset('sky_coord', data=self.sky_coord.to_string(), dtype=str_dt)
        hfile['meta'].create_dataset('bands', data=str(self.bands), dtype=str_dt)
        hfile['meta'].create_dataset('cutout_size', data=self.cutout_size)
        hfile['meta'].create_dataset('zps', data=str(self.im_zps), dtype=str_dt)
        hfile['meta'].create_dataset('pixel_scales', data=str(self.im_pixel_scales), dtype=str_dt)
        hfile['meta'].create_dataset('phot_pix_unit', data=str(self.phot_pix_unit), dtype=str_dt)

        # Save paths and exts
        hfile['paths'].create_dataset('im_paths', data=str(self.im_paths), dtype=str_dt)
        hfile['paths'].create_dataset('seg_paths', data=str(self.seg_paths), dtype=str_dt)
        hfile['paths'].create_dataset('rms_err_paths', data=str(self.rms_err_paths), dtype=str_dt)
        hfile['paths'].create_dataset('im_exts', data=str(self.im_exts), dtype=str_dt)
        hfile['paths'].create_dataset('seg_exts', data=str(self.seg_exts), dtype=str_dt)
        hfile['paths'].create_dataset('rms_err_exts', data=str(self.rms_err_exts), dtype=str_dt)

        hfile['aperture_photometry'].create_dataset('aperture_dict', data=str(self.aperture_dict), dtype=str_dt)

        # Save raw data
        for band in self.bands:
            hfile['raw_data'].create_dataset(f'phot_{band}', data=self.phot_imgs[band])
            hfile['raw_data'].create_dataset(f'rms_err_{band}', data=self.rms_err_imgs[band])
            hfile['raw_data'].create_dataset(f'seg_{band}', data=self.seg_imgs[band])

        # Save headers
        for band in self.bands:
            hfile['headers'].create_dataset(f'{band}', data=str(self.phot_img_headers[band]), dtype=str_dt)

    def convolve_with_psf():
        if getattr(self, 'psf_matched_data', None) is None:
            # Try and load from .h5
            h5file = h5.File(self.h5_path, 'a')
            if 'psf_matched_data' in h5file.keys():
                self.psf_matched_data = h5file['psf_matched_data']
            else:
                # Do the convolution
                self.psf_matched_data = {}
                for band in self.bands:
                    kernel = self.psf_kernels[band]

                    # Convolve the image with the PSF
                    psf_matched_img = convolve_fft(self.phot_imgs[band], kernel, normalize_kernel=True, boundary='wrap')

                    # Save to psf_matched_data
                    self.psf_matched_data[band] = psf_matched_img
                h5file['psf_matched_data'] = self.psf_matched_data
                h5file.close()
            

if __name__ == "__main__":
    # Test the Galaxy class
    galaxy = ResolvedGalaxy.init_from_galfind(52, 'NGDEEP2', 'v11', excl_bands = ['F435W', 'F775W', 'F850LP'])
    galaxy.dump_to_h5()
    galaxy2 = ResolvedGalaxy.init_from_h5(52)
    print(galaxy2.galaxy_id)
          
    