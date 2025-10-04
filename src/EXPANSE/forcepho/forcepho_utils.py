"""
Utility functions for preparing forcepho inputs from ResolvedGalaxy objects.

This module provides helper functions to convert EXPANSE ResolvedGalaxy data
into the format required by forcepho (https://forcepho.readthedocs.io/).

Key forcepho input requirements:
- Image data (cutouts) for each band
- Uncertainty/error maps
- PSF models for each band
- Source positions (RA, Dec)
- WCS information
- Zero points and exposure times
"""

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table
import os


def prepare_pixel_data(resolved_galaxy, bands=None, psf_matched=True):
    """
    Extract pixel data from ResolvedGalaxy for forcepho.
    
    Parameters
    ----------
    resolved_galaxy : ResolvedGalaxy
        The ResolvedGalaxy object containing image data
    bands : list, optional
        List of bands to include. If None, uses all bands from resolved_galaxy
    psf_matched : bool, optional
        Whether to use PSF-matched data. Default is True.
        
    Returns
    -------
    pixel_data : dict
        Dictionary with band names as keys and image arrays as values
    error_data : dict
        Dictionary with band names as keys and error arrays as values
    """
    if bands is None:
        bands = resolved_galaxy.bands
        
    pixel_data = {}
    error_data = {}
    
    for band in bands:
        if psf_matched and resolved_galaxy.psf_matched_data is not None:
            if band in resolved_galaxy.psf_matched_data:
                pixel_data[band] = resolved_galaxy.psf_matched_data[band]
            else:
                pixel_data[band] = resolved_galaxy.phot_imgs[band]
        else:
            pixel_data[band] = resolved_galaxy.phot_imgs[band]
            
        # Get error data
        if psf_matched and resolved_galaxy.psf_matched_rms_err is not None:
            if band in resolved_galaxy.psf_matched_rms_err:
                error_data[band] = resolved_galaxy.psf_matched_rms_err[band]
            else:
                error_data[band] = resolved_galaxy.rms_err_imgs[band]
        else:
            error_data[band] = resolved_galaxy.rms_err_imgs[band]
            
    return pixel_data, error_data


def prepare_psf_data(resolved_galaxy, bands=None, psf_type="star_stack"):
    """
    Extract PSF data from ResolvedGalaxy for forcepho.
    
    Parameters
    ----------
    resolved_galaxy : ResolvedGalaxy
        The ResolvedGalaxy object containing PSF data
    bands : list, optional
        List of bands to include. If None, uses all bands from resolved_galaxy
    psf_type : str, optional
        Type of PSF to use ('star_stack' or 'webbpsf'). Default is 'star_stack'.
        
    Returns
    -------
    psf_data : dict
        Dictionary with band names as keys and PSF arrays as values
    psf_meta : dict
        Dictionary with band names as keys and PSF metadata as values
    """
    if bands is None:
        bands = resolved_galaxy.bands
        
    psf_data = {}
    psf_meta = {}
    
    if resolved_galaxy.psfs is not None and psf_type in resolved_galaxy.psfs:
        for band in bands:
            if band in resolved_galaxy.psfs[psf_type]:
                psf_data[band] = resolved_galaxy.psfs[psf_type][band]
                
                # Get metadata if available
                if resolved_galaxy.psfs_meta is not None:
                    if psf_type in resolved_galaxy.psfs_meta:
                        if band in resolved_galaxy.psfs_meta[psf_type]:
                            psf_meta[band] = resolved_galaxy.psfs_meta[psf_type][band]
    
    return psf_data, psf_meta


def prepare_source_catalog(resolved_galaxy, positions=None):
    """
    Prepare source catalog for forcepho.
    
    Parameters
    ----------
    resolved_galaxy : ResolvedGalaxy
        The ResolvedGalaxy object
    positions : list of tuples, optional
        List of (ra, dec) positions in degrees. If None, uses the galaxy center.
        
    Returns
    -------
    catalog : astropy.table.Table
        Table with source positions and properties
    """
    if positions is None:
        # Use galaxy center
        positions = [(resolved_galaxy.sky_coord.ra.deg, resolved_galaxy.sky_coord.dec.deg)]
    
    # Create table with positions
    catalog = Table()
    catalog['ra'] = [pos[0] for pos in positions]
    catalog['dec'] = [pos[1] for pos in positions]
    catalog['source_id'] = np.arange(len(positions))
    
    # Add galaxy ID as metadata
    catalog.meta['galaxy_id'] = resolved_galaxy.galaxy_id
    
    return catalog


def prepare_wcs_info(resolved_galaxy, band=None):
    """
    Extract WCS information from ResolvedGalaxy.
    
    Parameters
    ----------
    resolved_galaxy : ResolvedGalaxy
        The ResolvedGalaxy object
    band : str, optional
        Band to get WCS from. If None, uses first available band.
        
    Returns
    -------
    wcs : astropy.wcs.WCS
        WCS object for the cutout
    """
    if band is None:
        band = resolved_galaxy.bands[0]
    
    # Get WCS from header if available
    if resolved_galaxy.phot_img_headers is not None:
        if band in resolved_galaxy.phot_img_headers:
            header = resolved_galaxy.phot_img_headers[band]
            wcs = WCS(header)
            return wcs
    
    # Create a simple WCS if header not available
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [resolved_galaxy.cutout_size / 2, resolved_galaxy.cutout_size / 2]
    wcs.wcs.crval = [resolved_galaxy.sky_coord.ra.deg, resolved_galaxy.sky_coord.dec.deg]
    
    # Get pixel scale
    if band in resolved_galaxy.im_pixel_scales:
        pixel_scale = resolved_galaxy.im_pixel_scales[band]
    else:
        pixel_scale = 0.03  # Default JWST NIRCam pixel scale in arcsec
        
    wcs.wcs.cdelt = [-pixel_scale / 3600, pixel_scale / 3600]  # Convert to degrees
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    
    return wcs


def prepare_photometric_properties(resolved_galaxy, bands=None):
    """
    Extract photometric properties for forcepho.
    
    Parameters
    ----------
    resolved_galaxy : ResolvedGalaxy
        The ResolvedGalaxy object
    bands : list, optional
        List of bands to include. If None, uses all bands from resolved_galaxy
        
    Returns
    -------
    properties : dict
        Dictionary with photometric properties including zero points,
        pixel scales, and exposure information
    """
    if bands is None:
        bands = resolved_galaxy.bands
        
    properties = {
        'zero_points': {},
        'pixel_scales': {},
        'pixel_units': {},
    }
    
    for band in bands:
        if band in resolved_galaxy.im_zps:
            properties['zero_points'][band] = resolved_galaxy.im_zps[band]
        
        if band in resolved_galaxy.im_pixel_scales:
            properties['pixel_scales'][band] = resolved_galaxy.im_pixel_scales[band]
            
        if band in resolved_galaxy.phot_pix_unit:
            properties['pixel_units'][band] = resolved_galaxy.phot_pix_unit[band]
    
    return properties


def create_forcepho_config(resolved_galaxy, bands=None, psf_type="star_stack",
                          positions=None, output_dir=None):
    """
    Create a complete configuration dictionary for forcepho.
    
    Parameters
    ----------
    resolved_galaxy : ResolvedGalaxy
        The ResolvedGalaxy object
    bands : list, optional
        List of bands to include. If None, uses all bands from resolved_galaxy
    psf_type : str, optional
        Type of PSF to use. Default is 'star_stack'.
    positions : list of tuples, optional
        List of (ra, dec) positions. If None, uses galaxy center.
    output_dir : str, optional
        Directory to save output files. If None, creates in current directory.
        
    Returns
    -------
    config : dict
        Configuration dictionary containing all forcepho inputs
    """
    if bands is None:
        bands = resolved_galaxy.bands
        
    if output_dir is None:
        output_dir = f"forcepho_output_{resolved_galaxy.galaxy_id}"
    
    # Prepare all components
    pixel_data, error_data = prepare_pixel_data(resolved_galaxy, bands=bands)
    psf_data, psf_meta = prepare_psf_data(resolved_galaxy, bands=bands, psf_type=psf_type)
    catalog = prepare_source_catalog(resolved_galaxy, positions=positions)
    wcs = prepare_wcs_info(resolved_galaxy)
    properties = prepare_photometric_properties(resolved_galaxy, bands=bands)
    
    config = {
        'galaxy_id': resolved_galaxy.galaxy_id,
        'bands': bands,
        'pixel_data': pixel_data,
        'error_data': error_data,
        'psf_data': psf_data,
        'psf_meta': psf_meta,
        'source_catalog': catalog,
        'wcs': wcs,
        'properties': properties,
        'output_dir': output_dir,
        'cutout_size': resolved_galaxy.cutout_size,
        'sky_coord': resolved_galaxy.sky_coord,
    }
    
    return config


def save_forcepho_inputs(config, output_dir=None):
    """
    Save forcepho input data to FITS files.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary from create_forcepho_config
    output_dir : str, optional
        Directory to save files. If None, uses config['output_dir'].
        
    Returns
    -------
    file_paths : dict
        Dictionary with paths to saved files
    """
    if output_dir is None:
        output_dir = config['output_dir']
        
    os.makedirs(output_dir, exist_ok=True)
    
    file_paths = {
        'images': {},
        'errors': {},
        'psfs': {},
        'catalog': None,
    }
    
    # Save image data
    for band in config['bands']:
        if band in config['pixel_data']:
            img_path = os.path.join(output_dir, f"{band}_image.fits")
            fits.writeto(img_path, config['pixel_data'][band], overwrite=True)
            file_paths['images'][band] = img_path
            
        if band in config['error_data']:
            err_path = os.path.join(output_dir, f"{band}_error.fits")
            fits.writeto(err_path, config['error_data'][band], overwrite=True)
            file_paths['errors'][band] = err_path
            
        if band in config['psf_data']:
            psf_path = os.path.join(output_dir, f"{band}_psf.fits")
            fits.writeto(psf_path, config['psf_data'][band], overwrite=True)
            file_paths['psfs'][band] = psf_path
    
    # Save source catalog
    catalog_path = os.path.join(output_dir, "source_catalog.fits")
    config['source_catalog'].write(catalog_path, overwrite=True)
    file_paths['catalog'] = catalog_path
    
    # Save WCS as a header
    wcs_path = os.path.join(output_dir, "wcs.fits")
    header = config['wcs'].to_header()
    # Create a minimal FITS file with just the WCS header
    fits.PrimaryHDU(header=header).writeto(wcs_path, overwrite=True)
    file_paths['wcs'] = wcs_path
    
    return file_paths


def validate_forcepho_inputs(config):
    """
    Validate that all required inputs for forcepho are present.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary from create_forcepho_config
        
    Returns
    -------
    valid : bool
        True if all required inputs are present
    missing : list
        List of missing required inputs
    """
    missing = []
    
    # Check required components
    if not config.get('pixel_data'):
        missing.append('pixel_data')
    if not config.get('error_data'):
        missing.append('error_data')
    if not config.get('source_catalog'):
        missing.append('source_catalog')
    if not config.get('wcs'):
        missing.append('wcs')
        
    # Check that all bands have data
    if config.get('bands'):
        for band in config['bands']:
            if band not in config.get('pixel_data', {}):
                missing.append(f'pixel_data for {band}')
            if band not in config.get('error_data', {}):
                missing.append(f'error_data for {band}')
    
    valid = len(missing) == 0
    return valid, missing
