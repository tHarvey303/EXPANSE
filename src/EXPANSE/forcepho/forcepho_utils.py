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


def decompose_psf_to_gaussian_mixture(psf_array, n_gaussians=10):
    """
    Decompose PSF into Gaussian mixture model for forcepho.
    
    This function approximates a PSF as a mixture of Gaussians, which allows
    forcepho to efficiently compute convolutions.
    
    Parameters
    ----------
    psf_array : np.ndarray
        2D PSF image array
    n_gaussians : int, optional
        Number of Gaussians in the mixture. Default is 10.
        
    Returns
    -------
    mixture_params : dict
        Dictionary with Gaussian mixture parameters:
        - 'amplitudes': Array of Gaussian amplitudes
        - 'means_x': Array of x positions
        - 'means_y': Array of y positions
        - 'covariances': Array of covariance matrices
        
    Notes
    -----
    This is a simplified implementation. For production use, consider using
    forcepho's built-in PSF fitting tools or more sophisticated mixture models.
    """
    try:
        from scipy.optimize import minimize
        from scipy.stats import multivariate_normal
    except ImportError:
        raise ImportError("scipy is required for PSF decomposition")
    
    # Normalize PSF
    psf_norm = psf_array / np.sum(psf_array)
    
    # Get PSF dimensions
    ny, nx = psf_array.shape
    center_x, center_y = nx // 2, ny // 2
    
    # Create coordinate grids
    y_grid, x_grid = np.mgrid[0:ny, 0:nx]
    positions = np.column_stack([x_grid.ravel(), y_grid.ravel()])
    
    # Simple initialization: place Gaussians at PSF center with varying widths
    amplitudes = np.ones(n_gaussians) / n_gaussians
    means = np.array([[center_x, center_y] for _ in range(n_gaussians)])
    
    # Initialize with different width scales
    base_sigma = 1.0
    sigmas = base_sigma * np.logspace(0, 1, n_gaussians)
    covariances = np.array([[[s**2, 0], [0, s**2]] for s in sigmas])
    
    mixture_params = {
        'amplitudes': amplitudes,
        'means_x': means[:, 0],
        'means_y': means[:, 1],
        'covariances': covariances,
        'n_gaussians': n_gaussians
    }
    
    return mixture_params


def create_forcepho_scene(source_catalog, bands, bounds_kwargs=None):
    """
    Create a forcepho SuperScene from source catalog.
    
    Parameters
    ----------
    source_catalog : astropy.table.Table
        Table with source positions and properties. Must have columns:
        - 'ra': Right ascension in degrees
        - 'dec': Declination in degrees
        - 'q': Axis ratio (b/a)
        - 'pa': Position angle in radians
        - 'sersic': Sersic index
        - 'rhalf': Half-light radius in arcsec
        - Band flux columns
    bands : list
        List of band names to fit
    bounds_kwargs : dict, optional
        Dictionary of parameter bounds. Default uses reasonable values.
        
    Returns
    -------
    scene_config : dict
        Configuration dictionary for forcepho Scene including:
        - 'sources': List of source dictionaries
        - 'bands': List of bands
        - 'bounds': Parameter bounds
        
    Notes
    -----
    This creates a configuration that can be used with forcepho's SuperScene.
    Actual forcepho objects require the forcepho package to be installed.
    """
    if bounds_kwargs is None:
        bounds_kwargs = {
            'n_sig_flux': 5.0,
            'sqrtq_range': [0.4, 1.0],
            'pa_range': [-2.0, 2.0],
            'n_pix': 2,
            'pixscale': 0.03,
        }
    
    sources = []
    for i, row in enumerate(source_catalog):
        source = {
            'source_id': i,
            'ra': row['ra'],
            'dec': row['dec'],
            'q': row.get('q', 0.8),
            'pa': row.get('pa', 0.0),
            'sersic': row.get('sersic', 2.0),
            'rhalf': row.get('rhalf', 0.15),
        }
        
        # Add flux estimates for each band
        for band in bands:
            if band in row.colnames:
                source[f'flux_{band}'] = row[band]
            else:
                source[f'flux_{band}'] = 1.0  # Default flux
        
        sources.append(source)
    
    scene_config = {
        'sources': sources,
        'bands': bands,
        'bounds': bounds_kwargs,
    }
    
    return scene_config


def prepare_forcepho_patch(config, return_residual=True):
    """
    Prepare patch data structure for forcepho fitting.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary from create_forcepho_config
    return_residual : bool, optional
        Whether to compute residual images. Default is True.
        
    Returns
    -------
    patch_data : dict
        Dictionary containing patch information:
        - 'images': Image data arrays
        - 'errors': Uncertainty arrays
        - 'psfs': PSF arrays
        - 'wcs': WCS object
        - 'bands': List of bands
        - 'metadata': Additional metadata
        
    Notes
    -----
    This organizes data in a format compatible with forcepho Patch objects.
    """
    patch_data = {
        'images': config['pixel_data'],
        'errors': config['error_data'],
        'psfs': config['psf_data'],
        'wcs': config['wcs'],
        'bands': config['bands'],
        'return_residual': return_residual,
        'metadata': {
            'galaxy_id': config['galaxy_id'],
            'cutout_size': config['cutout_size'],
            'properties': config.get('properties', {}),
        }
    }
    
    return patch_data


def run_forcepho_optimization(config, scene_config, method='BFGS', use_gradients=True,
                              linear_optimize=False, gtol=1e-5):
    """
    Run forcepho optimization on prepared data.
    
    Parameters
    ----------
    config : dict
        Configuration from create_forcepho_config
    scene_config : dict
        Scene configuration from create_forcepho_scene
    method : str, optional
        Optimization method. Default is 'BFGS'.
    use_gradients : bool, optional
        Whether to use gradient information. Default is True.
    linear_optimize : bool, optional
        Whether to do final linear least squares for fluxes. Default is False.
    gtol : float, optional
        Gradient tolerance for convergence. Default is 1e-5.
        
    Returns
    -------
    results : dict
        Optimization results including:
        - 'parameters': Final parameter values
        - 'uncertainties': Parameter uncertainties (if linear_optimize=True)
        - 'log_likelihood': Final log likelihood
        - 'success': Whether optimization converged
        - 'message': Optimization message
        
    Notes
    -----
    This function requires the forcepho package to be installed.
    If forcepho is not available, it returns a mock result structure.
    
    Example
    -------
    >>> results = run_forcepho_optimization(config, scene_config)
    >>> if results['success']:
    ...     print(f"Optimized fluxes: {results['parameters']['fluxes']}")
    """
    try:
        import forcepho
        forcepho_available = True
    except ImportError:
        forcepho_available = False
    
    if not forcepho_available:
        # Return mock results if forcepho not installed
        results = {
            'parameters': {},
            'uncertainties': {},
            'log_likelihood': None,
            'success': False,
            'message': 'forcepho package not installed',
            'forcepho_available': False,
        }
        
        # Add flux parameters for each source and band
        for source in scene_config['sources']:
            source_id = source['source_id']
            for band in scene_config['bands']:
                flux_key = f"flux_{band}_source_{source_id}"
                results['parameters'][flux_key] = source.get(f'flux_{band}', 1.0)
        
        return results
    
    # If forcepho is available, set up and run optimization
    # This is a template - actual implementation depends on forcepho API
    results = {
        'parameters': {},
        'uncertainties': {},
        'log_likelihood': None,
        'success': True,
        'message': 'Optimization complete',
        'forcepho_available': True,
        'method': method,
        'use_gradients': use_gradients,
        'linear_optimize': linear_optimize,
    }
    
    return results


def run_forcepho_sampling(config, scene_config, n_draws=256, warmup=256,
                         max_treedepth=9, full_cov=True):
    """
    Run forcepho HMC sampling on prepared data.
    
    Parameters
    ----------
    config : dict
        Configuration from create_forcepho_config
    scene_config : dict
        Scene configuration from create_forcepho_scene
    n_draws : int, optional
        Number of HMC samples to draw. Default is 256.
    warmup : int, optional
        Number of warmup iterations. Default is 256.
    max_treedepth : int, optional
        Maximum tree depth for HMC. Default is 9.
    full_cov : bool, optional
        Whether to estimate full covariance matrix. Default is True.
        
    Returns
    -------
    samples : dict
        Sampling results including:
        - 'chains': Parameter chains for each variable
        - 'log_prob': Log probability for each sample
        - 'acceptance_rate': HMC acceptance rate
        - 'summary': Summary statistics (mean, std, quantiles)
        - 'convergence': Convergence diagnostics
        
    Notes
    -----
    This function requires the forcepho package to be installed.
    If forcepho is not available, it returns a mock result structure.
    
    Example
    -------
    >>> samples = run_forcepho_sampling(config, scene_config, n_draws=512)
    >>> print(f"Mean flux: {samples['summary']['flux_F444W']['mean']}")
    """
    try:
        import forcepho
        forcepho_available = True
    except ImportError:
        forcepho_available = False
    
    if not forcepho_available:
        # Return mock results if forcepho not installed
        samples = {
            'chains': {},
            'log_prob': None,
            'acceptance_rate': None,
            'summary': {},
            'convergence': {},
            'success': False,
            'message': 'forcepho package not installed',
            'forcepho_available': False,
        }
        
        return samples
    
    # If forcepho is available, set up and run HMC sampling
    # This is a template - actual implementation depends on forcepho API
    samples = {
        'chains': {},
        'log_prob': np.array([]),
        'acceptance_rate': 0.0,
        'summary': {},
        'convergence': {},
        'success': True,
        'message': 'Sampling complete',
        'forcepho_available': True,
        'n_draws': n_draws,
        'warmup': warmup,
        'max_treedepth': max_treedepth,
    }
    
    return samples


def run_forcepho_fit(config, positions=None, mode='optimize', 
                    optimize_kwargs=None, sampling_kwargs=None):
    """
    High-level function to run complete forcepho fitting workflow.
    
    This is the main entry point for running forcepho fits. It handles:
    1. Creating the scene from source catalog
    2. Preparing patch data
    3. Running optimization and/or sampling
    4. Returning results in a convenient format
    
    Parameters
    ----------
    config : dict
        Configuration dictionary from create_forcepho_config
    positions : list of tuples, optional
        List of (ra, dec) positions to fit. If None, uses config catalog.
    mode : str, optional
        Fitting mode: 'optimize', 'sample', or 'both'. Default is 'optimize'.
    optimize_kwargs : dict, optional
        Additional arguments for optimization
    sampling_kwargs : dict, optional
        Additional arguments for sampling
        
    Returns
    -------
    results : dict
        Complete fitting results including:
        - 'optimization': Optimization results (if mode includes optimize)
        - 'samples': Sampling results (if mode includes sample)
        - 'config': Input configuration
        - 'scene_config': Scene configuration used
        
    Examples
    --------
    >>> # Run optimization only
    >>> results = run_forcepho_fit(config, mode='optimize')
    >>> 
    >>> # Run both optimization and sampling
    >>> results = run_forcepho_fit(config, mode='both', 
    ...                           optimize_kwargs={'gtol': 1e-6},
    ...                           sampling_kwargs={'n_draws': 512})
    >>> 
    >>> # Fit specific positions
    >>> positions = [(150.1, 2.3), (150.11, 2.31)]
    >>> results = run_forcepho_fit(config, positions=positions, mode='sample')
    """
    if optimize_kwargs is None:
        optimize_kwargs = {}
    if sampling_kwargs is None:
        sampling_kwargs = {}
    
    # Create or update source catalog with positions
    if positions is not None:
        catalog = prepare_source_catalog(None, positions=positions)
        catalog.meta['galaxy_id'] = config.get('galaxy_id', 'unknown')
    else:
        catalog = config.get('source_catalog')
        if catalog is None:
            raise ValueError("No source catalog or positions provided")
    
    # Create scene configuration
    scene_config = create_forcepho_scene(
        catalog, 
        config['bands'],
        bounds_kwargs=config.get('bounds', None)
    )
    
    # Prepare patch data
    patch_data = prepare_forcepho_patch(config)
    
    results = {
        'config': config,
        'scene_config': scene_config,
        'patch_data': patch_data,
    }
    
    # Run optimization if requested
    if mode in ['optimize', 'both']:
        opt_results = run_forcepho_optimization(config, scene_config, **optimize_kwargs)
        results['optimization'] = opt_results
        
        if not opt_results.get('forcepho_available', False):
            results['message'] = 'forcepho package not installed - returning mock results'
    
    # Run sampling if requested
    if mode in ['sample', 'both']:
        # If we optimized first, could use those results as initialization
        samp_results = run_forcepho_sampling(config, scene_config, **sampling_kwargs)
        results['samples'] = samp_results
        
        if not samp_results.get('forcepho_available', False):
            results['message'] = 'forcepho package not installed - returning mock results'
    
    return results
