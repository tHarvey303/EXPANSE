# forcepho Integration for EXPANSE

This module provides utilities for preparing and running forcepho on ResolvedGalaxy objects.

## Overview

forcepho (https://forcepho.readthedocs.io/) is a tool for pixel-level forced photometry. This module helps prepare the required input files from EXPANSE's ResolvedGalaxy data structure.

## Usage

### Basic Usage

```python
from EXPANSE import ResolvedGalaxy

# Load a galaxy
galaxy = ResolvedGalaxy.init_from_h5("path/to/galaxy.h5")

# Prepare forcepho inputs
config, file_paths = galaxy.run_forcepho()
```

### Custom Source Positions

```python
# Prepare inputs for specific positions
positions = [(150.1, 2.3), (150.11, 2.31)]  # RA, Dec in degrees
config, file_paths = galaxy.run_forcepho(positions=positions)
```

### Custom Bands and PSF Type

```python
# Use specific bands and PSF type
config, file_paths = galaxy.run_forcepho(
    bands=["F277W", "F356W", "F444W"],
    psf_type="webbpsf",
    output_dir="my_forcepho_output"
)
```

## Functions

### prepare_pixel_data
Extract pixel data and error maps from ResolvedGalaxy.

### prepare_psf_data
Extract PSF models from ResolvedGalaxy.

### prepare_source_catalog
Create a source catalog with positions.

### prepare_wcs_info
Extract WCS information for coordinate transformations.

### prepare_photometric_properties
Extract zero points, pixel scales, and other photometric properties.

### create_forcepho_config
Create a complete configuration dictionary with all required inputs.

### save_forcepho_inputs
Save all inputs to FITS files for use with forcepho.

### validate_forcepho_inputs
Validate that all required inputs are present and correctly formatted.

## Output Structure

When `save_inputs=True`, the module creates the following directory structure:

```
forcepho_output_{galaxy_id}/
├── F277W_image.fits
├── F277W_error.fits
├── F277W_psf.fits
├── F356W_image.fits
├── F356W_error.fits
├── F356W_psf.fits
├── F444W_image.fits
├── F444W_error.fits
├── F444W_psf.fits
├── source_catalog.fits
└── wcs.fits
```

## Configuration Dictionary

The `create_forcepho_config` function returns a dictionary with:

- `galaxy_id`: Galaxy identifier
- `bands`: List of bands included
- `pixel_data`: Dictionary of image arrays by band
- `error_data`: Dictionary of uncertainty arrays by band
- `psf_data`: Dictionary of PSF arrays by band
- `psf_meta`: Dictionary of PSF metadata by band
- `source_catalog`: Astropy Table with source positions
- `wcs`: WCS object for coordinate transformations
- `properties`: Photometric properties (zero points, pixel scales, units)
- `output_dir`: Directory for output files
- `cutout_size`: Size of image cutouts
- `sky_coord`: Galaxy sky coordinates

## Requirements

The forcepho integration requires the following ResolvedGalaxy attributes:

- Image cutouts (`phot_imgs`)
- Error maps (`rms_err_imgs`)
- PSF models (`psfs`)
- Sky coordinates (`sky_coord`)
- Photometric zero points (`im_zps`)
- Pixel scales (`im_pixel_scales`)

Optional but recommended:
- PSF-matched data (`psf_matched_data`, `psf_matched_rms_err`)
- WCS headers (`phot_img_headers`)

## Notes

- This module focuses on preparing the input files for forcepho
- Actual forcepho execution should be done using the forcepho package
- PSF-matched data is used by default when available
- Source positions default to the galaxy center if not specified
