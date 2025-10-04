# forcepho Integration for EXPANSE

This module provides utilities for preparing and running forcepho on ResolvedGalaxy objects.

## Overview

forcepho (https://forcepho.readthedocs.io/) is a tool for pixel-level forced photometry. This module helps prepare the required input files from EXPANSE's ResolvedGalaxy data structure and provides functions to run forcepho optimization and sampling.

## Usage

### Basic Usage - Prepare Inputs Only

```python
from EXPANSE import ResolvedGalaxy

# Load a galaxy
galaxy = ResolvedGalaxy.init_from_h5("path/to/galaxy.h5")

# Prepare forcepho inputs (returns 3 values now)
config, file_paths, _ = galaxy.run_forcepho()
```

### Run Forcepho Optimization

```python
# Prepare inputs and run optimization
config, file_paths, results = galaxy.run_forcepho(run_fit=True, fit_mode='optimize')

# Check results
if results['optimization']['success']:
    print("Optimization successful!")
    print(f"Final parameters: {results['optimization']['parameters']}")
```

### Run Forcepho Sampling (HMC)

```python
# Run HMC sampling
config, file_paths, results = galaxy.run_forcepho(
    run_fit=True,
    fit_mode='sample',
    sampling_kwargs={'n_draws': 512, 'warmup': 256}
)

# Check sampling results
if 'samples' in results:
    print(f"Generated {results['samples']['n_draws']} samples")
```

### Run Both Optimization and Sampling

```python
# First optimize, then sample
config, file_paths, results = galaxy.run_forcepho(
    run_fit=True,
    fit_mode='both',
    optimize_kwargs={'gtol': 1e-6},
    sampling_kwargs={'n_draws': 512}
)

# Access both sets of results
opt_params = results['optimization']['parameters']
samples = results['samples']['chains']
```

### Custom Source Positions

```python
# Fit multiple sources in the galaxy cutout
positions = [(150.1, 2.3), (150.11, 2.31)]  # RA, Dec in degrees
config, file_paths, results = galaxy.run_forcepho(
    positions=positions,
    run_fit=True,
    fit_mode='optimize'
)
```

### Custom Bands and PSF Type

```python
# Use specific bands and PSF type
config, file_paths, results = galaxy.run_forcepho(
    bands=["F277W", "F356W", "F444W"],
    psf_type="webbpsf",
    output_dir="my_forcepho_output",
    run_fit=True
)
```

## Functions

### Input Preparation Functions

#### prepare_pixel_data
Extract pixel data and error maps from ResolvedGalaxy.

#### prepare_psf_data
Extract PSF models from ResolvedGalaxy.

#### prepare_source_catalog
Create a source catalog with positions.

#### prepare_wcs_info
Extract WCS information for coordinate transformations.

#### prepare_photometric_properties
Extract zero points, pixel scales, and other photometric properties.

#### create_forcepho_config
Create a complete configuration dictionary with all required inputs.

#### save_forcepho_inputs
Save all inputs to FITS files for use with forcepho.

#### validate_forcepho_inputs
Validate that all required inputs are present and correctly formatted.

### Forcepho Fitting Functions

#### decompose_psf_to_gaussian_mixture
Decompose PSF into Gaussian mixture model for efficient convolution in forcepho.

#### create_forcepho_scene
Create a forcepho Scene configuration from source catalog.

#### prepare_forcepho_patch
Organize data into patch structure for forcepho fitting.

#### run_forcepho_optimization
Run BFGS optimization to find maximum likelihood parameters.

#### run_forcepho_sampling
Run Hamiltonian Monte Carlo sampling to explore posterior distribution.

#### run_forcepho_fit
High-level function to run complete forcepho fitting workflow.

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

## Fitting Results

When `run_fit=True`, the results dictionary contains:

### Optimization Results (`mode='optimize'` or `mode='both'`)
- `parameters`: Final optimized parameter values
- `uncertainties`: Parameter uncertainties (if linear_optimize=True)
- `log_likelihood`: Final log likelihood value
- `success`: Whether optimization converged
- `message`: Status message

### Sampling Results (`mode='sample'` or `mode='both'`)
- `chains`: Parameter chains for each variable
- `log_prob`: Log probability for each sample
- `acceptance_rate`: HMC acceptance rate
- `summary`: Summary statistics (mean, std, quantiles)
- `convergence`: Convergence diagnostics

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

- **Input preparation**: This module always prepares the input files for forcepho
- **Fitting**: Actual forcepho optimization/sampling requires the forcepho package to be installed
- **Mock results**: If forcepho is not installed, the fitting functions return mock results to show the expected structure
- **PSF-matched data**: Used by default when available
- **Source positions**: Default to the galaxy center if not specified
- **Gradients**: forcepho uses gradients for efficient optimization and HMC sampling

## Installation

To use the fitting functionality, install forcepho:

```bash
pip install forcepho
```

For GPU acceleration (requires CUDA):
```bash
pip install forcepho[cuda]
```
