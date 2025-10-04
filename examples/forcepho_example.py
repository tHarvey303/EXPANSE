"""
Example script demonstrating forcepho integration with EXPANSE.

This script shows how to prepare forcepho inputs from a ResolvedGalaxy object.
"""

from EXPANSE import ResolvedGalaxy

# Example 1: Basic usage with default settings
# ==============================================
# Load a galaxy from an h5 file
galaxy = ResolvedGalaxy.init_from_h5("path/to/galaxy.h5")

# Prepare forcepho inputs for the galaxy center
# This will create FITS files for all bands with:
# - Image cutouts
# - Error/uncertainty maps
# - PSF models
# - Source catalog with galaxy center position
# - WCS information
config, file_paths = galaxy.run_forcepho()

# The config dictionary contains all the data
print(f"Galaxy ID: {config['galaxy_id']}")
print(f"Bands: {config['bands']}")
print(f"Output directory: {config['output_dir']}")
print(f"Cutout size: {config['cutout_size']}")

# The file_paths dictionary contains paths to saved FITS files
print(f"\nSaved files:")
for band in file_paths['images']:
    print(f"  {band} image: {file_paths['images'][band]}")
    print(f"  {band} error: {file_paths['errors'][band]}")
    print(f"  {band} PSF: {file_paths['psfs'][band]}")
print(f"  Catalog: {file_paths['catalog']}")


# Example 2: Multiple source positions
# =====================================
# Fit multiple sources in the galaxy cutout
positions = [
    (150.1, 2.3),     # RA, Dec in degrees for source 1
    (150.11, 2.31),   # RA, Dec in degrees for source 2
]

config, file_paths = galaxy.run_forcepho(positions=positions)

# The source catalog now contains multiple sources
print(f"\nNumber of sources: {len(config['source_catalog'])}")


# Example 3: Custom bands and PSF type
# ====================================
# Use only specific bands and a different PSF type
config, file_paths = galaxy.run_forcepho(
    bands=["F277W", "F356W", "F444W"],  # Only these bands
    psf_type="webbpsf",                  # Use WebbPSF models
    psf_matched=True,                     # Use PSF-matched data
    output_dir="forcepho_output_custom"  # Custom output directory
)


# Example 4: Access data without saving files
# ============================================
# Get the configuration without saving FITS files
config, _ = galaxy.run_forcepho(save_inputs=False)

# Access the data directly from the config dictionary
pixel_data = config['pixel_data']  # Dict with band: image array
error_data = config['error_data']  # Dict with band: error array
psf_data = config['psf_data']      # Dict with band: PSF array
wcs = config['wcs']                # WCS object

# Example: Get data for F444W band
f444w_image = pixel_data['F444W']
f444w_error = error_data['F444W']
f444w_psf = psf_data['F444W']

print(f"\nF444W image shape: {f444w_image.shape}")
print(f"F444W error shape: {f444w_error.shape}")
print(f"F444W PSF shape: {f444w_psf.shape}")


# Example 5: Using individual utility functions
# =============================================
from EXPANSE.forcepho import (
    prepare_pixel_data,
    prepare_psf_data,
    prepare_source_catalog,
    prepare_wcs_info,
    validate_forcepho_inputs,
)

# Prepare components individually
pixel_data, error_data = prepare_pixel_data(galaxy, bands=["F277W", "F356W"])
psf_data, psf_meta = prepare_psf_data(galaxy, psf_type="star_stack")
catalog = prepare_source_catalog(galaxy, positions=[(150.1, 2.3)])
wcs = prepare_wcs_info(galaxy, band="F277W")

# Create custom configuration
custom_config = {
    'pixel_data': pixel_data,
    'error_data': error_data,
    'psf_data': psf_data,
    'source_catalog': catalog,
    'wcs': wcs,
}

# Validate the configuration
valid, missing = validate_forcepho_inputs(custom_config)
if valid:
    print("\n✓ All required inputs are present")
else:
    print(f"\n✗ Missing inputs: {missing}")


# Example 6: Save configuration to files
# ======================================
from EXPANSE.forcepho import save_forcepho_inputs

# Save the configuration to FITS files
file_paths = save_forcepho_inputs(config, output_dir="my_forcepho_run")
print(f"\nFiles saved to: my_forcepho_run/")
