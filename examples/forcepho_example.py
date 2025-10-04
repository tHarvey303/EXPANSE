"""
Example script demonstrating forcepho integration with EXPANSE.

This script shows how to prepare forcepho inputs from a ResolvedGalaxy object
and run forcepho optimization and sampling.
"""

from EXPANSE import ResolvedGalaxy

# Example 1: Basic usage - prepare inputs only
# ==============================================
# Load a galaxy from an h5 file
galaxy = ResolvedGalaxy.init_from_h5("path/to/galaxy.h5")

# Prepare forcepho inputs for the galaxy center
# Note: run_forcepho now returns 3 values (config, file_paths, fit_results)
# The third value is None when run_fit=False (default)
config, file_paths, _ = galaxy.run_forcepho()

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


# Example 2: Run forcepho optimization
# =====================================
# Prepare inputs and run optimization to find best-fit parameters
config, file_paths, results = galaxy.run_forcepho(
    run_fit=True,
    fit_mode='optimize'
)

# Check if optimization succeeded
if results and results.get('optimization'):
    opt_results = results['optimization']
    if opt_results['success']:
        print("\nOptimization successful!")
        print(f"Final log likelihood: {opt_results['log_likelihood']}")
        print(f"Optimized parameters: {opt_results['parameters']}")
    else:
        print(f"\nOptimization message: {opt_results['message']}")


# Example 3: Run forcepho sampling (HMC)
# =======================================
# Run Hamiltonian Monte Carlo sampling to explore posterior
config, file_paths, results = galaxy.run_forcepho(
    run_fit=True,
    fit_mode='sample',
    sampling_kwargs={
        'n_draws': 512,
        'warmup': 256,
        'max_treedepth': 9,
    }
)

# Check sampling results
if results and results.get('samples'):
    samples = results['samples']
    print(f"\nGenerated {samples['n_draws']} samples")
    print(f"Acceptance rate: {samples['acceptance_rate']}")
    if samples.get('summary'):
        print("Parameter summaries:")
        for param, stats in samples['summary'].items():
            print(f"  {param}: {stats}")


# Example 4: Run both optimization and sampling
# ==============================================
# First optimize to find MAP, then sample around it
config, file_paths, results = galaxy.run_forcepho(
    run_fit=True,
    fit_mode='both',
    optimize_kwargs={'gtol': 1e-6, 'linear_optimize': True},
    sampling_kwargs={'n_draws': 512, 'warmup': 256}
)

# Access both optimization and sampling results
if results:
    if 'optimization' in results:
        print("\nOptimization results available")
        print(f"  Parameters: {results['optimization']['parameters']}")
        if 'uncertainties' in results['optimization']:
            print(f"  Uncertainties: {results['optimization']['uncertainties']}")
    
    if 'samples' in results:
        print("\nSampling results available")
        print(f"  Number of samples: {results['samples']['n_draws']}")


# Example 5: Multiple source positions
# =====================================
# Fit multiple sources in the galaxy cutout
positions = [
    (150.1, 2.3),     # RA, Dec in degrees for source 1
    (150.11, 2.31),   # RA, Dec in degrees for source 2
]

config, file_paths, results = galaxy.run_forcepho(
    positions=positions,
    run_fit=True,
    fit_mode='optimize'
)

# The source catalog now contains multiple sources
print(f"\nNumber of sources: {len(config['source_catalog'])}")
if results and results.get('scene_config'):
    print(f"Scene has {len(results['scene_config']['sources'])} sources")


# Example 6: Custom bands and PSF type with fitting
# ==================================================
# Use only specific bands and run optimization
config, file_paths, results = galaxy.run_forcepho(
    bands=["F277W", "F356W", "F444W"],  # Only these bands
    psf_type="webbpsf",                  # Use WebbPSF models
    psf_matched=True,                     # Use PSF-matched data
    output_dir="forcepho_output_custom", # Custom output directory
    run_fit=True,                         # Run fitting
    fit_mode='optimize',                  # Optimization only
    optimize_kwargs={'use_gradients': True, 'gtol': 1e-5}
)


# Example 7: Access data without saving files
# ============================================
# Get the configuration without saving FITS files, and run fitting
config, _, results = galaxy.run_forcepho(
    save_inputs=False,
    run_fit=True,
    fit_mode='optimize'
)

# Access the data directly from the config dictionary
pixel_data = config['pixel_data']  # Dict with band: image array
error_data = config['error_data']  # Dict with band: error array
psf_data = config['psf_data']      # Dict with band: PSF array
wcs = config['wcs']                # WCS object

# Example: Get data for F444W band
if 'F444W' in pixel_data:
    f444w_image = pixel_data['F444W']
    f444w_error = error_data['F444W']
    f444w_psf = psf_data['F444W']
    
    print(f"\nF444W image shape: {f444w_image.shape}")
    print(f"F444W error shape: {f444w_error.shape}")
    print(f"F444W PSF shape: {f444w_psf.shape}")


# Example 8: Using lower-level functions directly
# ================================================
from EXPANSE.forcepho import (
    prepare_pixel_data,
    prepare_psf_data,
    prepare_source_catalog,
    create_forcepho_scene,
    run_forcepho_fit,
)

# Prepare components individually
pixel_data, error_data = prepare_pixel_data(galaxy, bands=["F277W", "F356W"])
psf_data, psf_meta = prepare_psf_data(galaxy, psf_type="star_stack")
catalog = prepare_source_catalog(galaxy, positions=[(150.1, 2.3)])

# Create scene configuration
scene_config = create_forcepho_scene(catalog, bands=["F277W", "F356W"])

# Run fitting with the prepared configuration
fit_results = run_forcepho_fit(
    config,
    positions=[(150.1, 2.3)],
    mode='both',
    optimize_kwargs={'gtol': 1e-6},
    sampling_kwargs={'n_draws': 256}
)

print("\nFit completed")
if fit_results.get('optimization'):
    print(f"  Optimization success: {fit_results['optimization']['success']}")
if fit_results.get('samples'):
    print(f"  Samples generated: {fit_results['samples']['n_draws']}")


# Example 9: PSF decomposition
# =============================
from EXPANSE.forcepho import decompose_psf_to_gaussian_mixture

# Get PSF for a band
if 'F444W' in config['psf_data']:
    psf_array = config['psf_data']['F444W']
    
    # Decompose into Gaussian mixture
    mixture_params = decompose_psf_to_gaussian_mixture(psf_array, n_gaussians=10)
    
    print(f"\nPSF decomposed into {mixture_params['n_gaussians']} Gaussians")
    print(f"Amplitudes: {mixture_params['amplitudes']}")

