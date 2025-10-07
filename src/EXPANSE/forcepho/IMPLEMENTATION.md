# forcepho Integration Implementation Summary

## Overview
Successfully implemented forcepho integration into the EXPANSE package, focusing on preparing required input files from ResolvedGalaxy objects as per the forcepho documentation (https://forcepho.readthedocs.io/).

## Files Created

### Core Module Files
1. **src/EXPANSE/forcepho/__init__.py** (28 lines)
   - Exports all public functions
   - Clean module interface

2. **src/EXPANSE/forcepho/forcepho_utils.py** (375 lines)
   - `prepare_pixel_data()` - Extracts image cutouts and error maps
   - `prepare_psf_data()` - Extracts PSF models
   - `prepare_source_catalog()` - Creates astropy Table with source positions
   - `prepare_wcs_info()` - Extracts/creates WCS for coordinate transformations
   - `prepare_photometric_properties()` - Collects zero points, pixel scales, units
   - `create_forcepho_config()` - Main function to generate complete configuration
   - `save_forcepho_inputs()` - Saves all inputs to FITS files
   - `validate_forcepho_inputs()` - Validates required inputs are present

### Documentation
3. **src/EXPANSE/forcepho/README.md** (124 lines)
   - Usage instructions
   - Function documentation
   - Output structure description
   - Requirements and notes

4. **examples/forcepho_example.py** (123 lines)
   - 6 comprehensive usage examples
   - Demonstrates all major features
   - Shows both simple and advanced usage

### Modified Files
5. **src/EXPANSE/ResolvedGalaxy.py**
   - Added `run_forcepho()` method (82 lines)
   - Integrated with existing patterns (similar to run_bagpipes, run_prospector)
   - Comprehensive docstring with examples

6. **src/EXPANSE/__init__.py**
   - Added `from . import forcepho` import

## Implementation Details

### Design Philosophy
- **Minimal changes**: Only added new functionality without modifying existing code
- **Pattern consistency**: Followed existing module structure (eazy, prospector, synthesizer)
- **Data preservation**: Uses existing ResolvedGalaxy attributes without modification
- **Flexible interface**: Supports both quick default usage and fine-grained control

### Key Features
1. **Automatic data extraction**: Pulls all required data from ResolvedGalaxy
2. **PSF handling**: Supports both 'star_stack' and 'webbpsf' PSF types
3. **Custom positions**: Allows user to specify source positions or use galaxy center
4. **Band selection**: Can process all bands or user-specified subset
5. **File generation**: Automatically creates FITS files in organized directory structure
6. **Validation**: Checks that all required inputs are present before processing
7. **Flexible output**: Can return data in memory or save to disk

### Output Structure
```
forcepho_output_{galaxy_id}/
├── {band}_image.fits      # Image cutout for each band
├── {band}_error.fits      # Uncertainty map for each band
├── {band}_psf.fits        # PSF model for each band
├── source_catalog.fits    # Source positions and metadata
└── wcs.fits              # WCS information
```

### ResolvedGalaxy Integration
The `run_forcepho()` method:
- Returns configuration dict and file paths
- Validates inputs automatically
- Supports all parameters: positions, bands, psf_type, output_dir, save_inputs
- Provides helpful error messages
- Can be used with or without saving files

## Usage Examples

### Basic Usage
```python
galaxy = ResolvedGalaxy.init_from_h5("galaxy.h5")
config, file_paths = galaxy.run_forcepho()
```

### Custom Positions
```python
positions = [(150.1, 2.3), (150.11, 2.31)]
config, file_paths = galaxy.run_forcepho(positions=positions)
```

### Advanced Configuration
```python
config, file_paths = galaxy.run_forcepho(
    bands=["F277W", "F356W", "F444W"],
    psf_type="webbpsf",
    output_dir="custom_output",
    save_inputs=True
)
```

## Technical Notes

### Dependencies Used
- `numpy` - Array operations
- `astropy.io.fits` - FITS file I/O
- `astropy.wcs.WCS` - Coordinate transformations
- `astropy.table.Table` - Catalog creation
- `os` - File/directory operations

### ResolvedGalaxy Attributes Used
Required:
- `phot_imgs` - Image cutouts
- `rms_err_imgs` - Uncertainty maps
- `bands` - Band names
- `sky_coord` - Galaxy coordinates
- `cutout_size` - Image dimensions
- `galaxy_id` - Identifier

Optional (used when available):
- `psf_matched_data` - PSF-matched images
- `psf_matched_rms_err` - PSF-matched errors
- `psfs` - PSF models
- `psfs_meta` - PSF metadata
- `phot_img_headers` - WCS headers
- `im_zps` - Zero points
- `im_pixel_scales` - Pixel scales
- `phot_pix_unit` - Flux units

## Validation

All code validated with:
- Python AST syntax checking
- Function export verification
- Integration with existing code verified
- Documentation completeness checked

## Future Enhancements (Optional)

Potential additions for future work:
1. Unit tests with mock data (requires test dependencies)
2. Direct forcepho execution wrapper (requires forcepho package)
3. Result parsing and storage in ResolvedGalaxy
4. Batch processing multiple galaxies
5. Integration with EXPANSE GUI

## Conclusion

The forcepho integration provides a clean, well-documented interface for preparing forcepho inputs from EXPANSE ResolvedGalaxy objects. The implementation follows established patterns in the codebase and provides both simple default usage and advanced customization options.
