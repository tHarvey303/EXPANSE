[![workflow](https://github.com/tharvey303/EXPANSE/actions/workflows/python-app.yml/badge.svg)](https://github.com/duncanaustin98/galfind/actions)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License: GPLv3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

# EXPANSE - EXtended Pixel-resolved ANalysis of SEDs

Python package to perform resolved SED fitting using publically availabe SED fitting tools (Bagpipes and Dense Basis currently, Prospector, Beagle, etc planned), and display results. 

## Features

1. Interfaces with PIXEDFit, Voronoi binning, or single pixel binning to perform spatially resolved analysis.
2. Portable file format - all information about a galaxy and SED fitting results are stored in an HDF5 file, allowing easy transfer, or the ability to run different parts of processing on different computers - e.g. create object on desktop, fit on a remote cluster and then analyse the results on a laptop.
3. SED fitting using Bagpipes and Dense Basis with full support for all options, including custom priors, custom SFHs, dust laws, etc. Can also fit with a single pixel, or with a single SED for the whole galaxy. Multi-threading fitting is supported. 
4. Tools to model PSFs, either from stacking stars, or directly from WebbPSF. Can PSF homogenize internally, or accept pre-homogenized imaging.
5. Interactive web-based viewer to display results of SED fitting, including interactive RGB imaging, segmentation map, binning maps, spatially resolved property maps (stellar mass density, SFR density etc), per bin/pixel SEDs, SFHs and corner plots, as well as interfacing with FITSMAP. Can also measure quantities directly from the photometry, such as Beta slopes, MUV, D4000 break etc and produce spatially resolved maps.
6. Testing with Simulated galaxies from hydro sims, or generated parametrically, using the Synthesizer package. Allows comparison of recovered parameters like stellar mass, dust, SFH etc with true values. Works with FLARES, SPHINX20 and other sims which can be loaded via the Synthesizer package.
7. Morphological fitting with pysersic, including multi-component models and complex priors. Pyautogalaxy and pygalfitm are partially supported.
8. Radial/annular analysis of property maps, including SED fitting, radial profiles of properties, and annular SED fitting.

## Planned Features

1. Direct conversion of cutouts with modelled morphology into components for Pandeia - e.g. to plan JWST observations.
2. More interactivity in the viewer - e.g. to select regions for further analysis - e.g. to measure the SED of a region, or to fit a model to it, based on drawing an aperture, or grouping pixels (implemented for EAZY SED Fitting.)
3. Additional SED fitting tools, and improvement of pixel-by-pixel case. Prospector is in progress, and Beagle is planned.
4. Better tools for MetaClass for population analysis - spatially resolved main sequence, outshining etc.
5. Furhter reionization estimates - maps of fesc, zi_ion etc
6. More radial/annular tools - annular SED fitting, radial profiles of properties etc.
7. Bug fixes, documentation and examples.

## Installation

To install, clone the repository and run `pip install .` in the root directory. For best results, also install our fork of Bagpipes from [here](https://github.com/tHarvey303/bagpipes). 

## Usage

Example scripts and notebooks are provided in the scripts directory. Better documentation is coming soon.

To lauch the viewer, run `expanse-viewer` in the terminal after installation.

## Citation

The paper describing EXPANSE is currently in preparation. In the meantime, please cite this repository if you use it in your work.