[![workflow](https://github.com/tharvey303/EXPANSE/actions/workflows/python-app.yml/badge.svg)](https://github.com/tharvey303/galfind/actions)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License: GPLv3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

# EXPANSE - EXtended Pixel-resolved ANalysis of SEDs
<img src="gfx/EXPANSE_LOGO.png"  align="right" alt="logo" width="200px"/>

### Overview

EXPANSE is a Python package to perform resolved SED fitting and display and analyze the results.

### Features

1. **Multiple SED Fitting Tools**: Interfaces with Bagpipes, Prospector, EAZY-py, Synference and Dense Basis for flexible SED fitting options.
2. **Portable File Format**: All information about a galaxy and SED fitting results are stored in an HDF5 file, allowing easy transfer and processing on different computers.
3. **Interactive Viewer**: A web-based viewer to display results of SED fitting, including interactive RGB imaging and spatially resolved property maps.
4. **PSF Modelling and Homogenization**: Tools to model PSFs from stacking stars or directly from WebbPSF, with options for internal homogenization.
5. **Morphological Fitting**: Supports multi-component models and complex priors using pysersic, with partial support for Pyautogalaxy and pygalfitm.
6. **Testing with Simulated Galaxies**: Allows comparison of recovered parameters with true values using simulated galaxies from hydro sims or the Synthesizer package.
7. **Radial/Annular Analysis**: Interfaces with PIXEDFit, Voronoi binning, or single pixel binning to perform spatially resolved analysis.
8. **Population Analysis**: Tools to analyze populations of galaxies, including spatially resolved main sequence and outshining effects.
9. **Photometric Inference**: Directly infer Beta slope, MUV, emission line EWs, UV slope, D4000 break, and other photometric properties directly from resolved photometry.
10. **Interactive Aperture Placement**: Place apertures and regions interactively in the viewer to extract photometry and fit SEDs. Run EAZY and Dense Basis fits directly in the viewer.

## Planned Features

1. Direct conversion of cutouts with modelled morphology into components for Pandeia - e.g. to plan JWST observations.
2. Better integration with spectroscopic, IFU and slit stepping observations.
3. Incorporating de-lensing models to work with lensed galaxies. 
4. Bug fixes, documentation and examples.

## Quickstart

```python

from EXPANSE import ResolvedGalaxy

galaxy = ResolvedGalaxy.init('JOF_16.h5')

```

To create a new galaxy file from your own imaging data, see this [notebook](examples/intro.ipynb) for a full example.

## GUI Examples

Here are a couple of examples of the GUI features, including an interface to plot and compare the SED fitting results.

![GUI 1](https://github.com/tHarvey303/EXPANSE/blob/master/src/EXPANSE/gui/examples/EXPANSE_1.png)

This example shows an example of the interactive interface for EAZY SED fitting, which lets you place manual apertures and regions and fit their photometry with EAZY-py.
![GUI 2](https://github.com/tHarvey303/EXPANSE/blob/master/src/EXPANSE/gui/examples/EXPANSE_2.png)

To launch the viewer, run `expanse-viewer` in the terminal after installation.

## Installation

The easiest way to install is from the PyPI repository. If you have pip installed just run:

```bash
pip install astro-expanse
```

If you plan to modify or edit the code, you may prefer a manual install. To manually install, clone the repository:
```bash
git clone https://github.com/tHarvey303/EXPANSE.git
cd EXPANSE
pip install -e .
```

If you plan to use Bagpipes best results will be with my fork from [here](https://github.com/tHarvey303/bagpipes). 

```bash

pip install git+https://github.com/aabdurrouf/piXedfit@main
pip install git+https://github.com/tHarvey303/bagpipes/

```

## Examples

There is a notebook (examples/intro.ipynb) which goes over the basic features of EXPANSE from the beginning of loading in the data. All the plots for the EXPANSE paper are available in scripts/outshining_paper_figure.ipynb, which show more advanced usage. The examples/documentation are a work in progress.

Please get in touch if you're interested in using the code, I'm happy to chat!

To lauch the viewer, run `expanse-viewer` in the terminal after installation.


## Support

If you have a problem, find a bug or would like some advice please ![open an issue](https://github.com/tHarvey303/EXPANSE/issues/new/choose) or email me!  

## Citation

If you use EXPANSE in your work please cite ![this paper](https://ui.adsabs.harvard.edu/abs/2025MNRAS.542.2998H/abstract).

```
@ARTICLE{2025MNRAS.542.2998H,
       author = {{Harvey}, Thomas and {Conselice}, Christopher J. and {Adams}, Nathan J. and {Austin}, Duncan and {Li}, Qiong and {Rusakov}, Vadim and {Westcott}, Lewi and {Goolsby}, Caio M. and {Lovell}, Christopher C. and {Cochrane}, Rachel K. and {Vijayan}, Aswin P. and {Trussler}, James},
        title = "{Behind the spotlight: a systematic assessment of outshining using NIRCam medium bands in the JADES Origins Field}",
      journal = {\mnras},
     keywords = {galaxies: evolution, galaxies: high-redshift, galaxies: photometry, galaxies: star formation, galaxies: stellar content, Astrophysics of Galaxies},
         year = 2025,
        month = oct,
       volume = {542},
       number = {4},
        pages = {2998-3027},
          doi = {10.1093/mnras/staf1396},
archivePrefix = {arXiv},
       eprint = {2504.05244},
 primaryClass = {astro-ph.GA},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2025MNRAS.542.2998H},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}

```
