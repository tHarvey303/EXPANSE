[![workflow](https://github.com/tharvey303/EXPANSE/actions/workflows/python-app.yml/badge.svg)](https://github.com/tharvey303/galfind/actions)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License: GPLv3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

# EXPANSE - EXtended Pixel-resolved ANalysis of SEDs

EXPANSE is a Python package to perform resolved SED fitting using publically availabe SED fitting tools (Bagpipes, Prospector and Dense Basis currently, more planned), and display results. 

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

## Examples of GUI

Here are a couple of examples of some of the GUI features, including an interface to plot and compare the SED fitting results - SED plots, resolved parameter maps, SFH, corner plots.

![GUI 1](https://github.com/tHarvey303/EXPANSE/blob/master/src/EXPANSE/gui/examples/EXPANSE_1.png)

This example shows an example of the interactive interface for EAZY SED fitting, which lets you place manual apertures and regions and fit their photometry. 
![GUI 2](https://github.com/tHarvey303/EXPANSE/blob/master/src/EXPANSE/gui/examples/EXPANSE_2.png)

## Installation

The easiest way to install is from the PyPI repository. If you have pip installed just run 'pip install astro-expanse' and the code will be downloaded and installed.

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

## Citation

If you use EXPANSE in your work please cite ![this paper](https://ui.adsabs.harvard.edu/abs/2025MNRAS.542.2998H/abstract).

## Support

If you have a problem, find a bug or would like some advice please ![open an issue](https://github.com/tHarvey303/EXPANSE/issues/new/choose) or email me!  

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
