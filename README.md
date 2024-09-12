# EXPANSE - EXtended Pixel-resolved ANalysis of SEDs

Python package to perform resolved SED fitting using publically availabe SED fitting tools (Bagpipes and Dense Basis currently, Prospector, Beagle, etc planned), and display results. 

## Features

1. Interfaces with PIXEDFit, Voronoi binning, or single pixel binning to perform spatially resolved analysis.
2. Portable file format - all information about a galaxy and SED fitting results are stored in an HDF5 file, allowing easy transfer, or the ability to run different parts of processing on different computers - e.g. create object on desktop, fit on a remote cluster and then analyse the results on a laptop.
3. Tools to model PDFs, either from stacking stars, or directly from WebbPSF. Can PSF homogenize internally, or accept pre-homogenized imaging.
4. Interactive web-based viewer to display results of SED fitting, including interactive RGB imaging, segmentation map, binning maps, spatially resolved property maps (stellar mass density, SFR density etc), per bin/pixel SEDs, SFHs and corner plots, as well as interfacing with FITSMAP. Can also measure quantities directly from the photometry, such as Beta slopes, MUV, D4000 break etc and produce spatially resolved maps.
5. Testing with Simulated galaxies from hydro sims, or generated parametrically, using the Synthesizer package. Allows comparison of recovered parameters like stellar mass, dust, SFH etc with true values.

## Planned Features

1. Interfacing with other sims - SPHINX is planned, other suggestions welcome.
2. Further improvements to Viewer.
3. Additional SED fitting tools, and improvement of pixel-by-pixel case.
4. Better tools for MetaClass for population analysis - spatially resolved main sequence, outshining etc.
5. Furhter reionization estimates - maps of fesc, zi_ion etc
7. Bug fixes, documentation and examples.
