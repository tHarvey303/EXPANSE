import panel as pn
import numpy as np
from panel.layout.gridstack import GridStack
from matplotlib.figure import Figure
import matplotlib as mpl
import xarray as xr
import holoviews as hv
import hvplot.xarray
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from ResolvedGalaxy import ResolvedGalaxy, MockResolvedGalaxy
import h5py
from io import BytesIO
import sys
import inspect
import matplotlib as mpl
from matplotlib.colors import Normalize
from astropy import units as u
import functools
import h5py as h5
import os
import subprocess
from astropy.wcs import WCS
from astropy.io import fits

sys.setrecursionlimit(100000)
import resource

# change bokeh logging level
import logging
logging.getLogger('bokeh').setLevel(logging.ERROR)
# Change logging level for panel
logging.getLogger('panel').setLevel(logging.ERROR)
# Supress boken warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

resource.setrlimit(resource.RLIMIT_STACK, [0x10000000, resource.RLIM_INFINITY])

sys.path.append('/usr/local/texlive/')

file_path = os.path.abspath(__file__)
if 'nvme' in file_path:
    computer = 'morgan'
elif 'Users' in file_path:
    computer = 'mac'
    mpl.use('macOsX')
    mpl.rcParams['text.usetex'] = True 

#sns.set_context("paper")
#plt.style.use('paper.mplstyle')

# panel serve ResolvedSEDApp.py --autoreload --port 5003

ACCENT = "goldenrod"
LOGO = "https://assets.holoviz.org/panel/tutorials/matplotlib-logo.png"

plotpipes_dir = 'pipes_scripts/'
run_dir = 'pipes/'
galaxies_dir = 'galaxies/'
cache_pipes = {}

pn.extension(sizing_mode="stretch_width", design='material')
pn.extension('gridstack')
pn.extension(notifications=True)
try:
    import plotly
    pn.extension('plotly')
except ImportError:
    pass

facecolor = '#f7f7f7'

MAX_SIZE_MB = 150

TOTAL_FIT_COLORS = {'TOTAL_BIN': 'red', 'MAG_AUTO': 'blue', 'MAG_APER': 'green', 'MAG_ISO': 'purple', 'MAG_APER_TOTAL': 'orange', 'MAG_BEST': 'cyan', 'RESOLVED': 'black'}

stream = hv.streams.Tap(transient=True)

def make_hashable(args):
    """Converts unhashable types to hashable types."""
    hashable_args = []
    for arg in args:
        #print(type(arg))
        if isinstance(arg, list):
            hashable_args.append(tuple(arg))
        elif isinstance(arg, str) or arg is None:
            hashable_args.append(arg)
        else:
            hashable_args.append('Placeholder')

    #print([type(i) for i in hashable_args])
    #sys.exit()
    hashable_args = tuple(hashable_args)

    return hashable_args

def cached_function(func):
    cache = {}

    def wrapper(*args):
        hashable_args = make_hashable(args)
        if hashable_args in cache:
            #print(f"Using cached result for {hashable_args}...")
            return cache[hashable_args]
        else:
            #print(f"Computing result for {hashable_args}...")
            result = func(*args)
            cache[hashable_args] = result
            return result

    return wrapper


@pn.cache
def get_h5(url):
    response = requests.get(url)
    return h5py.File(BytesIO(response.content), 'r')

def update_image(value):
    #global shown_bins
    
    #shown_bins = []
    multi_choice_bins.value = []
    bin_plot = plot_bins(value, 'nipy_spectral_r')
    bin_map.object = bin_plot
    stream.source = bin_map.object

def possible_runs_select(sed_fitting_tool):

    if sed_fitting_tool == None:
        return None 

    options = resolved_galaxy.sed_fitting_table.get(sed_fitting_tool, None)
    options_resolved = []
    options_aperture = []
    if options is None:
        options = ['No runs found']
        
    else:
        for option in options.keys():
            ids = options[option]['#ID']
            # Check if all IDs can be converted to int
            any_int = False
            all_int = True
            for id in ids:
                try:
                    int(id)
                    any_int = True
                except ValueError:
                    all_int = False
            if any_int:
                options_resolved.append(option)
            if not all_int:
                options_aperture.append(option)
    
    which_run_resolved.options = options_resolved
    which_run_aperture.options = options_aperture
    if len(options_resolved) > 0:
        which_run_resolved.value = options_resolved[0]
    if len(options_aperture) > 0:
        which_run_aperture.value = options_aperture[0]
    
    return pn.Column(which_run_resolved, which_run_aperture)

def update_sidebar(active_tab, sidebar):
    sidebar.clear()
    settings_sidebar = pn.Column(pn.layout.Divider(), "### Settings", name='settings_sidebar')

    if active_tab == 0:
        settings_sidebar.append('#### File Upload')
        settings_sidebar.append(file_input)
        settings_sidebar.append(psf_mode_select)
        settings_sidebar.append('#### RGB Image')
        settings_sidebar.append(red_select)
        settings_sidebar.append(green_select)
        settings_sidebar.append(blue_select)
        settings_sidebar.append(stretch_slider)
        settings_sidebar.append(q_slider)
   
    elif active_tab == 1:
        settings_sidebar.append('#### Pixel Binning')
        settings_sidebar.append(which_map)
        settings_sidebar.append(pn.Row(show_galaxy, show_kron))
        settings_sidebar.append(scale_alpha)
        settings_sidebar.append(choose_show_band)
        settings_sidebar.append('#### SED Fitting Tool')
        settings_sidebar.append(which_sed_fitter)
        settings_sidebar.append(pn.bind(possible_runs_select, which_sed_fitter.param.value, watch=True))
        settings_sidebar.append(pn.Row(upscale_select, show_sed_photometry))

        settings_sidebar.append('#### SED Plot Config')
        settings_sidebar.append(which_flux_unit)
        settings_sidebar.append(multi_choice_bins)
        settings_sidebar.append(total_fit_options)

        #pn.bind(update_image, which_map.param.value, watch=True)

    elif active_tab == 2:
        settings_sidebar.append('#### Pixel Binning')
        settings_sidebar.append(which_map)
        phot_prop_func()


    sidebar.append(settings_sidebar)


def handle_map_click(x, y, resolved_galaxy, cmap, which_map_param, which_sed_fitter_param, 
                    which_flux_unit_param, multi_choice_bins_param, which_run_aperture_param, which_run_resolved_param,
                    total_fit_options_param, param_property, show_sed_photometry_param, mode='sed'):
    #print('handel map click')
    use = True
    if x == None or y == None:
        use = False
    if use:
        if not (0 < x < resolved_galaxy.cutout_size and 0 < y < resolved_galaxy.cutout_size):
            use = False
    
    map = resolved_galaxy.pixedfit_map
    if use:
        # Need logic here to know which map to use
        rbin = map[int(np.ceil(y)), int(np.ceil(x))]
        if rbin not in multi_choice_bins_param:
            multi_choice_bins.value = multi_choice_bins_param + [rbin]
            print(f'Added bin. {rbin}')
    
    multi_choice_bins_param_safe = []
    for i in multi_choice_bins_param:
        if i == 'RESOLVED':
            multi_choice_bins_param_safe.append(i)
        elif i != np.nan:
            multi_choice_bins_param_safe.append(int(i))
        else:
            pass
    
    #yield pn.indicators.LoadingSpinner(size = 50, name = 'Loading...', value = True)
        
    if mode == 'sed':
        obj = plot_sed(resolved_galaxy, map, cmap, which_map_param, which_sed_fitter_param, which_flux_unit_param, multi_choice_bins_param_safe, which_run_aperture_param, which_run_resolved_param, total_fit_options_param, show_sed_photometry_param)
    if mode == 'sfh':
        obj = plot_sfh(resolved_galaxy, map, cmap, which_map_param, which_sed_fitter_param, multi_choice_bins_param_safe, which_run_aperture_param, which_run_resolved_param, total_fit_options_param)
    if mode == 'corner':
        obj = plot_corner(resolved_galaxy, map, cmap, which_map_param, which_sed_fitter_param, multi_choice_bins_param_safe, which_run_aperture_param, which_run_resolved_param, total_fit_options_param)
    if mode == 'pdf':
        obj = plot_bagpipes_pdf(resolved_galaxy, map, cmap, param_property, which_run_aperture_param, which_run_resolved_param, multi_choice_bins_param_safe, total_fit_options_param, which_sed_fitter_param)

    return obj 
    #except Exception as e:
    #    pn.state.notifications.error(f'Error: {e}', duration=2000)

def plot_rgb(resolved_galaxy, red, green, blue, scale, q, psf_mode):
    if psf_mode == 'Original':
        use_psf_matched = False
    elif psf_mode == 'PSF Matched':
        use_psf_matched = True
    rgb = resolved_galaxy.plot_lupton_rgb(red, green, blue, scale, q, use_psf_matched=use_psf_matched)
    # Flip y axis to match image orientation
    rgb = np.flipud(rgb)
    rgb_img = hv.RGB(rgb, bounds=(0, 0, resolved_galaxy.cutout_size, resolved_galaxy.cutout_size)).opts(xaxis=None, yaxis=None)

    any_band = resolved_galaxy.bands[0]
    # Add scale bar
    scale_bar_label = hv.Text(17.5, 10, '1\'', fontsize=15).opts(color='white')
    scale_bar_size = resolved_galaxy.im_pixel_scales[any_band].to(u.arcsec).value # 1 pixel in arcsec
    scale_bar_10as = hv.Rectangles([(3, 3, 3+1/scale_bar_size, 4)]).opts(color='white', line_color='white')

    
    rgb_img = rgb_img * scale_bar_label
    rgb_img = rgb_img * scale_bar_10as
    # Add PSF sized circle
    if use_psf_matched:
        circle = hv.Ellipse(7, 15, 5).opts(color='white', line_color='white')
        rgb_img = rgb_img * circle

    center = resolved_galaxy.cutout_size/2
    a, b, theta = resolved_galaxy.plot_kron_ellipse(None, None, return_params = True)
    kron = hv.Ellipse(center, center, (a, b), orientation=theta*np.pi/180).opts(color='white', line_color='white')
    rgb_img = rgb_img * kron

    return pn.pane.HoloViews(rgb_img, height=400, width=430)

@cached_function
def plot_sed(resolved_galaxy, map, cmap, which_map_param, which_sed_fitter_param, which_flux_unit_param, multi_choice_bins_param, which_run_aperture_param, which_run_resolved_param, total_fit_options_param, show_sed_photometry_param, x_unit = u.micron):

    fig, ax = plt.subplots(figsize=(6, 3), constrained_layout=True, facecolor=facecolor)
    
    if which_flux_unit_param == 'uJy':
        y_unit = u.uJy
    elif which_flux_unit_param == 'ABmag':
        y_unit = u.ABmag
    elif which_flux_unit_param == 'ergscma':
        y_unit = u.erg/u.s/u.cm**2/u.AA

    if len(resolved_galaxy.photometry_table) > 0:
        #return pn.pane.Markdown('No photometry table found')
        psf_type = resolved_galaxy.use_psf_type
        table = resolved_galaxy.photometry_table[psf_type][which_map_param]
    else:
        table = None
    bands = resolved_galaxy.bands

    resolved_galaxy.get_filter_wavs()
    wavs = resolved_galaxy.filter_wavs
    
    show_sed_photometry_param = True if show_sed_photometry_param == 'Show' else False
    
    '''
    for option in total_fit_options_param:
        if option == 'MAG_APER':
            aper_dict = resolved_galaxy.aperture_dict[str(0.32*u.arcsec)]
        elif option in table['ID']:
            aper_dict = table[table['ID'] == option]
            
    flux = aper_dict['flux'] * u.Jy
    wave = aper_dict['wave'] * u.AA
    flux_err = aper_dict['flux_err'] * u.Jy
    
    if y_unit == u.ABmag:
        # Assymmetric error bars
        #[2.5*np.log10(flux/(flux-flux_err)), 2.5 * np.log10(1+flux_err/flux)]
        yerr = [2.5*abs(np.log10(flux.value/(flux.value - flux_err.value))), 2.5*np.log10(1 + flux_err.value/flux.value)]

    else:
        yerr = flux_err.to(y_unit, equivalencies = u.spectral_density(wave)).value

    ax.errorbar(wave.to(x_unit).value, flux.to(y_unit, equivalencies = u.spectral_density(wave)).value, yerr=yerr, fmt='o', linestyle='none', color='red', label=r"$ 0.32''$", mec = 'black')
    '''
    # pad legend down and left
    list_of_markers = ['o', 's', 'v', '^', '<', '>', '1', '2', '3', '4', '8', 'p', 'P', '*', 'h', 'H', '+', 'x', 'X', 'D', 'd', '|', '_']
    if show_sed_photometry_param:
        list_of_markers = list_of_markers[1:]
        
    colors = []
    colors_bin = []
    colors_total = []
    cmap = cm.get_cmap(cmap)

    if table is not None:
        for bin_pos, rbin in enumerate(multi_choice_bins_param + total_fit_options_param):
            
            if type(rbin) == str:
                color = TOTAL_FIT_COLORS[rbin]
                colors_total.append(color)
            else:
                # Get color between min and max of map
                color = cmap(Normalize(vmin=np.nanmin(map), vmax=np.nanmax(map))(rbin))
                
                colors_bin.append(color)

            colors.append(color)
            mask = np.array([True if str(i) == str(rbin) else False for i in table['ID']])
            table_row = table[mask]
            
            if len(table_row) == 0:
                print(f'No SED bins for {rbin}, psf_type {psf_type} and which_map_param {which_map_param}')
                continue

            #assert str(table_row['ID']) == str(rbin), f'Error: ID {str(table_row["ID"])} != {str(rbin)}'
            # loop through markers
            if bin_pos > len(list_of_markers) - 1:
                bin_pos = bin_pos % len(list_of_markers)

            marker = list_of_markers[bin_pos]

            for pos, band in enumerate(resolved_galaxy.bands):
                wav = wavs[band]
                flux = table_row[band]
                flux_err = table_row[f"{band}_err"]
                
                if not flux.isscalar:
                    flux = flux[0]
                    flux_err = flux_err[0]

                #if len(flux) == 0:
                #    continue
                if flux > 0:
                    if flux_err/flux < 0.1:
                        flux_err = 0.1 * flux

            
                if y_unit == u.ABmag:
                    # Assymmetric error bars - given - + because of inverse axis
                    yerr = [[np.abs(2.5*abs(np.log10(flux.value/(flux.value - flux_err.value))))], [np.abs(2.5*np.log10(1 + flux_err.value/flux.value))]]
                   
                    if np.isnan(yerr[0][0]):
                        print('fixed')
                        yerr[0][0] = 2 # Placeholder for big error
                    
                    # Swap to get correct order
                    yerr[0][0], yerr[1][0] = yerr[1][0], yerr[0][0]
                else:
                    yerr = flux_err.to(y_unit, equivalencies = u.spectral_density(wav)).value
                lab = int(rbin) if type(rbin) == float else rbin
                lab = lab if pos == 0 else ''
                #print(band)
                #print(flux.to(y_unit, equivalencies = u.spectral_density(wav)).value, yerr)
                ax.errorbar(wav.to(x_unit).value, flux.to(y_unit, equivalencies = u.spectral_density(wav)).value, yerr=yerr, fmt=marker, linestyle='none', color=color, label = lab, markeredgecolor = 'black' if show_sed_photometry_param else color)
    
    ax.legend(loc='upper left', frameon=False)    
    
    if y_unit == u.ABmag:
        ax.invert_yaxis()
    ax.set_xlabel(rf"$\rm{{Wavelength}}$ ({x_unit:latex})", fontsize='large')
    ax.set_ylabel(rf"$\rm{{Flux \ Density}}$ ({y_unit:latex})", fontsize='large')
    ax.set_xlim(ax.get_xlim())
    ax.set_ylim(ax.get_ylim())

    
    if which_sed_fitter_param == 'bagpipes' and (which_run_aperture_param != None or which_run_resolved_param != None):
        # Check if bagpipes run exists
        # Plot resolved fits
        if hasattr(resolved_galaxy, 'sed_fitting_table') and 'bagpipes' in resolved_galaxy.sed_fitting_table.keys() and which_run_resolved_param in resolved_galaxy.sed_fitting_table['bagpipes'].keys():
            if resolved_galaxy.galaxy_id not in cache_pipes.keys():
                cache_pipes[resolved_galaxy.galaxy_id] = {}
            if which_run_resolved_param not in cache_pipes[resolved_galaxy.galaxy_id].keys():
                cache_pipes[resolved_galaxy.galaxy_id][which_run_resolved_param] = {}
                
            cache = cache_pipes[resolved_galaxy.galaxy_id].get(which_run_resolved_param)

            fig, cache == resolved_galaxy.plot_bagpipes_fit(which_run_resolved_param, ax, fig, bins_to_show=multi_choice_bins_param, marker_colors=colors_bin, wav_units=x_unit, flux_units=y_unit, cache=cache, show_photometry=show_sed_photometry_param)
            cache_pipes[resolved_galaxy.galaxy_id][which_run_resolved_param] = cache

        # Plot aperture fits
        if hasattr(resolved_galaxy, 'sed_fitting_table') and 'bagpipes' in resolved_galaxy.sed_fitting_table.keys() and which_run_aperture_param in resolved_galaxy.sed_fitting_table['bagpipes'].keys():
            if resolved_galaxy.galaxy_id not in cache_pipes.keys():
                cache_pipes[resolved_galaxy.galaxy_id] = {}

            if which_run_aperture_param not in cache_pipes[resolved_galaxy.galaxy_id].keys():
                cache_pipes[resolved_galaxy.galaxy_id][which_run_aperture_param] = {}
                
            cache = cache_pipes[resolved_galaxy.galaxy_id].get(which_run_aperture_param)
            fig, cache == resolved_galaxy.plot_bagpipes_fit(which_run_aperture_param, ax, fig, bins_to_show=total_fit_options_param, marker_colors=colors_total, wav_units=x_unit, flux_units=y_unit, cache=cache, show_photometry=show_sed_photometry_param)

            cache_pipes[resolved_galaxy.galaxy_id][which_run_aperture_param] = cache
        


    plt.close(fig)
    return pn.pane.Matplotlib(fig, dpi=144, tight=True, format="svg", sizing_mode="scale_both")
    #return False

@cached_function     
def plot_bagpipes_pdf(resolved_galaxy, map, cmap, param_property, which_run_aperture_param, which_run_resolved_param, multi_choice_bins_param, total_fit_options_param, which_sed_fitter_param, run_dir = run_dir):
    cmap = cm.get_cmap(cmap)
    #print(multi_choice_bins_param + total_fit_options_param)
    # Ignore resolved
    if 'RESOLVED' in multi_choice_bins_param:
        multi_choice_bins_param.remove('RESOLVED')

    colors_bins = [cmap(Normalize(vmin=np.nanmin(map), vmax=np.nanmax(map))(rbin)) for pos, rbin in enumerate(multi_choice_bins_param)] 
    colors_total = [TOTAL_FIT_COLORS[rbin] for rbin in total_fit_options_param]
   
    if which_sed_fitter_param == 'bagpipes' and (which_run_aperture_param != None or which_run_resolved_param != None):
        # Check if bagpipes run exists
        fig = None
            # Plot resolved fits
        if hasattr(resolved_galaxy, 'sed_fitting_table') and 'bagpipes' in resolved_galaxy.sed_fitting_table.keys() and which_run_resolved_param in resolved_galaxy.sed_fitting_table['bagpipes'].keys():
            if resolved_galaxy.galaxy_id not in cache_pipes.keys():
                cache_pipes[resolved_galaxy.galaxy_id] = {}

            if which_run_resolved_param not in cache_pipes[resolved_galaxy.galaxy_id].keys():
                cache_pipes[resolved_galaxy.galaxy_id][which_run_resolved_param] = {}
                
            cache = cache_pipes[resolved_galaxy.galaxy_id].get(which_run_resolved_param)
            
            fig, cache = resolved_galaxy.plot_bagpipes_component_comparison(parameter = param_property, run_name=which_run_resolved_param, 
                                        bins_to_show = multi_choice_bins_param, save=False, run_dir = run_dir,
                                        facecolor=facecolor, cache = cache, colors=colors_bins)
            
            ax = fig.get_axes()[0]
            cache_pipes[resolved_galaxy.galaxy_id][which_run_resolved_param] = cache
            
        else:
            fig = None
            ax = None
        # Plot aperture fits
        if hasattr(resolved_galaxy, 'sed_fitting_table') and 'bagpipes' in resolved_galaxy.sed_fitting_table.keys() and which_run_aperture_param in resolved_galaxy.sed_fitting_table['bagpipes'].keys():
            if resolved_galaxy.galaxy_id not in cache_pipes.keys():
                cache_pipes[resolved_galaxy.galaxy_id] = {}
            if which_run_aperture_param not in cache_pipes[resolved_galaxy.galaxy_id].keys():
                cache_pipes[resolved_galaxy.galaxy_id][which_run_aperture_param] = {}
                
            cache = cache_pipes[resolved_galaxy.galaxy_id].get(which_run_aperture_param)

            
            fig, cache = resolved_galaxy.plot_bagpipes_component_comparison(parameter = param_property, run_name=which_run_aperture_param,
                                        bins_to_show = total_fit_options_param, save=False, run_dir = run_dir,
                                        facecolor=facecolor, cache = cache, colors=colors_total, fig = fig, axes = ax)

            cache_pipes[resolved_galaxy.galaxy_id][which_run_aperture_param] = cache

            return pn.pane.Matplotlib(fig, dpi=144, tight=True, format="svg", sizing_mode="scale_both", min_width = 500, min_height = 400)
        
        else:
            return pn.pane.Markdown('No Bagpipes results found.')
    else:
        return pn.pane.Markdown(f'Not implemented for {which_sed_fitter_param}.')


def plot_bins(bin_type, cmap, scale_alpha, show_galaxy, show_kron, psf_matched, band = 'F444W'):
    array = getattr(resolved_galaxy, f"{bin_type}_map")
    if array is None:
        # Empty plot
        return None
    array[array == 0] = np.nan
    dimensions = np.linspace(0, resolved_galaxy.cutout_size, resolved_galaxy.cutout_size)
    im_array = xr.DataArray(array, dims=['y', 'x'], name=f'{bin_type} bins', coords={'x': dimensions, 'y': dimensions})
    hvplot = im_array.hvplot('x','y').opts(cmap=cmap, xaxis=None, yaxis=None, clabel=f'{bin_type} Bin', alpha=scale_alpha)
    opts = np.unique(array)
    multi_choice_bins.options = ['RESOLVED']+list(opts[~np.isnan(opts)])

    if show_kron:
        center = resolved_galaxy.cutout_size/2
        a, b, theta = resolved_galaxy.plot_kron_ellipse(None, None, return_params = True)
        kron = hv.Ellipse(center, center, (a, b), orientation=theta*np.pi/180).opts(color='red', line_color='red')
        hvplot = hvplot * kron

    if show_galaxy:
        if psf_matched:
            data = resolved_galaxy.psf_matched_data['star_stack'][band]
        else:
            data = resolved_galaxy.phot_imgs[band]
        data = np.flipud(data)
        lower, upper = np.nanpercentile(data, [5, 99])
        if lower <= 0:
            lower = upper/1000

        new_hvplot = hv.Image(data, bounds=(0, 0, resolved_galaxy.cutout_size, resolved_galaxy.cutout_size)).opts(xaxis=None, yaxis=None, cmap = 'gray', cnorm='log', clim=(lower, upper))
        #data_array = xr.DataArray(data, dims=['y', 'x'], name='Galaxy Image', coords={'x': dimensions, 'y': dimensions}))
        #new_hvplot = data_array.hvplot('x', 'y').opts(cmap='gray', xaxis=None, yaxis=None)
        hvplot = new_hvplot * hvplot 

    bin_map = pn.pane.HoloViews(hvplot, height=300, width = 400, aspect_ratio=1)
    stream.source = bin_map.object

    return bin_map
     

@cached_function
def plot_sfh(resolved_galaxy, map, cmap, which_map_param, which_sed_fitter_param, multi_choice_bins_param, which_run_aperture_param, which_run_resolved_param, total_fit_options_param, x_unit = 'Gyr', facecolor='#f7f7f7'):
    
    cmap = cm.get_cmap(cmap)
    remove = False
    if 'RESOLVED' in multi_choice_bins_param:
        # get index
        index = multi_choice_bins_param.index('RESOLVED')
        multi_choice_bins_param.remove('RESOLVED')
        remove = True

    colors_bins = [cmap(Normalize(vmin=np.nanmin(map), vmax=np.nanmax(map))(rbin)) for pos, rbin in enumerate(multi_choice_bins_param)]
    # add black for resolved
    
    colors_total = [TOTAL_FIT_COLORS[rbin] for rbin in total_fit_options_param]
    # FIX HERE
    if hasattr(resolved_galaxy, 'sed_fitting_table') and 'bagpipes' in resolved_galaxy.sed_fitting_table.keys() and (which_run_aperture_param in resolved_galaxy.sed_fitting_table['bagpipes'].keys() or which_run_resolved_param in resolved_galaxy.sed_fitting_table['bagpipes'].keys()):
        if which_sed_fitter_param == 'bagpipes':
            if resolved_galaxy.galaxy_id not in cache_pipes.keys():
                cache_pipes[resolved_galaxy.galaxy_id] = {}

            if which_run_aperture_param not in cache_pipes[resolved_galaxy.galaxy_id].keys() and which_run_aperture_param is not None:
                cache_pipes[resolved_galaxy.galaxy_id][which_run_aperture_param] = {}

            if which_run_resolved_param not in cache_pipes[resolved_galaxy.galaxy_id].keys() and which_run_resolved_param is not None:
                cache_pipes[resolved_galaxy.galaxy_id][which_run_resolved_param] = {}

            if which_run_resolved_param is not None:
                

                bins_to_show = [int(i) for i in multi_choice_bins_param]
                if remove:
                    colors_bins.insert(index, 'black')
                    # reinstall resolved
                    multi_choice_bins_param.insert(index, 'RESOLVED')
                    bins_to_show.insert(index, 'RESOLVED')

                cache = cache_pipes[resolved_galaxy.galaxy_id].get(which_run_resolved_param)
                fig, cache = resolved_galaxy.plot_bagpipes_sfh(run_name=which_run_resolved_param, bins_to_show = bins_to_show, save=False, facecolor=facecolor, marker_colors=colors_bins, time_unit=x_unit, run_dir=run_dir, cache=cache) 
                cache_pipes[resolved_galaxy.galaxy_id][which_run_resolved_param] = cache
                if len(fig.get_axes()) != 0:
                    axes = fig.get_axes()[0]
                    #axes.set_xlim(0, 1)
                else:
                    axes = None
            else:
                fig = None
                axes = None

            if which_run_aperture_param is not None:
                bins_to_show = total_fit_options_param
                cache = cache_pipes[resolved_galaxy.galaxy_id].get(which_run_aperture_param)
                fig, cache = resolved_galaxy.plot_bagpipes_sfh(run_name=which_run_aperture_param, bins_to_show = bins_to_show, save=False, facecolor=facecolor, marker_colors=colors_total, time_unit=x_unit, run_dir=run_dir, cache=cache, fig = fig, axes = axes)
                cache_pipes[resolved_galaxy.galaxy_id][which_run_aperture_param] = cache

            if len(fig.get_axes()) != 0:
                ax = fig.get_axes()[0]
                #ax.set_xlim(0, 1)
            
            # Maybe fix

            '''
            try:
                import plotly as py
                import plotly.tools as tls
                fig = tls.mpl_to_plotly(fig)
                # set xlabel
                
                fig.update_layout(xaxis_title = 'Lookback Time (Gyr)', yaxis_title = 'SFR (M<sub>☉</sub>/yr)')
                # Set background color
                fig['layout']['plot_bgcolor'] = 'white'
                # set margin color
                fig['layout']['paper_bgcolor'] = facecolor
                # Rescale figure - it's too big
                fig['layout']['width'] = 500
                fig['layout']['height'] = 350
                # Fix other scaling - plot is tiny within window
                fig['layout']['autosize'] = False
                fig['layout']['margin'] = {'l': 20, 'r': 20, 't': 50, 'b': 20}
                # move legend
                # enable black axis borders
                fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
                fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True)

                fig.update_layout(legend=dict(
                    yanchor="top",
                    y=0.98,
                    xanchor="left",
                    x=0.01
                ))

                fig.update_layout(title_text="",
                          updatemenus=[
                dict(
                    buttons=list([
                        dict(label="Linear SFR",  
                            method="relayout", 
                            args=[{"yaxis.type": "linear"}]),
                        dict(label="Log SFR", 
                            method="relayout", 
                            args=[{"yaxis.type": "log"}]),
                                    ]),
                    x=0,
                    xanchor="left",
                    y=1.15,
                    yanchor="top"
                    ),

                dict(
                    buttons=list([
                        dict(label="Linear Time",  
                            method="relayout", 
                            args=[{"xaxis.type": "linear"}]),
                        dict(label="Log Time", 
                            method="relayout", 
                            args=[{"xaxis.type": "log"}]),
                                    ]),
                    x=0.2,
                    xanchor="left",
                    y=1.15,
                    yanchor="top"
                    ),
                ])
                
                # label buttons
                #fig.update_layout(
                #    annotations=[
                #        dict(text="Y-scale", x=0, xref="paper", y=1.22, yref="paper", align="left", bgcolor="rgba(0, 0, 0, 0)",
                #             showarrow=False),
                #        dict(text="X-scale", x=0.2, xref="paper", y=1.22, yref="paper", align="left", bgcolor="rgba(0, 0, 0, 0)",
                #             showarrow=False)
                #    ]
                #)
                
                return pn.pane.Plotly(fig, sizing_mode="scale_both", config={"scrollZoom": True}, max_width=400)
            except ImportError:
                pass
            '''

            plt.close(fig)

            return pn.pane.Matplotlib(fig, dpi=144, tight=True, format="svg", sizing_mode="scale_both", interactive = False)
    else:
        return pn.pane.Markdown('No Bagpipes results found.')

@cached_function
def plot_corner(resolved_galaxy, map, cmap, which_map_param, which_sed_fitter_param, multi_choice_bins_param, which_run_aperture_param, which_run_resolved_param, total_fit_options_param, facecolor='#f7f7f7'):
    # Can't always directly compare as models may have different parameters
    cmap = cm.get_cmap(cmap)
    # Ignore resolved
    if 'RESOLVED' in multi_choice_bins_param:
        multi_choice_bins_param.remove('RESOLVED')

    colors_bins = [cmap(Normalize(vmin=np.nanmin(map), vmax=np.nanmax(map))(rbin)) for pos, rbin in enumerate(multi_choice_bins_param)]
    colors_total = [TOTAL_FIT_COLORS[rbin] for rbin in total_fit_options_param]

    # FIX HERE
    if hasattr(resolved_galaxy, 'sed_fitting_table') and 'bagpipes' in resolved_galaxy.sed_fitting_table.keys() and (which_run_resolved_param in resolved_galaxy.sed_fitting_table['bagpipes'].keys() or which_run_aperture_param in resolved_galaxy.sed_fitting_table['bagpipes'].keys()):

        if which_sed_fitter_param == 'bagpipes':
            if resolved_galaxy.galaxy_id not in cache_pipes.keys():
                cache_pipes[resolved_galaxy.galaxy_id] = {}

            if which_run_aperture_param not in cache_pipes[resolved_galaxy.galaxy_id].keys():
                cache_pipes[resolved_galaxy.galaxy_id][which_run_aperture_param] = {}
            if which_run_aperture_param is not None:
                bins_to_show = total_fit_options_param
                cache = cache_pipes[resolved_galaxy.galaxy_id].get(which_run_aperture_param)
                fig, cache = resolved_galaxy.plot_bagpipes_corner(run_name=which_run_aperture_param, bins_to_show = bins_to_show, save=False, facecolor=facecolor, colors=colors_total, run_dir=run_dir, cache=cache) 
                cache_pipes[resolved_galaxy.galaxy_id][which_run_aperture_param] = cache
                #ax = fig.get_axes()[0]
                #ax.set_xlim(0, 1)
            '''
            # TEMP - see issue above.
            if resolved_galaxy.galaxy_id not in cache_pipes.keys():
                cache_pipes[resolved_galaxy.galaxy_id] = {}

            if which_run_resolved_param not in cache_pipes[resolved_galaxy.galaxy_id].keys():
                cache_pipes[resolved_galaxy.galaxy_id][resolved_galaxy.galaxy_id[which_run_resolved_param] = {}
            if which_run_resolved_param is not None:
                bins_to_show = [int(i) for i in multi_choice_bins_param]
                cache = cache_pipes[resolved_galaxy.galaxy_id].get(which_run_resolved_param)
                fig, cache = resolved_galaxy.plot_bagpipes_corner(run_name=which_run_resolved_param, bins_to_show = bins_to_show, save=False, facecolor=facecolor, colors=colors_bins, run_dir=run_dir, cache=cache, fig = fig)
                cache_pipes[resolved_galaxy.galaxy_id][which_run_resolved_param] = cache
            '''

            if fig == None:
                fig = plt.figure()

            fig.set_facecolor(facecolor)
            plt.close(fig)

            return pn.pane.Matplotlib(fig, dpi=144, tight=True, format="svg", sizing_mode="stretch_both", max_width=800, min_width = 500)

        
    else:
        return pn.pane.Markdown('No Bagpipes results found.')

def do_other_plot(plot_option):
    try:
        if plot_option == 'Galaxy Region':
            fig = resolved_galaxy.plot_gal_region(facecolor=facecolor)
        elif plot_option == 'Fluxes':
            fig = resolved_galaxy.pixedfit_plot_map_fluxes()
        elif plot_option == 'Segmentation Map':
            fig = resolved_galaxy.plot_seg_stamps()
        elif plot_option == 'Radial SNR':
            fig = resolved_galaxy.pixedfit_plot_radial_SNR()
    except Exception as e:
        print(e)
        fig = plt.figure()

    plt.close(fig)
    return pn.pane.Matplotlib(fig, dpi=144, tight=True, format="svg", max_width = 1000, sizing_mode = 'stretch_both')

def do_snr_plot(band):
    return pn.pane.Matplotlib(resolved_galaxy.plot_snr_map(band=band, facecolor = facecolor), dpi=144, tight=True, format="svg", width=300, height=300)

options_direct = {'Beta': f'beta_phot',
                'mUV': f'mUV_phot',
                'MUV': f'MUV_phot',
                'Lobs_UV': f'LUV_phot',
                'AUV': f'AUV_from_beta_phot',
                'SFR_UV': f'SFR_UV_phot',
                'fesc':'fesc_from_beta_phot',
                'EW_rest_optical':'EW_rest_optical',
                'line_flux_rest_optical':'line_flux_rest_optical',
                'xi_ion':'xi_ion'}

@cached_function
def plot_phot_property(property, cmap,  strong_line_names = ["Hbeta", "[OIII]-4959", "[OIII]-5007", "Halpha"],
                    rest_UV_wav_lims = [1250., 3000.] * u.Angstrom, ref_wav =  1_500. * u.AA, 
                    dust_author_year = 'M99', kappa_UV_conv_author_year = 'MD14',  density = False, logmap = False, 
                    frame = 'obs', conv_author_year = 'M99',):

    rest_UV_wav_text = f'{int(rest_UV_wav_lims[0].value)}_{int(rest_UV_wav_lims[1].value)}AA'

    author_func = {'fesc_from_beta_phot':{'conv_author_year':'Chisholm22'}}
    #with button_obj.param.update(loading = True):
    #yield pn.indicators.LoadingSpinner(size = 50, name = 'Loading...', value = True)

    default_options = {'SFR_UV':{'density':True, 'logmap':True}}
    if property in default_options:
        density = default_options[property]['density']
        logmap = default_options[property]['logmap']

    if options_direct[property] in author_func:
        print('Setting author year')
        conv_author_year = options_direct[property]['conv_author_year']

    fig = resolved_galaxy.galfind_phot_property_map(options_direct[property], rest_UV_wav_lims = rest_UV_wav_lims, 
                                                ref_wav = ref_wav, dust_author_year = dust_author_year,
                                                kappa_UV_conv_author_year = kappa_UV_conv_author_year, cmap = cmap,
                                                facecolor = facecolor, logmap = logmap, density = density, frame = frame,
                                                conv_author_year = conv_author_year, strong_line_names = strong_line_names)
    plt.close(fig)
    empty_phot_page.loading = False

    return pn.pane.Matplotlib(fig, dpi=144, tight=True, format="svg", sizing_mode='scale_width', min_width=200)

def synthesizer_page():

    empty_page = pn.Column()

    empty_page.append('### Synthesizer Properties')

    top_row = pn.Row()

    middle_row = pn.Row()

    # Plot photometry maps

    fig = pn.pane.Matplotlib(resolved_galaxy.plot_property_maps(facecolor = facecolor), dpi=144, tight=True, format="svg", sizing_mode='stretch_both', max_height = 500)

    top_row.append(fig)

    empty_page.append(top_row)

    empty_page.append(pn.layout.Divider())

    # plot spectra

    empty_page.append('### Spectra')

    
    fig = pn.pane.Matplotlib(resolved_galaxy.plot_mock_spectra(components = ['det_segmap_fnu'], facecolor = facecolor), dpi=144, tight=True, format="svg", sizing_mode='stretch_both', max_height = 500)

    middle_row.append(fig)

    empty_page.append(middle_row)

    return empty_page
    #return pn.pane.Markdown('Not implemented yet.')

    

    return empty_page

def fitsmap(fitsmap_dir = '/nvme/scratch/work/tharvey/fitsmap', port = 8000, band = 'F444W'):

    import socket

    a_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    location = ("127.0.0.1", port)
    check = a_socket.connect_ex(location)

    if check != 0:
        yield pn.indicators.LoadingSpinner(size = 50, name = 'Starting fitsmap server...', value = True)
        print('Starting fitsmap server...')
        command = f'cd {fitsmap_dir}/{resolved_galaxy.field}_{resolved_galaxy.survey}; fitsmap serve'
        subprocess.Popen(command, shell=True)
    
    ra = resolved_galaxy.sky_coord.ra.deg
    dec = resolved_galaxy.sky_coord.dec.deg
    image_path = resolved_galaxy.im_paths[band]
    path = image_path.replace('/mosaic_1084_wispnathan/', '/NIRCam/mosaic_1084_wispnathan/') if 'NIRCam' not in image_path else image_path
    header = fits.getheader(path, ext = 1)
    wcs = WCS(header)
    x_cent, y_cent = wcs.all_world2pix(ra, dec, 0)

    yield pn.pane.HTML(f'<iframe src="http://localhost:{port}/?ra={x_cent}&dec={y_cent}&zoom=7" width="100%" height="100%"></iframe>', sizing_mode='stretch_both', max_height = 520)



def phot_prop_func():
    empty_page = empty_phot_page
    
    # Delete all content
    empty_page.clear()

    empty_page.append('### Photometry Properties')
    top_row = pn.Row()
    
    middle_row = pn.Row()

    options_phot = list([key for key in options_direct.keys() if 'rest_optical' not in key])
    drop_down1 = pn.widgets.Select(options = options_phot, value = 'Beta', width = 100)
    cmap_1 = pn.widgets.Select( options = plt.colormaps(), value = 'viridis', width = 100)
    drop_down2 = pn.widgets.Select(options = options_phot, value = 'MUV', width = 100)
    cmap_2 = pn.widgets.Select( options = plt.colormaps(), value = 'viridis', width = 100)
    drop_down3 = pn.widgets.Select(options = options_phot, value = 'SFR_UV', width = 100)
    cmap_3 = pn.widgets.Select( options = plt.colormaps(), value = 'viridis', width = 100)

    plot1 =  pn.param.ParamFunction(pn.bind(plot_phot_property, drop_down1.param.value, cmap_1.param.value, watch=False), loading_indicator=True)
    top_row.append(pn.Column(pn.Row(drop_down1, cmap_1), plot1, sizing_mode = 'stretch_width'))
    plot2 =  pn.param.ParamFunction(pn.bind(plot_phot_property, drop_down2.param.value, cmap_2.param.value, watch=False), loading_indicator=True)
    top_row.append(pn.Column(pn.Row(drop_down2, cmap_2), plot2, sizing_mode = 'stretch_width'))
    plot3 =  pn.param.ParamFunction(pn.bind(plot_phot_property, drop_down3.param.value, cmap_3.param.value, watch=False), loading_indicator=True)
    top_row.append(pn.Column(pn.Row(drop_down3, cmap_3), plot3, sizing_mode = 'stretch_width'))

    empty_page.append(top_row)

    empty_page.append(pn.layout.Divider())
    empty_page.append('### Inferred Line Properties')

    #strong_line_options = 
    drop_down_4 = pn.widgets.Select(name = 'Line', options = ["Hbeta", "[OIII]-4959", "[OIII]-5007", "Halpha"], width = 100, value = "Hbeta")
    line_plot1 =  pn.param.ParamFunction(pn.bind(plot_phot_property, 'line_flux_rest_optical', 'viridis', drop_down_4.param.value, watch=False), loading_indicator=True)
    middle_row.append(pn.Column(line_plot1, sizing_mode = 'stretch_width'))
    line_plot2 =  pn.param.ParamFunction(pn.bind(plot_phot_property, 'EW_rest_optical', 'viridis', drop_down_4.param.value, watch=False), loading_indicator=True)
    middle_row.append(pn.Column(line_plot2, sizing_mode = 'stretch_width'))
    line_plot2 =  pn.param.ParamFunction(pn.bind(plot_phot_property, 'line_lum_rest_optical', 'viridis', drop_down_4.param.value, watch=False), loading_indicator=True)
    middle_row.append(pn.Column(line_plot2, sizing_mode = 'stretch_width'))

    empty_page.append(drop_down_4)

    empty_page.append(middle_row)
    
    return empty_page
    #page = pn.Column(top_row, middle_row, sizing_mode='stretch_both', loading = True)


    #return pn.pane.Markdown('Not implemented yet.')

@cached_function
def other_bagpipes_results_func(param_property, p_run_name, norm_param, total_params, weight_mass_sfr, outtype = 'median'):
    #param_property = str(param_property)
    #p_run_name = str(p_run_name)

    #yield pn.Row(pn.indicators.LoadingSpinner(size = 50, name = 'Loading...', value = True), height=400, width=400)

    if param_property is None or p_run_name is None:
        print('no property')
        return pn.pane.Markdown('No property or run selected.')
    
    if 'bagpipes' not in resolved_galaxy.sed_fitting_table.keys():
        return pn.pane.Markdown('No Bagpipes results found.')

    print(p_run_name)
    options = list(resolved_galaxy.sed_fitting_table['bagpipes'][p_run_name].keys())
    options = [i for i in options if not (i.startswith('#') or i.endswith('16') or i.endswith('84'))]
    actual_options = [i.replace('_50', '') for i in options]
    dist_options = [i for i in options if i.endswith('50')]
    pdf_param_property.options = [i.replace('_50', '') for i in dist_options]
    other_bagpipes_properties_dropdown.options = actual_options
    other_bagpipes_properties_dropdown.value = param_property

    no_log = ['ssfr']
    if param_property in no_log:
        norm.value = 'linear'
        norm.disabled = True
    else:
        norm.disabled = False


    if f'{param_property}_50' not in dist_options:
        if not param_property.endswith('-'):
            param_property = f'{param_property}-'

    if outtype == 'gif':
        logmap = False
        scale = 'linear'
        if param_property in ['stellar_mass', 'sfr']:
            
            if norm_param == 'log':
                logmap = True
        else:
            scale = norm_param
            weight_mass_sfr = False
                
        
        
        fig = resolved_galaxy.plot_bagpipes_map_gif(parameter = param_property,  weight_mass_sfr=weight_mass_sfr, logmap=logmap, path = 'temp', facecolor=facecolor, cmap = 'magma')
    elif outtype == 'median':
        fig = resolved_galaxy.plot_bagpipes_results(facecolor=facecolor, parameters=[param_property], max_on_row=3, run_name=p_run_name, norm = norm_param, total_params = total_params)

    if fig is not None:
        if type(fig) == plt.Figure:
            plt.close(fig)
            #print('returning fig')
            return pn.pane.Matplotlib(fig, dpi=144, tight=True, format="svg", width = 400, height = 400)
        else:
            #import html
            #iframe = f'<iframe src="{html.escape(fig)}" width="100%" height="100%"></iframe>'
            #from IPython.display import HTML
            #return pn.pane.HTML(HTML(fig), width = 400, height = 400)
            return pn.pane.GIF(fig, width = 400, height = 400)
    else:
        print('Returning none')
        return pn.pane.Markdown('No Bagpipes results found.')


def sed_results_plot_func(which_sed_fitter_param, which_run_resolved_param, upscale_select_param):
    if which_sed_fitter_param == 'bagpipes':
        sed_results_plot = resolved_galaxy.plot_bagpipes_results(facecolor=facecolor, parameters=['stellar_mass', 'sfr'], max_on_row=3, run_name = which_run_resolved_param, norm = 'linear', weight_mass_sfr = upscale_select_param)
        if sed_results_plot is not None:
            plt.close(sed_results_plot)
            sed_results = pn.Row(pn.pane.Matplotlib(sed_results_plot, dpi=144, tight=True, format="svg", sizing_mode="scale_both"))
        else:
            sed_results = pn.pane.Markdown('No Bagpipes results found.')
    else:
        sed_results = pn.pane.Markdown('Not implemented yet.')
    
    return sed_results

def plot_cutouts(psf_matched_param, psf_type_param):
    if psf_matched_param == 'Original':
        psf_matched_param = False
    elif psf_matched_param == 'PSF Matched':
        psf_matched_param = True
    fig = resolved_galaxy.plot_cutouts(facecolor=facecolor, psf_matched = psf_matched_param, psf_type = psf_type_param)
    plt.close(fig)
    return pn.pane.Matplotlib(fig, dpi=144, max_width=950, tight=True, format="svg", sizing_mode='scale_width')



def handle_file_upload(value, components):
    global bin_map
    global resolved_galaxy
    global stream 
    global sed_results_grid
    global cmap
    global which_map
    global which_sed_fitter
    global which_flux_unit
    global multi_choice_bins
    global total_fit_options
    global which_run_resolved
    global which_run_aperture
    global red_select
    global green_select
    global blue_select
    global stretch_slider
    global q_slider
    global psf_mode_select
    global other_bagpipes_properties_dropdown
    global scale_alpha
    global show_galaxy
    global show_kron
    global choose_show_band
    global empty_phot_page
    global pdf_param_property
    global norm
    global upscale_select
    global show_sed_photometry

    which_map = pn.widgets.RadioButtonGroup(options=['pixedfit', 'voronoi'], value='pixedfit', name='Pixel Binning')

    which_sed_fitter = pn.widgets.Select(name='SED Fitter', value='bagpipes', options=['bagpipes'])

    which_flux_unit = pn.widgets.Select(name='Flux Unit', value='uJy', options=['uJy', 'ABmag', 'ergscma'])

    multi_choice_bins = pn.widgets.MultiChoice(name='Bins', options=['RESOLVED'], delete_button=True, placeholder='Click on bin map to add bins')

    total_fit_options = pn.widgets.MultiChoice(name='Aperture Fits', options=['TOTAL_BIN', 'MAG_AUTO', 'MAG_BEST', 'MAG_ISO', 'MAG_APER_TOTAL'], value=['TOTAL_BIN'], delete_button=True, placeholder='Select combined fits to show')

    #which_run = pn.widgets.Select(name='Run', value=None, options=[])
    which_run_resolved = pn.widgets.Select(name='Resolved SED Fitting Run', value=None, options=[])
    which_run_aperture = pn.widgets.Select(name='Integrated SED Fitting Run', value=None, options=[])

    scale_alpha = pn.widgets.FloatSlider(name='Scale Alpha', start=0, end=1, value=1, step=0.01)
    show_galaxy = pn.widgets.Checkbox(name='Show Galaxy', value=False)
    show_kron = pn.widgets.Checkbox(name='Show Kron', value=False)


    file = BytesIO(value)

    hfile = h5.File(file, 'r')
    # what is the filename
    
    mtype = 'mock' if 'mock_galaxy' in hfile.keys() else 'resolved'

    if mtype == 'mock':
        resolved_galaxy = MockResolvedGalaxy.init_from_h5(file)
    else:
        resolved_galaxy = ResolvedGalaxy.init_from_h5(file)

    sidebar, tabs = components

    id = resolved_galaxy.galaxy_id
    survey = resolved_galaxy.survey

    choose_show_band = pn.widgets.Select(name='Show Band', options = resolved_galaxy.bands, value = 'F444W')


    #cutout_grid = GridStack(sizing_mode='stretch_both', allow_resize=True, allow_drag=True, max_height=500, nrows=4)

    cutout_grid = pn.Column(scroll = False)

    psf_mode_select = pn.widgets.Select(name='PSF Mode', options=['PSF Matched', 'Original'], value='PSF Matched')
    

    sed_results_grid = GridStack(sizing_mode='stretch_both', allow_resize=True, allow_drag=True, min_height=800, mode='override')

    #cutout_grid[0, :6] = 
    
    row = pn.Row(pn.bind(plot_cutouts, psf_mode_select.param.value, 'star_stack'), sizing_mode='stretch_both', scroll=False)
    cutout_grid.append(row)

    cmap = 'nipy_spectral_r'
 
    #hvplot_bins = plot_bins('pixedfit', cmap)
    bin_map = pn.bind(plot_bins, 'pixedfit', cmap, scale_alpha.param.value, show_galaxy.param.value, show_kron.param.value, psf_mode_select.param.value, choose_show_band.param.value)
    # set stream off holoviews object instead
    
    pdf_param_property = pn.widgets.Select(name='PDF Property', options = ['stellar_mass', 'sfr'], value = 'stellar_mass')
    
    # Get SED options
    possible_runs_select('bagpipes')

    show_sed_photometry = pn.widgets.Select(name='Fit Photometry', options = ['Show', 'Don\'t show'], value='Don\'t show', width = 150)
    
    sed_obj =  pn.param.ParamFunction(pn.bind(handle_map_click, stream.param.x, stream.param.y, resolved_galaxy, 
                            cmap, which_map.param.value, which_sed_fitter.param.value, 
                            which_flux_unit.param.value, multi_choice_bins.param.value,
                            which_run_aperture.param.value, which_run_resolved.param.value, total_fit_options.param.value, None, show_sed_photometry.param.value,
                            'sed', watch=False), loading_indicator = True)
    
    sfh_obj =  pn.param.ParamFunction(pn.bind(handle_map_click, stream.param.x, stream.param.y, resolved_galaxy, 
                            cmap, which_map.param.value, which_sed_fitter.param.value, 
                            which_flux_unit.param.value, multi_choice_bins.param.value,
                            which_run_aperture.param.value, which_run_resolved.param.value, total_fit_options.param.value, None, None,
                             'sfh', watch=False), loading_indicator = True)
    
    corner_obj =  pn.param.ParamFunction(pn.bind(handle_map_click, stream.param.x, stream.param.y, resolved_galaxy, cmap,
                            which_map.param.value, which_sed_fitter.param.value, 
                            which_flux_unit.param.value, multi_choice_bins.param.value,
                            which_run_aperture.param.value, which_run_resolved.param.value, total_fit_options.param.value, None, None,
                            'corner', watch=False), loading_indicator = True)

    pdf_obj =  pn.param.ParamFunction(pn.bind(handle_map_click, stream.param.x, stream.param.y, resolved_galaxy, cmap,
                            which_map.param.value, which_sed_fitter.param.value, 
                            which_flux_unit.param.value, multi_choice_bins.param.value,
                            which_run_aperture.param.value, which_run_resolved.param.value, total_fit_options.param.value, pdf_param_property.param.value, None,
                            'pdf', watch=False), loading_indicator = True)
    
    red_select = pn.widgets.MultiChoice(name='Red Channel', options = resolved_galaxy.bands, value = ['F444W'])
    green_select = pn.widgets.MultiChoice(name='Green Channel', options = resolved_galaxy.bands, value = ['F277W'])
    blue_select = pn.widgets.MultiChoice(name='Blue Channel', options = resolved_galaxy.bands, value = ['F150W'])

    stretch_slider = pn.widgets.EditableFloatSlider(name='Stretch', start=0.001, end=10, value=6, step=0.001)
    q_slider = pn.widgets.EditableFloatSlider(name='Q', start=0.000001, end=0.01, value=0.001, step=0.000001)
   
    band_select = pn.widgets.Select(name='Band', options=resolved_galaxy.bands, value='F444W')
    snr_plot = pn.bind(do_snr_plot, band_select.param.value)

    other_plot_select = pn.widgets.Select(options=['Galaxy Region', 'Fluxes', 'Segmentation Map', 'Radial SNR'], value='Galaxy Region')
    other_plot = pn.bind(do_other_plot, other_plot_select.param.value)
    # Show RGB cutout
    cutout_grid.append(pn.Row(pn.Column('### RGB Image', pn.bind(plot_rgb, resolved_galaxy, red_select.param.value, green_select.param.value, blue_select.param.value, stretch_slider.param.value, q_slider.param.value, psf_mode_select.param.value), scroll=False),
    pn.Column(band_select, snr_plot, scroll=False), sizing_mode = 'scale_both', max_width=1000, scroll=False, min_height=500))
    cutout_grid.append(pn.Row(pn.Column('### Other Plots', pn.Row(other_plot_select, scroll=False), pn.Row(other_plot, min_height = 500, scroll=False)), sizing_mode = 'scale_both', max_width=1000, scroll=False, min_height=500))

    #obj = pn.bind(plot_sed, bin_map.param.tap.x, bin_map.param.tap.y, resolved_galaxy, cmap, shown_bins, watch=True)
    '''
    sed_results_grid[0, :2] = bin_map
    sed_results_grid[0, 2:] = sed_obj
    sed_results_grid[2, :3] = sfh_obj
    sed_results_grid[3:5, :3] = corner_obj
    '''
    upscale_select = pn.widgets.Select(name='Upscale Mass/SFR', options = [True, False], value = True, width=100)

    norm = pn.widgets.RadioButtonGroup(name='Scaling', options=['linear', 'log'], value='linear', button_type='primary')
    #sed_results_plot = sed_results_plot_func(resolved_galaxy, which_sed_fitter.param.value, which_run.param.value)
    sed_results = pn.bind(sed_results_plot_func, which_sed_fitter.param.value, which_run_resolved.param.value, upscale_select.param.value, watch=True)
    
    other_bagpipes_properties_dropdown = pn.widgets.Select(name='SED Properties Map', options = ['constant:age_max'], value = 'constant:age_max')
    type_select = pn.widgets.Select(name='Type', options = ['median', 'gif'], value = 'median', width=90)
   
    #other_bagpipes_results = other_bagpipes_results_func(resolved_galaxy, other_bagpipes_properties_dropdown.param.value, which_run.param.value)
    other_bagpipes_results = pn.param.ParamFunction(pn.bind(other_bagpipes_results_func, other_bagpipes_properties_dropdown.param.value, which_run_resolved.param.value, norm.param.value, total_fit_options.param.value, upscale_select.param.value, type_select.param.value, watch=False), 
                                                    loading_indicator = True)
    '''
    sed_results_grid[2, 3:5] = sed_results
    '''

    # Alternative to SED results grid using Columns and Rows
    top_row = pn.Row(bin_map, sed_obj, height=350)
    mid_row = pn.Row(sfh_obj, sed_results, height=300)
    bot_row = pn.Row(corner_obj, pn.Column(pn.Row(other_bagpipes_properties_dropdown), pn.Row(norm, type_select), pn.Row(other_bagpipes_results)), height=750)
    even_lower = pn.Row(pn.Column(pdf_param_property, pdf_obj, max_height=400, max_width=400))

    combined = pn.Column(top_row, mid_row, bot_row, even_lower, scroll=False)

    empty_phot_page = pn.Column(loading = True, sizing_mode='stretch_both', min_height=400, min_width = 400)#, max_width=1000)
    #phot_prop = pn.param.ParamFunction(phot_prop_func,  lazy=True)

    galaxy_tabs = pn.Tabs(('Cutouts', cutout_grid), ('SED Results', combined), ('Photometric Properties', empty_phot_page), dynamic=True, scroll=False)
    
    if subprocess.getstatusoutput('which fitsmap')[0] == 0:
        fitsmap_page = pn.param.ParamFunction(fitsmap, watch=False, lazy = True)
        #fitsmap_page = pn.bind(fits, watch=False)
        galaxy_tabs.append(('Fitsmap', fitsmap_page))

    if type(resolved_galaxy) == MockResolvedGalaxy:
        print('Adding synthesizer')
        mock_page = pn.param.ParamFunction(synthesizer_page, watch=False, lazy = True)
        galaxy_tabs.append(('Synthesizer', mock_page))
   
    tabs.append((f"{id} ({survey})", galaxy_tabs))

    # Only show sidebar if tab is SED Results
    #@pn.depends(which_map.param.value)
    update_sidebar(0, sidebar)

    pn.param.ParamFunction(pn.bind(update_sidebar, galaxy_tabs.param.active, sidebar, watch=True), loading_indicator = True)


def choose_file(value, components):

    path = f'{galaxies_dir}/{value}'

    # check if path exists
    if os.path.exists(path):
        # make a bytesstream
        with open(path, 'rb') as f:
            value = f.read()
        
    handle_file_upload(value, components)


def resolved_sed_interface():
    global file_input

    tabs = pn.Tabs(closable=True, dynamic=True, scroll=False, min_height=2000)
    
    file_input = pn.widgets.FileInput(accept='.h5')

    choose_file_input = pn.widgets.Select(name='Select Remote File', options = [None], value = None, width = 200)

    choose_file_input.options = [None] + sorted([f for f in os.listdir(galaxies_dir) if f.endswith('.h5')])

    sidebar = pn.Column("### Upload .h5", file_input, '### or', choose_file_input)
    
    components = [sidebar, tabs]

    pn.bind(handle_file_upload, file_input, components, watch=True) # watch = True is required for these!
    pn.bind(choose_file, choose_file_input, components, watch=True)

    return pn.template.FastListTemplate(
        title="Resolved SED Viewer", sidebar=[sidebar], main=[tabs], accent=ACCENT
    )

#resolved_sed_interface().servable()

pn.serve(resolved_sed_interface, websocket_max_message_size=MAX_SIZE_MB*1024*1024,
    http_server_kwargs={'max_buffer_size': MAX_SIZE_MB*1024*1024})







