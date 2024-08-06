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
from ResolvedGalaxy import ResolvedGalaxy
import h5py
from io import BytesIO
import sys
import inspect
import matplotlib as mpl
from astropy import units as u
import functools
import os
import subprocess
from astropy.wcs import WCS
from astropy.io import fits

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

plotpipes_dir = '/Users/user/Documents/PhD/bagpipes_dir/'
run_dir = 'pipes/'
cache_pipes = {}

pn.extension(sizing_mode="stretch_width", design='material')
pn.extension('gridstack')
pn.extension(notifications=True)
facecolor = '#f7f7f7'

MAX_SIZE_MB = 150

TOTAL_FIT_COLORS = {'TOTAL_BIN': 'red', 'MAG_AUTO': 'blue', 'MAG_APER': 'green', 'MAG_ISO': 'purple', 'MAG_APER_TOTAL': 'orange', 'MAG_BEST': 'cyan'}

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
    if options is None:
        options = ['No runs found']
        
    else:
        options = list(options.keys())
    
    which_run.options = options
    which_run.value = options[0]

    return which_run

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
        settings_sidebar.append(show_galaxy)
        settings_sidebar.append(scale_alpha)
        settings_sidebar.append(choose_show_band)
        settings_sidebar.append('#### SED Fitting Tool')
        settings_sidebar.append(which_sed_fitter)
        settings_sidebar.append(pn.bind(possible_runs_select, which_sed_fitter.param.value, watch=True))

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


def handle_map_click(x, y, resolved_galaxy, cmap, which_map_param, which_sed_fitter_param, which_flux_unit_param, multi_choice_bins_param, which_run_param, total_fit_options_param, mode='sed'):
    print('handel map click')
    use = True
    if x == None or y == None:
        use = False
    if use:
        if not (0 < x < resolved_galaxy.cutout_size and 0 < y < resolved_galaxy.cutout_size):
            use = False
    
    map = resolved_galaxy.pixedfit_map
    if use:
        # Need logic here to know which map to use
        bin = map[int(np.ceil(y)), int(np.ceil(x))]
        if bin not in multi_choice_bins_param:
            multi_choice_bins.value = multi_choice_bins_param + [bin]
            print(f'Added bin. {bin}')
    
    multi_choice_bins_param_safe = []
    for i in multi_choice_bins_param:
        if i != np.nan:
            multi_choice_bins_param_safe.append(int(i))
        else:
            pass
    
    yield pn.indicators.LoadingSpinner(size = 50, name = 'Loading...', value = True)
        
    if mode == 'sed':
        obj = plot_sed(resolved_galaxy, map, cmap, which_map_param, which_sed_fitter_param, which_flux_unit_param, multi_choice_bins_param_safe, which_run_param, total_fit_options_param)
    if mode == 'sfh':
        obj = plot_sfh(resolved_galaxy, map, cmap, which_map_param, which_sed_fitter_param, multi_choice_bins_param_safe, which_run_param, total_fit_options_param)
    if mode == 'corner':
        obj = plot_corner(resolved_galaxy, map, cmap, which_map_param, which_sed_fitter_param, multi_choice_bins_param_safe, which_run_param, total_fit_options_param)

    yield obj 
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

    return pn.pane.HoloViews(rgb_img, height=400, width=430)

@cached_function
def plot_sed(resolved_galaxy, map, cmap, which_map_param, which_sed_fitter_param, which_flux_unit_param, multi_choice_bins_param, which_run_param, total_fit_options_param, x_unit = u.micron):

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
    colors = []
    cmap = cm.get_cmap(cmap)

    if table is not None:
        for bin_pos, bin in enumerate(multi_choice_bins_param + total_fit_options_param):
            if type(bin) == str:
                color = TOTAL_FIT_COLORS[bin]
            else:
                color = cmap(bin/np.nanmax(map))

            colors.append(color)
            table_row = table[table['ID'] == bin]
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
                    # Assymmetric error bars
                    yerr = [[2.5*abs(np.log10(flux.value/(flux.value - flux_err.value)))], [2.5*np.log10(1 + flux_err.value/flux.value)]]

                else:
                    yerr = flux_err.to(y_unit, equivalencies = u.spectral_density(wav)).value
                lab = int(bin) if type(bin) == float else bin
                lab = lab if pos == 0 else ''
                ax.errorbar(wav.to(x_unit).value, flux.to(y_unit, equivalencies = u.spectral_density(wav)).value, yerr=yerr, fmt=marker, linestyle='none', color=color, label = lab)
    
    ax.legend(loc='upper left', frameon=False)    
    
    if y_unit == u.ABmag:
        ax.invert_yaxis()
    ax.set_xlabel(rf"$\rm{{Wavelength}}$ ({x_unit:latex})", fontsize='large')
    ax.set_ylabel(rf"$\rm{{Flux \ Density}}$ ({y_unit:latex})", fontsize='large')
    ax.set_xlim(ax.get_xlim())
    ax.set_ylim(ax.get_ylim())

    
    if which_sed_fitter_param == 'bagpipes' and which_run_param != None:
        # Check if bagpipes run exists
        if hasattr(resolved_galaxy, 'sed_fitting_table') and 'bagpipes' in resolved_galaxy.sed_fitting_table.keys() and which_run_param in resolved_galaxy.sed_fitting_table['bagpipes'].keys():
            if which_run_param not in cache_pipes:
                cache_pipes[which_run_param] = {}
            cache = cache_pipes.get(which_run_param)
            fig, cache == resolved_galaxy.plot_bagpipes_fit(which_run_param, ax, fig, bins_to_show=multi_choice_bins_param + total_fit_options_param, marker_colors=colors, wav_units=x_unit, flux_units=y_unit, cache=cache)
            cache_pipes[which_run_param] = cache

 
    return pn.pane.Matplotlib(fig, dpi=144, tight=True, format="svg", sizing_mode="scale_both")
    #return False

def plot_bins(bin_type, cmap, scale_alpha, show_galaxy, psf_matched, band = 'F444W'):
    array = getattr(resolved_galaxy, f"{bin_type}_map")
    if array is None:
        # Empty plot
        return None
    array[array == 0] = np.nan
    dimensions = np.linspace(0, resolved_galaxy.cutout_size, resolved_galaxy.cutout_size)
    im_array = xr.DataArray(array, dims=['y', 'x'], name=f'{bin_type} bins', coords={'x': dimensions, 'y': dimensions})
    hvplot = im_array.hvplot('x','y').opts(cmap=cmap, xaxis=None, yaxis=None, clabel=f'{bin_type} Bin', alpha=scale_alpha)
    multi_choice_bins.options = list(np.unique(array))
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
def plot_sfh(resolved_galaxy, map, cmap, which_map_param, which_sed_fitter_param, multi_choice_bins_param, which_run_param, total_fit_options_param, x_unit = 'Gyr', facecolor='#f7f7f7'):
    
    cmap = cm.get_cmap(cmap)
    colors = [cmap(bin/np.nanmax(map)) for bin in multi_choice_bins_param] + [TOTAL_FIT_COLORS[bin] for bin in total_fit_options_param]

    if hasattr(resolved_galaxy, 'sed_fitting_table') and 'bagpipes' in resolved_galaxy.sed_fitting_table.keys() and which_run_param in resolved_galaxy.sed_fitting_table['bagpipes'].keys():   
        if which_sed_fitter_param == 'bagpipes':
            if which_run_param not in cache_pipes:
                cache_pipes[which_run_param] = {}

            bins_to_show = [int(i) for i in multi_choice_bins_param] + total_fit_options_param
            cache = cache_pipes.get(which_run_param)
            fig, cache = resolved_galaxy.plot_bagpipes_sfh(run_name=which_run_param, bins_to_show = bins_to_show, save=False, facecolor=facecolor, marker_colors=colors, time_unit=x_unit, run_dir=run_dir, plotpipes_dir=plotpipes_dir, cache=cache) 
            cache_pipes[which_run_param] = cache

            if len(fig.get_axes()) != 0:
                ax = fig.get_axes()[0]
                ax.set_xlim(0, 1)
        

            return pn.pane.Matplotlib(fig, dpi=144, tight=True, format="svg", sizing_mode="scale_both")
    else:
        return pn.pane.Markdown('No Bagpipes results found.')

@cached_function
def plot_corner(resolved_galaxy, map, cmap, which_map_param, which_sed_fitter_param, multi_choice_bins_param, which_run_param, total_fit_options_param, facecolor='#f7f7f7'):
    
    cmap = cm.get_cmap(cmap)
    colors = [cmap(bin/np.nanmax(map)) for bin in multi_choice_bins_param] + [TOTAL_FIT_COLORS[bin] for bin in total_fit_options_param]


    if hasattr(resolved_galaxy, 'sed_fitting_table') and 'bagpipes' in resolved_galaxy.sed_fitting_table.keys() and which_run_param in resolved_galaxy.sed_fitting_table['bagpipes'].keys():   

        if which_sed_fitter_param == 'bagpipes':
            if which_run_param not in cache_pipes:
                cache_pipes[which_run_param] = {}

            bins_to_show = [int(i) for i in multi_choice_bins_param] + total_fit_options_param
            cache = cache_pipes.get(which_run_param)
            fig, cache = resolved_galaxy.plot_bagpipes_corner(run_name=which_run_param, bins_to_show = bins_to_show, save=False, facecolor=facecolor, colors=colors, run_dir=run_dir, plotpipes_dir=plotpipes_dir, cache=cache) 
            cache_pipes[which_run_param] = cache
            #ax = fig.get_axes()[0]
            #ax.set_xlim(0, 1)
            if fig == None:
                fig = plt.figure()

            fig.set_facecolor(facecolor)

            return pn.pane.Matplotlib(fig, dpi=144, tight=True, format="svg", sizing_mode="stretch_both", max_width=800, min_width = 500)

        
    else:
        return pn.pane.Markdown('No Bagpipes results found.')

def do_other_plot(plot_option):
    if plot_option == 'Galaxy Region':
        fig = resolved_galaxy.plot_gal_region(facecolor=facecolor)
    elif plot_option == 'Fluxes':
        fig = resolved_galaxy.pixedfit_plot_map_fluxes()
    elif plot_option == 'Segmentation Map':
        fig = resolved_galaxy.plot_seg_stamps()
    elif plot_option == 'Radial SNR':
        resolved_galaxy.pixedfit_plot_radial_SNR()
    
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
def plot_phot_property(property, button_obj, rest_UV_wav_lims = [1250., 3000.] * u.Angstrom, ref_wav =  1_500. * u.AA, 
                    dust_author_year = 'M99', kappa_UV_conv_author_year = 'MD14',  density = False, logmap = False, 
                    frame = 'obs', conv_author_year = 'M99', strong_line_names = ["Hbeta", "[OIII]-4959", "[OIII]-5007", "Halpha"]):

    rest_UV_wav_text = f'{int(rest_UV_wav_lims[0].value)}_{int(rest_UV_wav_lims[1].value)}AA'

    #with button_obj.param.update(loading = True):
    yield pn.indicators.LoadingSpinner(size = 50, name = 'Loading...', value = True)

    default_options = {'SFR_UV':{'density':True, 'logmap':True}}
    if property in default_options:
        density = default_options[property]['density']
        logmap = default_options[property]['logmap']

    fig = resolved_galaxy.galfind_phot_property_map(options_direct[property], rest_UV_wav_lims = rest_UV_wav_lims, 
                                                ref_wav = ref_wav, dust_author_year = dust_author_year,
                                                kappa_UV_conv_author_year = kappa_UV_conv_author_year,
                                                facecolor = facecolor, logmap = logmap, density = density, frame = frame,
                                                conv_author_year = conv_author_year, strong_line_names = strong_line_names)

    empty_phot_page.loading = False

    yield pn.pane.Matplotlib(fig, dpi=144, tight=True, format="svg", sizing_mode='scale_width', min_width=200)


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
    header = fits.getheader(image_path.replace('/mosaic_1084_wispnathan/', '/NIRCam/mosaic_1084_wispnathan/'), ext = 1)
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

    options_phot = list(options_direct.keys())
    drop_down1 = pn.widgets.Select(options = options_phot, value = 'Beta')
    drop_down2 = pn.widgets.Select(options = options_phot, value = 'MUV')
    drop_down3 = pn.widgets.Select(options = options_phot, value = 'SFR_UV')

    plot1 = pn.bind(plot_phot_property, drop_down1.param.value, drop_down1, watch=False)
    top_row.append(pn.Column(drop_down1, plot1, sizing_mode = 'stretch_width'))
    plot2 = pn.bind(plot_phot_property, drop_down2.param.value, drop_down2, watch=False)
    top_row.append(pn.Column(drop_down2, plot2, sizing_mode = 'stretch_width'))
    plot3 = pn.bind(plot_phot_property, drop_down3.param.value, drop_down3, watch=False)
    top_row.append(pn.Column(drop_down3, plot3, sizing_mode = 'stretch_width'))

    empty_page.append(top_row)

    empty_page.append(pn.layout.Divider())
    middle_row = empty_page.append('### Inferred Line Properties')

    #strong_line_options = 
    line_plot = pn.bind(plot_phot_property, 'line_flux_rest_optical', watch=False)
    
    
    empty_page.append(middle_row)
    
    return empty_page
    #page = pn.Column(top_row, middle_row, sizing_mode='stretch_both', loading = True)


    #return pn.pane.Markdown('Not implemented yet.')

@cached_function
def other_bagpipes_results_func(param_property, p_run_name, norm, total_params):
    #param_property = str(param_property)
    #p_run_name = str(p_run_name)

    if param_property is None or p_run_name is None:
        print('no property')
        return pn.pane.Markdown('No property or run selected.')

    else:
        options = list(resolved_galaxy.sed_fitting_table['bagpipes'][p_run_name].keys())
        options = [i for i in options if not (i.startswith('#') or i.endswith('16') or i.endswith('84'))]
        actual_options = [i.replace('_50', '') for i in options]
        dist_options = [i for i in options if i.endswith('50')]
        other_bagpipes_properties_dropdown.options = actual_options
        if f'{param_property}_50' not in dist_options:
            if not param_property.endswith('-'):
                param_property = f'{param_property}-'

        fig = resolved_galaxy.plot_bagpipes_results(facecolor=facecolor, parameters=[param_property], max_on_row=3, run_name=p_run_name, norm = norm, total_params = total_params)
        if fig is not None:
            print('returning fig')
            return pn.pane.Matplotlib(fig, dpi=144, tight=True, format="svg", sizing_mode='scale_both', min_width=400, width = 400, height = 400)
        else:
            print('Returning none')
            return pn.pane.Markdown('No Bagpipes results found.')


def sed_results_plot_func(which_sed_fitter_param, which_run_param):
    if which_sed_fitter_param == 'bagpipes':
        sed_results_plot = resolved_galaxy.plot_bagpipes_results(facecolor=facecolor, parameters=['stellar_mass', 'sfr'], max_on_row=3, run_name = which_run_param, norm = 'linear')
        if sed_results_plot is not None:
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
    return pn.pane.Matplotlib(resolved_galaxy.plot_cutouts(facecolor=facecolor, psf_matched = psf_matched_param, psf_type = psf_type_param), dpi=144, max_width=950, tight=True, format="svg", sizing_mode='scale_width')



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
    global which_run
    global red_select
    global green_select
    global blue_select
    global stretch_slider
    global q_slider
    global psf_mode_select
    global other_bagpipes_properties_dropdown
    global scale_alpha
    global show_galaxy
    global choose_show_band
    global empty_phot_page

    which_map = pn.widgets.RadioButtonGroup(options=['pixedfit', 'voronoi'], value='pixedfit', name='Pixel Binning')

    which_sed_fitter = pn.widgets.Select(name='SED Fitter', value='bagpipes', options=['bagpipes'])

    which_flux_unit = pn.widgets.Select(name='Flux Unit', value='uJy', options=['uJy', 'ABmag', 'ergscma'])

    multi_choice_bins = pn.widgets.MultiChoice(name='Bins', options=[], delete_button=True, placeholder='Click on bin map to add bins')

    total_fit_options = pn.widgets.MultiChoice(name='Aperture Fits', options=['TOTAL_BIN', 'MAG_AUTO', 'MAG_BEST', 'MAG_ISO', 'MAG_APER_TOTAL'], value=['TOTAL_BIN'], delete_button=True, placeholder='Select combined fits to show')

    which_run = pn.widgets.Select(name='Run', value=None, options=[])

    scale_alpha = pn.widgets.FloatSlider(name='Scale Alpha', start=0, end=1, value=1, step=0.01)
    show_galaxy = pn.widgets.Checkbox(name='Show Galaxy', value=False)

    file = BytesIO(value)
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
    bin_map = pn.bind(plot_bins, 'pixedfit', cmap, scale_alpha.param.value, show_galaxy.param.value, psf_mode_select.param.value, choose_show_band.param.value)
    # set stream off holoviews object instead
    

    sed_obj = pn.bind(handle_map_click, stream.param.x, stream.param.y, resolved_galaxy, 
                            cmap, which_map.param.value, which_sed_fitter.param.value, 
                            which_flux_unit.param.value, multi_choice_bins.param.value,
                            which_run.param.value, total_fit_options.param.value,
                            'sed', watch=False)
    
    sfh_obj = pn.bind(handle_map_click, stream.param.x, stream.param.y, resolved_galaxy, 
                            cmap, which_map.param.value, which_sed_fitter.param.value, 
                            which_flux_unit.param.value, multi_choice_bins.param.value,
                            which_run.param.value, total_fit_options.param.value,
                             'sfh', watch=False)
    
    corner_obj = pn.bind(handle_map_click, stream.param.x, stream.param.y, resolved_galaxy, cmap,
                            which_map.param.value, which_sed_fitter.param.value, 
                            which_flux_unit.param.value, multi_choice_bins.param.value,
                            which_run.param.value, total_fit_options.param.value,
                            'corner', watch=False)
    
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
    
    norm = pn.widgets.RadioButtonGroup(name='Scaling', options=['linear', 'log'], value='linear')
    #sed_results_plot = sed_results_plot_func(resolved_galaxy, which_sed_fitter.param.value, which_run.param.value)
    sed_results = pn.bind(sed_results_plot_func, which_sed_fitter.param.value, which_run.param.value, watch=True)
    
    other_bagpipes_properties_dropdown = pn.widgets.Select(name='SED Properties Map', options = ['chisq_phot-'], value = 'chisq_phot-')
    #other_bagpipes_results = other_bagpipes_results_func(resolved_galaxy, other_bagpipes_properties_dropdown.param.value, which_run.param.value)
    other_bagpipes_results = pn.bind(other_bagpipes_results_func, other_bagpipes_properties_dropdown.param.value, which_run.param.value, norm.param.value, total_fit_options.param.value, watch = True)
    '''
    sed_results_grid[2, 3:5] = sed_results
    '''
   
    # Alternative to SED results grid using Columns and Rows
    top_row = pn.Row(bin_map, sed_obj, height=350)
    mid_row = pn.Row(sfh_obj, sed_results, height=300)
    bot_row = pn.Row(corner_obj, pn.Column(pn.Row(other_bagpipes_properties_dropdown), pn.Row(norm), pn.Row(other_bagpipes_results)), height=750)
    combined = pn.Column(top_row, mid_row, bot_row, scroll=False)

    empty_phot_page = pn.Column(loading = True, sizing_mode='stretch_both', min_height=400, min_width = 400)#, max_width=1000)
    #phot_prop = pn.param.ParamFunction(phot_prop_func,  lazy=True)

    galaxy_tabs = pn.Tabs(('Cutouts', cutout_grid), ('SED Results', combined), ('Photometric Properties', empty_phot_page), dynamic=True, scroll=False)
    
    if subprocess.getstatusoutput('which fitsmap')[0] == 0:
        fitsmap_page = pn.param.ParamFunction(fitsmap, watch=False, lazy = True)
        #fitsmap_page = pn.bind(fits, watch=False)
        galaxy_tabs.append(('Fitsmap', fitsmap_page))
   
    tabs.append((f"{id} ({survey})", galaxy_tabs))

    # Only show sidebar if tab is SED Results
    #@pn.depends(which_map.param.value)
    update_sidebar(0, sidebar)

    pn.bind(update_sidebar, galaxy_tabs.param.active, sidebar, watch=True)


def resolved_sed_interface():
    global file_input

    tabs = pn.Tabs(closable=True, dynamic=True, scroll=False, min_height=2000)
    
    file_input = pn.widgets.FileInput(accept='.h5')
    
    sidebar = pn.Column("### Upload .h5", file_input)
    
    components = [sidebar, tabs]

    pn.bind(handle_file_upload, file_input, components, watch=True)

    return pn.template.FastListTemplate(
        title="Resolved SED Viewer", sidebar=[sidebar], main=[tabs], accent=ACCENT
    )

#resolved_sed_interface().servable()

pn.serve(resolved_sed_interface, websocket_max_message_size=MAX_SIZE_MB*1024*1024,
    http_server_kwargs={'max_buffer_size': MAX_SIZE_MB*1024*1024})







