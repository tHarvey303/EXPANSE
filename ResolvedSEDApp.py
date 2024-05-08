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
from astropy import units as u
sys.path.append('/usr/local/texlive/')

mpl.use('macOsX')

mpl.rcParams['text.usetex'] = True 
#sns.set_context("paper")
#plt.style.use('paper.mplstyle')

# panel serve ResolvedSEDApp.py --autoreload

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


stream = hv.streams.Tap(transient=True)

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
    options = list(resolved_galaxy.sed_fitting_table[sed_fitting_tool].keys())
    
    which_run.options = options
    which_run.value = options[0]

    return which_run

def update_sidebar(active_tab, sidebar):
    sidebar.clear()
    settings_sidebar = pn.Column(pn.layout.Divider(), "### Settings", name='settings_sidebar')
    
    if active_tab == 0:
        
        settings_sidebar.append('Red Channel')
        settings_sidebar.append(red_select)
        settings_sidebar.append('Green Channel') 
        settings_sidebar.append(green_select)
        settings_sidebar.append('Blue Channel')
        settings_sidebar.append(blue_select)
   
    elif active_tab == 1:
        settings_sidebar.append('#### Pixel Binning')
        settings_sidebar.append(which_map)
        settings_sidebar.append('#### SED Fitting Tool')
        settings_sidebar.append(which_sed_fitter)
        settings_sidebar.append(pn.bind(possible_runs_select, which_sed_fitter.param.value, watch=True))

        settings_sidebar.append('#### SED Plot Config')
        settings_sidebar.append(which_flux_unit)
        settings_sidebar.append(multi_choice_bins)

        pn.bind(update_image, which_map.param.value, watch=True)
        

    sidebar.append(settings_sidebar)

def handle_map_click(x, y, galaxy, cmap, which_map_param, which_sed_fitter_param, which_flux_unit_param, multi_choice_bins_param, which_run_param, mode='sed'):

    use = True
    if x == None or y == None:
        use = False
    if use:
        if not (0 < x < galaxy.cutout_size and 0 < y < galaxy.cutout_size):
            use = False
    
    map = galaxy.pixedfit_map
    if use:
        # Need logic here to know which map to use
        bin = map[int(np.ceil(y)), int(np.ceil(x))]
        if bin not in multi_choice_bins_param:
            multi_choice_bins.value = multi_choice_bins_param + [bin]
    
    multi_choice_bins_param_safe = [int(i) for i in multi_choice_bins_param]
    if mode == 'sed':
        obj = plot_sed(galaxy, map, cmap, which_map_param, which_sed_fitter_param, which_flux_unit_param, multi_choice_bins_param_safe, which_run_param)
    if mode == 'sfh':
        obj = plot_sfh(galaxy, map, cmap, which_map_param, which_sed_fitter_param, multi_choice_bins_param_safe, which_run_param)
    if mode == 'corner':
        obj = plot_corner(galaxy, map, cmap, which_map_param, which_sed_fitter_param, multi_choice_bins_param_safe, which_run_param)

    return obj 
    #except Exception as e:
    #    pn.state.notifications.error(f'Error: {e}', duration=2000)

def plot_sed(galaxy, map, cmap, which_map_param, which_sed_fitter_param, which_flux_unit_param, multi_choice_bins_param, which_run_param, x_unit = u.micron):
    
    table = resolved_galaxy.photometry_table['webbpsf'][which_map_param]
    # filter table to ID

    fig, ax = plt.subplots(figsize=(6, 3), constrained_layout=True, facecolor=facecolor)

        #ax.text(0.5, 0.5, f"SED bin: {bin}", fontsize=20, ha='center', va='center', color=color)
    if which_flux_unit_param == 'uJy':
        y_unit = u.uJy
    elif which_flux_unit_param == 'ABmag':
        y_unit = u.ABmag
    elif which_flux_unit_param == 'ergscma':
        y_unit = u.erg/u.s/u.cm**2/u.AA

    # Plot integrated SED
    bands = resolved_galaxy.bands
   
    aper_dict = resolved_galaxy.aperture_dict[str(0.32*u.arcsec)]
    
    flux = aper_dict['flux'] * u.Jy
    wave = aper_dict['wave'] * u.AA
    flux_err = aper_dict['flux_err'] * u.Jy
   
    if y_unit == u.ABmag:
        # Assymmetric error bars
        yerr = [[2.5*np.log10(flux[0].value/(flux[0].value - flux_err[0].value))], [2.5*np.log10(1 + flux_err[0].value/flux[0].value)]]
    else:
        yerr = flux_err.to(y_unit, equivalencies = u.spectral_density(wave)).value
    ax.errorbar(wave.to(x_unit).value, flux.to(y_unit, equivalencies = u.spectral_density(wave)).value, yerr=yerr, fmt='o', linestyle='none', color='red', label=r"$ 0.32''$", mec = 'black')
    # pad legend down and left
    list_of_markers = ['o', 's', 'v', '^', '<', '>', '1', '2', '3', '4', '8', 'p', 'P', '*', 'h', 'H', '+', 'x', 'X', 'D', 'd', '|', '_']
    colors = []
    for bin_pos, bin in enumerate(multi_choice_bins_param):
        cmap = cm.get_cmap(cmap)
        color = cmap(bin/np.nanmax(map))
        colors.append(color)
        table_row = table[table['ID'] == bin]
        # loop through markers
        if bin_pos > len(list_of_markers) - 1:
            bin_pos = bin_pos % len(list_of_markers)

        marker = list_of_markers[bin_pos]

        for pos, (wav, band) in enumerate(zip(wave, resolved_galaxy.bands)):
            flux = table_row[band]
            flux_err = table_row[f"{band}_err"]
            
            if len(flux) == 0:
                continue
            if flux[0] > 0:
                if flux_err[0]/flux[0] < 0.1:
                    flux_err = 0.1 * flux

           
            if y_unit == u.ABmag:
                # Assymmetric error bars
                yerr = [[2.5*np.log10(flux.value/(flux[0].value - flux_err[0].value))], [2.5*np.log10(1 + flux_err[0].value/flux[0].value)]]
            else:
                 yerr = flux_err.to(y_unit, equivalencies = u.spectral_density(wav)).value
            
            ax.errorbar(wav.to(x_unit).value, flux.to(y_unit, equivalencies = u.spectral_density(wav)).value, yerr=yerr, fmt=marker, linestyle='none', color=color, label = int(bin) if pos == 0 else '')
    
    fig.legend(loc='upper right', bbox_to_anchor=(0.99, 0.99), frameon=False)    
    if y_unit == u.ABmag:
        ax.invert_yaxis()
    ax.set_xlabel(rf"$\rm{{Wavelength}}$ ({x_unit:latex})", fontsize='large')
    ax.set_ylabel(rf"$\rm{{Flux \ Density}}$ ({y_unit:latex})", fontsize='large')
    ax.set_xlim(ax.get_xlim())
    ax.set_ylim(ax.get_ylim())

    if which_sed_fitter_param == 'bagpipes' and which_run_param != None:
        if which_run_param not in cache_pipes:
            cache_pipes[which_run_param] = {}
        cache = cache_pipes.get(which_run_param)
        fig, cache == resolved_galaxy.plot_bagpipes_fit(which_run_param, ax, fig, bins_to_show=multi_choice_bins_param, marker_colors=colors, wav_units=x_unit, flux_units=y_unit, cache=cache)
        cache_pipes[which_run_param] = cache

 
    return pn.pane.Matplotlib(fig, dpi=144, tight=True, format="svg", sizing_mode="scale_height")
    #return False

def plot_bins(bin_type, cmap):
    array = getattr(resolved_galaxy, f"{bin_type}_map")
    array[array == 0] = np.nan
    dimensions = np.linspace(0, resolved_galaxy.cutout_size, resolved_galaxy.cutout_size)
    im_array = xr.DataArray(array, dims=['y', 'x'], name=f'{bin_type} bins', coords={'x': dimensions, 'y': dimensions})
    hvplot = im_array.hvplot('x','y').opts(cmap=cmap, xaxis=None, yaxis=None, clabel=f'{bin_type} Bin')
    multi_choice_bins.options = list(np.unique(array))
    return hvplot

def plot_sfh(galaxy, map, cmap, which_map_param, which_sed_fitter_param, multi_choice_bins_param, which_run_param, x_unit = 'Gyr', facecolor='#f7f7f7'):
    
    cmap = cm.get_cmap(cmap)
    colors = [cmap(bin/np.nanmax(map)) for bin in multi_choice_bins_param]

    if which_sed_fitter_param == 'bagpipes':
        if which_run_param not in cache_pipes:
            cache_pipes[which_run_param] = {}

        bins_to_show = [int(i) for i in multi_choice_bins_param]
        cache = cache_pipes.get(which_run_param)
        fig, cache = resolved_galaxy.plot_bagpipes_sfh(run_name=which_run_param, bins_to_show = bins_to_show, save=False, facecolor=facecolor, marker_colors=colors, time_unit=x_unit, run_dir=run_dir, plotpipes_dir=plotpipes_dir, cache=cache) 
        cache_pipes[which_run_param] = cache

        if len(fig.get_axes()) != 0:
            ax = fig.get_axes()[0]
            ax.set_xlim(0, 1)
    

    return pn.pane.Matplotlib(fig, dpi=144, tight=True, format="svg", sizing_mode="stretch_width")

def plot_corner(galaxy, map, cmap, which_map_param, which_sed_fitter_param, multi_choice_bins_param, which_run_param, facecolor='#f7f7f7'):
    
    cmap = cm.get_cmap(cmap)
    colors = [cmap(bin/np.nanmax(map)) for bin in multi_choice_bins_param]

    if which_sed_fitter_param == 'bagpipes':
        if which_run_param not in cache_pipes:
            cache_pipes[which_run_param] = {}

        bins_to_show = [int(i) for i in multi_choice_bins_param]
        cache = cache_pipes.get(which_run_param)
        fig, cache = resolved_galaxy.plot_bagpipes_corner(run_name=which_run_param, bins_to_show = bins_to_show, save=False, facecolor=facecolor, colors=colors, run_dir=run_dir, plotpipes_dir=plotpipes_dir, cache=cache) 
        cache_pipes[which_run_param] = cache
        #ax = fig.get_axes()[0]
        #ax.set_xlim(0, 1)
        if fig == None:
            fig = plt.figure()

        fig.set_facecolor(facecolor)

        return pn.pane.Matplotlib(fig, dpi=144, tight=True, format="svg", sizing_mode="stretch_width")

    
    

    

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
    global which_run

    which_map = pn.widgets.RadioButtonGroup(options=['pixedfit', 'voronoi'], value='pixedfit', name='Pixel Binning')

    which_sed_fitter = pn.widgets.Select(name='SED Fitter', value='bagpipes', options=['bagpipes'])

    which_flux_unit = pn.widgets.Select(name='Flux Unit', value='uJy', options=['uJy', 'ABmag', 'ergscma'])

    multi_choice_bins = pn.widgets.MultiChoice(name='Bins', options=[], delete_button=True, placeholder='Click on bin map to add bins')

    which_run = pn.widgets.Select(name='Run', value=None, options=[])

    file = BytesIO(value)
    resolved_galaxy = ResolvedGalaxy.init_from_h5(file)

    sidebar, tabs = components

    id = resolved_galaxy.galaxy_id
    survey = resolved_galaxy.survey

    cutout_grid = GridStack(sizing_mode='stretch_both', allow_resize=True, allow_drag=True, min_height=400)

    sed_results_grid = GridStack(sizing_mode='stretch_both', allow_resize=True, allow_drag=True, min_height=800, mode='override')

    cutout_grid[0:2, :] = pn.Row(pn.pane.Matplotlib(resolved_galaxy.plot_cutouts(facecolor=facecolor), dpi=144, max_height=400,  tight=True, format="svg", sizing_mode="stretch_width"))
    
    cmap = 'nipy_spectral_r'
 
    hvplot_bins = plot_bins('pixedfit', cmap)
    
    bin_map = pn.pane.HoloViews(hvplot_bins, sizing_mode='stretch_height')
    # set stream off holoviews object instead
    stream.source = bin_map.object

    sed_obj = pn.bind(handle_map_click, stream.param.x, stream.param.y, resolved_galaxy, 
                            cmap, which_map.param.value, which_sed_fitter.param.value, 
                            which_flux_unit.param.value, multi_choice_bins.param.value,
                            which_run.param.value, 'sed', watch=False)
    
    sfh_obj = pn.bind(handle_map_click, stream.param.x, stream.param.y, resolved_galaxy, 
                            cmap, which_map.param.value, which_sed_fitter.param.value, 
                            which_flux_unit.param.value, multi_choice_bins.param.value,
                            which_run.param.value, 'sfh', watch=False)
    
    corner_obj = pn.bind(handle_map_click, stream.param.x, stream.param.y, resolved_galaxy, cmap,
                            which_map.param.value, which_sed_fitter.param.value, 
                            which_flux_unit.param.value, multi_choice_bins.param.value,
                            which_run.param.value, 'corner', watch=False)
    
    red_select = pn.widgets.MultiChoice(name='Red Channel', options = resolved_galaxy.bands, value = ['F444W'])
    green_select = pn.widgets.MultiChoice(name='Green Channel', options = resolved_galaxy.bands, value = ['F277W'])
    blue_select = pn.widgets.MultiChoice(name='Blue Channel', options = resolved_galaxy.bands, value = ['F150W'])

    #obj = pn.bind(plot_sed, bin_map.param.tap.x, bin_map.param.tap.y, resolved_galaxy, cmap, shown_bins, watch=True)

    sed_results_grid[0, :2] = bin_map
    sed_results_grid[0, 2:] = sed_obj
    sed_results_grid[2, :3] = sfh_obj
    sed_results_grid[3:5, :3] = corner_obj
    sed_results =  pn.Row(pn.pane.Matplotlib(resolved_galaxy.plot_bagpipes_results(facecolor=facecolor, parameters=['stellar_mass', 'sfr'], max_on_row=3), dpi=144, tight=True, format="svg", sizing_mode="stretch_width"))
    sed_results_grid[2, 3:5] = sed_results
    
    galaxy_tabs = pn.Tabs(('Cutouts', cutout_grid), ('SED Results', sed_results_grid), dynamic=True)
    
    tabs.append((f"{id} ({survey})", galaxy_tabs))

    # Only show sidebar if tab is SED Results
    #@pn.depends(which_map.param.value)

    pn.bind(update_sidebar, galaxy_tabs.param.active, sidebar, watch=True)

def plot_rgb(galaxy, red, green, blue, scale):
    fig, ax = plt.subplots(figsize=(6, 3), constrained_layout=True, facecolor=facecolor)
    ax.imshow(galaxy.rgb_image(red, green, blue, scale), origin='lower')
    ax.axis('off')
    return pn.pane.Matplotlib(fig, dpi=144, tight=True, format="svg", sizing_mode="stretch_width")
    

def resolved_sed_interface():

    tabs = pn.Tabs(closable=True, dynamic=True)
    
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







