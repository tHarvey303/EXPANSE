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
sys.path.append('/usr/local/texlive/')

mpl.use('macOsX')
mpl.rcParams['text.usetex'] = True 
#sns.set_context("paper")
#plt.style.use('paper.mplstyle')

# panel serve ResolvedSEDApp.py --autoreload

ACCENT = "goldenrod"
LOGO = "https://assets.holoviz.org/panel/tutorials/matplotlib-logo.png"

pn.extension(sizing_mode="stretch_width", design='material')
pn.extension('gridstack')
facecolor = '#f7f7f7'


@pn.cache
def get_h5(url):
    response = requests.get(url)
    return h5py.File(BytesIO(response.content), 'r')

def plot_sed(x, y, galaxy, cmap, shown_bins):
    if not (0 < x < galaxy.cutout_size and 0 < y < galaxy.cutout_size):
        x = galaxy.cutout_size // 2
        y = galaxy.cutout_size // 2
    
    bin = galaxy.pixedfit_map[int(np.ceil(y)), int(np.ceil(x))]
    #if bin in shown_bins:
    #    shown_bins.remove(bin)
    #else:
    #    shown_bins.append(bin)
    if bin not in shown_bins:
        shown_bins.append(bin)
    print(shown_bins)
    fig, ax = plt.subplots(figsize=(6, 3), constrained_layout=True, facecolor=facecolor)
    
    for bin in shown_bins:
        cmap = cm.get_cmap(cmap)
        color = cmap(bin/np.nanmax(galaxy.pixedfit_map))
        ax.text(0.5, 0.5, f"SED bin: {bin}", fontsize=20, ha='center', va='center', color=color)
    
    ax.set_xlabel(r"$\rm{Wavelength} \ (\mu m)$")
    ax.set_ylabel(r"$\rm{Flux Density} \ (\mu Jy)$")
    
    return pn.pane.Matplotlib(fig, dpi=144, tight=True, format="svg", sizing_mode="stretch_width")
    #return False

def handle_file_upload(value, tabs):
    global shown_bins
    file = BytesIO(value)
    resolved_galaxy = ResolvedGalaxy.init_from_h5(file)

    
    id = resolved_galaxy.galaxy_id
    survey = resolved_galaxy.survey

    cutout_grid = GridStack(sizing_mode='stretch_both', allow_resize=True, allow_drag=True, min_height=400)

    sed_results_grid = GridStack(sizing_mode='stretch_both', allow_resize=True, allow_drag=True, min_height=400)

    cutout_grid[0:2, :] = pn.Row(pn.pane.Matplotlib(resolved_galaxy.plot_cutouts(facecolor=facecolor), dpi=144, max_height=400,  tight=True, format="svg", sizing_mode="stretch_width"))

    #cmap = cm.get_cmap('viridis')
    # Get the RGB values of the colormap 
    #color_based_on_xy = lambda x, y: cmap(array[x, y, 0])
    # Make random 64 x 64 x 3 grid
    array = resolved_galaxy.pixedfit_map 
    im_array = xr.DataArray(array, dims=['y', 'x'], name='pixedfit bins', coords={'x': np.linspace(0, 64, 64), 'y': np.linspace(0, 64, 64)})
    cmap = 'viridis'
    hvplot = im_array.hvplot('x','y').opts(cmap=cmap, xaxis=None, yaxis=None, clabel='Pixedfit Bin')
    stream = hv.streams.Tap(source=hvplot, x=31, y=31)
    image = pn.pane.HoloViews(hvplot, sizing_mode='stretch_height')

    
    #pn.bind(plot_sed, stream.param.x, stream.param.y, resolved_galaxy, watch=True)
    #wavseries = im_array.interactive.sel(x=stream.param.x, y=stream.param.y,
    #                            method="nearest").hvplot('wav').opts(width=350, height=350, color='black')
    shown_bins = []
    obj = pn.bind(plot_sed, stream.param.x, stream.param.y, resolved_galaxy, cmap, shown_bins, watch=False)
    #pn.Column(image, wavseries.dmap())
    sed_results_grid[0, :2] = image
    sed_results_grid[0, 2:] = obj
    sed_results =  pn.Row(pn.pane.Matplotlib(resolved_galaxy.plot_bagpipes_results(facecolor=facecolor), dpi=144, tight=True, format="svg", sizing_mode="stretch_width"))
    sed_results_grid[2, :5] = sed_results
    galaxy_tabs = pn.Tabs(('Cutouts', cutout_grid), ('SED Results', sed_results_grid), dynamic=True)

    tabs.append((f"{id} ({survey})", galaxy_tabs))
    # Extract data etc
    #tab.object = object
    return tabs

def resolved_sed_interface():

    ## State
        
    #gstack[0, 0:2] = hv_pane1
    #gstack[0, 2:4] = hv_pane2

    tabs = pn.Tabs(closable=True, dynamic=True)

    file_input = pn.widgets.FileInput(accept='.h5')

    pn.bind(handle_file_upload, file_input, tabs, watch=True)
    file_input_component = pn.Column("### Upload .h5", file_input)

    return pn.template.FastListTemplate(
        title="Resolved SED Viewer", sidebar=[file_input_component], main=[tabs], accent=ACCENT
    )

resolved_sed_interface().servable()







