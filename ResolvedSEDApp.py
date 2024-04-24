import panel as pn
import numpy as np
from panel.layout.gridstack import GridStack
from matplotlib.figure import Figure
import xarray as xr
import holoviews as hv
import hvplot.xarray
import matplotlib.cm as cm
from ResolvedGalaxy import ResolvedGalaxy
import h5py
from io import BytesIO
# panel serve ResolvedSEDApp.py --autoreload

ACCENT = "goldenrod"
LOGO = "https://assets.holoviz.org/panel/tutorials/matplotlib-logo.png"

pn.extension(sizing_mode="stretch_width", design='material')
pn.extension('gridstack')


@pn.cache
def get_h5(url):
    response = requests.get(url)
    return h5py.File(BytesIO(response.content), 'r')

def handle_file_upload(value, tabs):
    file = BytesIO(value)
    resolved_galaxy = ResolvedGalaxy.init_from_h5(file)

    id = resolved_galaxy.galaxy_id
    survey = resolved_galaxy.survey

    
    # Could I instatiate the object here?
    # Extract data from h5
    # field, id, etc

    tabs.append(pn.Column(f"### {id} ({survey})"))
    # Extract data etc
    #tab.object = object
    return tabs

def resolved_sed_interface():

    ## State
        
    gstack = GridStack(sizing_mode='stretch_both', min_height=600, allow_resize=True, allow_drag=True, ncols=5, nrows=3)

    #gstack[0, 0:2] = hv_pane1
    #gstack[0, 2:4] = hv_pane2

    tabs = pn.Tabs(closable=True)

    file_input = pn.widgets.FileInput(accept='.h5,.fits,.png')

    pn.bind(handle_file_upload, file_input, tabs, watch=True)
    file_input_component = pn.Column("### Upload .h5", file_input)

    

    return pn.template.FastListTemplate(
        title="Resolved SED Viewer", sidebar=[file_input_component], main=[tabs], accent=ACCENT
    )

resolved_sed_interface().servable()







