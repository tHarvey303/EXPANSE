import panel as pn
import param
import holoviews as hv
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
import h5py as h5
from astropy.io import fits
import functools
from astropy.wcs import WCS
import click
import copy
from functools import lru_cache
import os
from .ResolvedGalaxy import ResolvedGalaxy, MockResolvedGalaxy
from astropy import units as u
import xarray as xr
from bokeh.models import PrintfTickFormatter, Label
import matplotlib.cm as cm
from holoviews import opts, streams
from matplotlib.colors import Normalize
import param
import functools
import inspect
import copy

MAX_SIZE_MB = 150
ACCENT = "goldenrod"
galaxies_dir = os.path.join(
    os.path.dirname(
        os.path.dirname((os.path.dirname(os.path.abspath(__file__))))
    ),
    "galaxies",
)


def custom_depends(*dependencies, watch=False):
    def decorator(func):
        @param.depends(*dependencies, watch=watch)
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            triggered = [
                dep
                for dep in dependencies
                if self.param[dep.split(".")[-1]].changed
            ]
            return func(self, triggered=triggered, *args, **kwargs)

        return wrapper

    return decorator


def check_dependencies(cache_attr="_cache"):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            # Get the current function name
            current_function = func.__name__

            # Get the dependencies for this function
            dependencies = [
                o.name
                for o in self.param.method_dependencies(current_function)
            ]
            print(
                f"Current function: {current_function}, Dependencies: {dependencies}"
            )

            # Get the current values of the dependencies
            current_args = [
                copy.deepcopy(getattr(self, dep)) for dep in dependencies
            ]

            # Check if we have a cache for this function
            if not hasattr(self, cache_attr):
                setattr(self, cache_attr, {})
            cache = getattr(self, cache_attr)

            if current_function not in cache:
                print("First time running")
                result = func(self, *args, **kwargs)
                cache[current_function] = {
                    "args": current_args,
                    "result": result,
                }
                return result

            # Check if any of the arguments have changed
            if current_args != cache[current_function]["args"]:
                print("Arguments have changed")
                result = func(self, *args, **kwargs)
                cache[current_function]["args"] = current_args
                cache[current_function]["result"] = result
                return result
            else:
                print("Arguments have not changed")
                return cache[current_function]["result"]

        return wrapper

    return decorator


class GalaxyTab(param.Parameterized):
    # Parameters for SED results tab
    galaxy_id = param.String()
    survey = param.String()
    # resolved_galaxy = param.Parameter()
    active_tab = param.Integer(default=0)
    which_map = param.Selector(
        default="pixedfit", objects=["pixedfit", "voronoi", "pixel-by-pixel"]
    )
    which_sed_fitter = param.Selector(default="bagpipes", objects=["bagpipes"])
    which_flux_unit = param.Selector(
        default="uJy", objects=["uJy", "ABmag", "ergscma"]
    )
    multi_choice_bins = param.List(default=["RESOLVED"])
    which_run_aperture = param.String(default=None)
    which_run_resolved = param.String(default=None)
    total_fit_options = param.List(default=["TOTAL_BIN"])
    show_sed_photometry = param.Selector(
        default="Don't show", objects=["Show", "Don't show"]
    )
    scale_alpha = param.Number(default=1, bounds=(0, 1))
    show_galaxy = param.Boolean(default=False)
    show_kron = param.Boolean(default=False)
    psf_mode = param.Selector(
        default="PSF Matched", objects=["PSF Matched", "Original"]
    )
    choose_show_band = param.Selector(default="F444W")
    other_bagpipes_properties = param.Selector(default="mass_weighted_age")
    norm = param.Selector(default="linear", objects=["linear", "log"])
    type_select = param.Selector(default="median", objects=["median", "gif"])
    pdf_param_property = param.Selector(
        default="stellar_mass", objects=["stellar_mass", "sfr"]
    )
    upscale_select = param.Boolean(default=True)

    # Parameters for SED accordian
    sed_log_x = param.Boolean(
        default=False, doc="Use logarithmic x-axis for SED plot"
    )
    sed_log_y = param.Boolean(
        default=False, doc="Use logarithmic y-axis for SED plot"
    )
    sed_x_min = param.Number(
        default=0.3, doc="Minimum x-axis value for SED plot"
    )
    sed_x_max = param.Number(
        default=5, doc="Maximum x-axis value for SED plot"
    )
    sed_y_min = param.Number(
        default=None, allow_None=True, doc="Minimum y-axis value for SED plot"
    )
    sed_y_max = param.Number(
        default=None, allow_None=True, doc="Maximum y-axis value for SED plot"
    )

    # Parameters for cutouts tab
    red_channel = param.List(default=["F444W"])
    green_channel = param.List(default=["F277W"])
    blue_channel = param.List(default=["F150W"])
    stretch = param.Number(default=6, bounds=(0.001, 10))
    q = param.Number(default=0.001, bounds=(0.000001, 0.01))
    psf_mode = param.Selector(
        default="PSF Matched", objects=["PSF Matched", "Original"]
    )
    band_select = param.Selector(default="F444W")
    other_plot_select = param.Selector(
        default="Galaxy Region",
        objects=["Galaxy Region", "Fluxes", "Segmentation Map", "Radial SNR"],
    )

    # Parameters for interactive tab
    interactive_band = param.Selector(default="F444W")
    drawn_shapes = param.Dict(default={})
    which_interactive_sed_fitter = param.Selector(
        default="EAZY-py", objects=["EAZY-py", "Dense Basis"]
    )

    interactive_aperture_radius = param.Number(default=5.33, bounds=(1, 40))

    TOTAL_FIT_COLORS = {
        "TOTAL_BIN": "springgreen",
        "MAG_AUTO": "blue",
        "MAG_APER": "sandybrown",
        "MAG_ISO": "purple",
        "MAG_APER_TOTAL": "orange",
        "MAG_BEST": "cyan",
        "RESOLVED": "red",
    }

    @param.depends()
    def __init__(self, resolved_galaxy, facecolor="#f7f7f7", **params):
        galaxy_id = resolved_galaxy.galaxy_id
        survey = resolved_galaxy.survey

        super().__init__(
            galaxy_id=galaxy_id,
            survey=survey,
            **params,
        )

        self.resolved_galaxy = resolved_galaxy

        self.facecolor = facecolor
        self.setup_widgets()
        self.setup_streams()
        self.setup_tabs()

    @param.depends()
    def setup_widgets(self):
        self.which_map_widget = pn.widgets.RadioButtonGroup(
            name="Pixel Binning",
            options=["pixedfit", "voronoi", "pixel-by-pixel"],
            value=self.which_map,
        )
        self.which_map_widget.link(self, value="which_map")

        self.which_sed_fitter_widget = pn.widgets.Select(
            name="SED Fitter",
            options=["bagpipes"],
            value=self.which_sed_fitter,
        )
        self.which_sed_fitter_widget.link(self, value="which_sed_fitter")

        self.which_flux_unit_widget = pn.widgets.Select(
            name="Flux Unit",
            options=["uJy", "ABmag", "ergscma"],
            value=self.which_flux_unit,
        )
        self.which_flux_unit_widget.link(self, value="which_flux_unit")

        self.multi_choice_bins_widget = pn.widgets.MultiChoice(
            name="Bins",
            options=["RESOLVED"],
            value=self.multi_choice_bins,
            delete_button=True,
            placeholder="Click on bin map to add bins",
        )
        self.multi_choice_bins_widget.link(
            self, value="multi_choice_bins", bidirectional=True
        )

        self.total_fit_options_widget = pn.widgets.MultiChoice(
            name="Aperture Fits",
            options=[
                "TOTAL_BIN",
                "MAG_AUTO",
                "MAG_BEST",
                "MAG_ISO",
                "MAG_APER_TOTAL",
            ],
            value=self.total_fit_options,
            delete_button=True,
            placeholder="Select combined fits to show",
        )
        self.total_fit_options_widget.link(self, value="total_fit_options")

        self.show_sed_photometry_widget = pn.widgets.Select(
            name="Fit Photometry",
            options=["Show", "Don't show"],
            value=self.show_sed_photometry,
        )

        self.which_run_resolved_widget = pn.widgets.Select(
            name="Run Name", options=[], value=self.which_run_resolved
        )

        self.which_run_resolved_widget.link(self, value="which_run_resolved")

        self.which_run_aperture_widget = pn.widgets.Select(
            name="Run Name",
            options=[],
            value=self.which_run_aperture,
        )
        self.which_run_aperture_widget.link(self, value="which_run_aperture")

        # Call the possible_runs_select method to update the options
        self.possible_runs_select(self.which_sed_fitter)

        self.show_sed_photometry_widget.link(self, value="show_sed_photometry")

        self.scale_alpha_widget = pn.widgets.FloatSlider(
            name="Scale Alpha",
            start=0,
            end=1,
            value=self.scale_alpha,
            step=0.01,
        )
        self.scale_alpha_widget.link(self, value="scale_alpha")

        self.show_galaxy_widget = pn.widgets.Checkbox(
            name="Show Galaxy", value=self.show_galaxy
        )
        self.show_galaxy_widget.link(self, value="show_galaxy")

        self.show_kron_widget = pn.widgets.Checkbox(
            name="Show Kron", value=self.show_kron
        )
        self.show_kron_widget.link(self, value="show_kron")

        self.psf_mode_widget = pn.widgets.Select(
            name="PSF Mode",
            options=["PSF Matched", "Original"],
            value=self.psf_mode,
        )
        self.psf_mode_widget.link(self, value="psf_mode")

        self.choose_show_band_widget = pn.widgets.Select(
            name="Show Band",
            options=self.resolved_galaxy.bands,
            value=self.choose_show_band,
        )
        self.choose_show_band_widget.link(self, value="choose_show_band")

        self.other_bagpipes_properties_widget = pn.widgets.Select(
            name="SED Properties Map",
            options=["mass_weighted_age"],
            value=self.other_bagpipes_properties,
        )
        self.other_bagpipes_properties_widget.link(
            self, value="other_bagpipes_properties"
        )

        self.norm_widget = pn.widgets.RadioButtonGroup(
            name="Scaling",
            options=["linear", "log"],
            value=self.norm,
            button_type="primary",
        )
        self.norm_widget.link(self, value="norm")

        self.type_select_widget = pn.widgets.Select(
            name="Type",
            options=["median", "gif"],
            value=self.type_select,
            width=90,
        )
        self.type_select_widget.link(self, value="type_select")

        self.pdf_param_property_widget = pn.widgets.Select(
            name="PDF Property",
            options=["stellar_mass", "sfr"],
            value=self.pdf_param_property,
        )
        self.pdf_param_property_widget.link(self, value="pdf_param_property")

        self.upscale_select_widget = pn.widgets.Select(
            name="Upscale Mass/SFR",
            options=[True, False],
            value=self.upscale_select,
        )
        self.upscale_select_widget.link(self, value="upscale_select")

        # Cutouts tab widgets
        self.red_channel_widget = pn.widgets.MultiChoice(
            name="Red Channel",
            options=self.resolved_galaxy.bands,
            value=self.red_channel,
        )
        self.red_channel_widget.link(self, value="red_channel")

        self.green_channel_widget = pn.widgets.MultiChoice(
            name="Green Channel",
            options=self.resolved_galaxy.bands,
            value=self.green_channel,
        )
        self.green_channel_widget.link(self, value="green_channel")

        self.blue_channel_widget = pn.widgets.MultiChoice(
            name="Blue Channel",
            options=self.resolved_galaxy.bands,
            value=self.blue_channel,
        )
        self.blue_channel_widget.link(self, value="blue_channel")

        self.stretch_widget = pn.widgets.EditableFloatSlider(
            name="Stretch", start=0.001, end=10, value=self.stretch, step=0.001
        )
        self.stretch_widget.link(self, value="stretch")

        self.q_widget = pn.widgets.EditableFloatSlider(
            name="Q", start=0.000001, end=0.01, value=self.q, step=0.000001
        )
        self.q_widget.link(self, value="q")

        self.band_select_widget = pn.widgets.Select(
            name="Band",
            options=self.resolved_galaxy.bands,
            value=self.band_select,
        )
        self.band_select_widget.link(self, value="band_select")

        self.other_plot_select_widget = pn.widgets.Select(
            options=[
                "Galaxy Region",
                "Fluxes",
                "Segmentation Map",
                "Radial SNR",
            ],
            value=self.other_plot_select,
        )
        self.other_plot_select_widget.link(self, value="other_plot_select")

        self.sed_log_x_widget = pn.widgets.Checkbox(
            name="Log X-axis", value=self.sed_log_x
        )
        self.sed_log_y_widget = pn.widgets.Checkbox(
            name="Log Y-axis", value=self.sed_log_y
        )
        self.sed_x_min_widget = pn.widgets.FloatInput(
            name="X-axis Min", value=self.sed_x_min
        )
        self.sed_x_max_widget = pn.widgets.FloatInput(
            name="X-axis Max", value=self.sed_x_max
        )
        self.sed_y_min_widget = pn.widgets.FloatInput(
            name="Y-axis Min", value=self.sed_y_min
        )
        self.sed_y_max_widget = pn.widgets.FloatInput(
            name="Y-axis Max", value=self.sed_y_max
        )

        self.sed_log_x_widget.link(self, value="sed_log_x")
        self.sed_log_y_widget.link(self, value="sed_log_y")
        self.sed_x_min_widget.link(self, value="sed_x_min")
        self.sed_x_max_widget.link(self, value="sed_x_max")
        self.sed_y_min_widget.link(self, value="sed_y_min")
        self.sed_y_max_widget.link(self, value="sed_y_max")

        self.sed_plot_controls = pn.Accordion(
            (
                "SED Plot Controls",
                pn.Column(
                    pn.Row(self.sed_log_x_widget, self.sed_log_y_widget),
                    pn.Row(self.sed_x_min_widget, self.sed_x_max_widget),
                    pn.Row(self.sed_y_min_widget, self.sed_y_max_widget),
                ),
            ),
            active=[0],  # Open by default
        )

        self.interactive_photometry_button = pn.widgets.Button(
            name="Measure Photometry", button_type="primary"
        )
        self.interactive_sed_fitting_dropdown = pn.widgets.Select(
            name="SED Fitting",
            options=["EAZY-py", "Dense Basis"],
            value=self.which_interactive_sed_fitter,
        )
        self.interactive_sed_fitting_dropdown.link(
            self, value="which_interactive_sed_fitter"
        )

        self.interactive_sed_fitting_button = pn.widgets.Button(
            name="Run SED Fitting", button_type="primary"
        )
        self.interactive_band_widget = pn.widgets.Select(
            name="Band",
            options=self.resolved_galaxy.bands,
            value=self.interactive_band,
        )

        self.interactive_band_widget.link(self, value="interactive_band")

        self.save_shape_photometry_button = pn.widgets.Button(
            name="Save regions", button_type="success"
        )

        self.clear_interactive_button = pn.widgets.Button(
            name="Clear Shapes", button_type="danger"
        )

        self.interactive_poly = hv.Polygons([])
        self.interactive_boxes = hv.Rectangles([])
        self.interactive_points = hv.Points([])

        self.interactive_aperture_radius_widget = (
            pn.widgets.EditableFloatSlider(
                name="Aperture Radius (pixels)",
                start=1,
                end=40,
                value=5.33,
                step=0.01,
            )
        )

        self.interactive_aperture_radius_widget.link(
            self, value="interactive_aperture_radius"
        )

    @param.depends("point_selector.point")
    def update_selected_bins(self):
        if self.point_selector.point:
            x, y = self.point_selector.point
            bin_map = getattr(self.resolved_galaxy, f"{self.which_map}_map")
            selected_bin = int(bin_map[int(y), int(x)])

            if (
                selected_bin not in self.multi_choice_bins
                and selected_bin != 0
            ):
                self.multi_choice_bins = self.multi_choice_bins + [
                    selected_bin
                ]
            elif selected_bin in self.multi_choice_bins:
                self.multi_choice_bins = [
                    bin
                    for bin in self.multi_choice_bins
                    if bin != selected_bin
                ]

            # Update the widget
            self.multi_choice_bins_widget.value = self.multi_choice_bins

    @param.depends(
        # "resolved_galaxy",
        "which_map",
        "which_run_resolved",
        "which_sed_fitter",
        "norm",
        "upscale_select",
        "type_select",
    )
    @check_dependencies()
    def plot_bagpipes_results(
        self, parameters=["stellar_mass", "sfr"], max_on_row=3
    ):
        if self.which_sed_fitter == "bagpipes":
            sed_results_plot = self.resolved_galaxy.plot_bagpipes_results(
                facecolor=self.facecolor,
                parameters=parameters,
                max_on_row=max_on_row,
                run_name=self.which_run_resolved,
                norm=self.norm,
                weight_mass_sfr=self.upscale_select,
            )

            if sed_results_plot is not None:
                plt.close(sed_results_plot)
                sed_results = pn.Row(
                    pn.pane.Matplotlib(
                        sed_results_plot,
                        dpi=144,
                        tight=True,
                        format="svg",
                        sizing_mode="scale_both",
                    )
                )
            else:
                sed_results = pn.pane.Markdown("No Bagpipes results found.")
        else:
            sed_results = pn.pane.Markdown("Not implemented yet.")

        return sed_results

    @param.depends()
    def plot_bagpipes_table(self):
        table_pd = self.resolved_galaxy.compose_bagpipes_pandas_table()

        from bokeh.models.widgets.tables import (
            NumberFormatter,
            HTMLTemplateFormatter,
        )
        # Format floats to 3 significant figures
        # bokeh_formatters = {
        # Render columns with tags as HTML

        # Write a custom HTML template which displays upper and lower errors (given the other columns for the 16th and 84th percentiles)

        formatter = HTMLTemplateFormatter(template="<div><%= value %></div>")

        error_formatter = (
            lambda value,
            err_l,
            err_u: f"<div><%= value %><sup>+<%= {err_u} %></sup><sub>-<%= {err_l} %></sub></div>"
        )

        bokeh_formatters = {}

        # get table column names
        columns = table_pd.columns

        ignore_cols = []
        html_cols = []
        """
        for col in columns:
            if col.endswith("err"):
                ignore_cols.append(col)
                if col[:-4] not in html_cols:
                    
                    html_cols.append(col[:-4])
        """

        for col in columns[2:]:  # html_cols:
            bokeh_formatters[col] = formatter

        table = pn.widgets.Tabulator(
            table_pd,
            pagination="remote",
            page_size=10,
            height=400,
            theme="bootstrap",
            layout="fit_data_table",
            hidden_columns=ignore_cols,
            show_index=False,
            formatters=bokeh_formatters,
            selectable=False,
        )

        return table

    @param.depends()
    def create_sed_results_tab(self):
        sed_results_grid = pn.Column(sizing_mode="stretch_both")

        # Add the update_selected_bins method to be called when the bin map is clicked
        # bin_map = pn.panel(bin_map).add_periodic_callback(self.update_selected_bins, period=100)

        top_row = pn.Row(
            self.plot_bins,
            pn.param.ParamMethod(self.plot_sed, loading_indicator=True),
            sizing_mode="stretch_width",
            height=350,
        )

        # sfh_plot = pn.pane.Matplotlib(self.plot_sfh, sizing_mode='stretch_both')
        mid_row = pn.Row(
            self.plot_sfh,
            self.plot_bagpipes_results,
            height=300,
            min_height=300,
        )

        bagpipes_controls = pn.Column(
            self.other_bagpipes_properties_widget,
            pn.Row(self.norm_widget, self.type_select_widget),
            sizing_mode="stretch_width",
        )

        bot_row = pn.Row(
            self.plot_corner,
            pn.Column(
                bagpipes_controls,
                pn.Row(self.other_bagpipes_results_func),
            ),
            height=750,
        )

        pdf_controls = pn.Column(
            self.pdf_param_property_widget, sizing_mode="stretch_width"
        )

        even_lower = pn.Row(
            pn.Column(pdf_controls, self.plot_bagpipes_pdf, max_height=400),
            self.plot_bagpipes_table,
            # sizing_mode='stretch_both'
            max_height=400,
            max_width=1000,
        )

        sed_results_grid.extend([top_row, mid_row, bot_row, even_lower])

        return sed_results_grid

    @param.depends()
    def update_active_tab(self, event):
        self.active_tab = event.new
        if hasattr(self, "app"):
            print("Updating sidebar")
            self.app.update_sidebar()
        else:
            print("No app attribute")
            print("Warning: No app attribute found. Cannot update sidebar.")

    @param.depends()
    def setup_streams(self):
        self.point_selector = hv.streams.Tap(
            transient=True, source=hv.Image(([]))
        )
        # self.point_selector = hv.streams.PointSelector(

    @param.depends()
    def setup_tabs(self):
        self.cutouts_tab = pn.param.ParamMethod(
            self.create_cutouts_tab, loading_indicator=True, lazy=True
        )
        self.sed_results_tab = pn.param.ParamMethod(
            self.create_sed_results_tab, loading_indicator=False, lazy=True
        )
        self.photometric_properties_tab = pn.param.ParamMethod(
            self.create_photometric_properties_tab,
            loading_indicator=True,
            lazy=True,
        )

        self.info_tabs = pn.Tabs(
            ("Cutouts", self.cutouts_tab),
            ("SED Results", self.sed_results_tab),
            ("Photometric Properties", self.photometric_properties_tab),
            dynamic=True,
            scroll=False,
        )

        self.interactive_tab = pn.panel(
            self.create_interactive_tab, loading_indicator=True
        )
        self.info_tabs.append(("Interactive", self.interactive_tab))

        if isinstance(self.resolved_galaxy, MockResolvedGalaxy):
            self.synthesizer_tab = pn.panel(
                self.create_synthesizer_tab, loading_indicator=True
            )
            self.info_tabs.append(("Synthesizer", self.synthesizer_tab))

        self.info_tabs.param.watch(self.update_active_tab, "active")

    @param.depends()
    def create_cutouts_tab(self):
        cutout_grid = pn.Column(scroll=True, sizing_mode="stretch_width")

        # Cutouts row
        cutouts_row = pn.Row(
            self.plot_cutouts,
            sizing_mode="stretch_width",
            scroll=True,
        )
        cutout_grid.append(cutouts_row)

        rgb_row = pn.Row(
            pn.Column("### RGB Image", self.plot_rgb),
            sizing_mode="stretch_width",
        )
        cutout_grid.append(rgb_row)

        # SNR Map row
        snr_row = pn.Row(
            pn.Column(
                "### SNR Map", self.band_select_widget, self.do_snr_plot
            ),
            sizing_mode="stretch_width",
        )
        cutout_grid.append(snr_row)

        # Other Plots row
        other_plots_row = pn.Row(
            pn.Column(
                "### Other Plots",
                self.other_plot_select_widget,
                self.do_other_plot,
            ),
            sizing_mode="stretch_width",
        )
        cutout_grid.append(other_plots_row)

        return cutout_grid

    @param.depends("band_select")
    @check_dependencies()
    def do_snr_plot(self):
        fig = self.resolved_galaxy.plot_snr_map(
            band=self.band_select, facecolor=self.facecolor
        )
        plt.close(fig)
        return pn.pane.Matplotlib(
            fig, dpi=144, tight=True, format="svg", width=300, height=300
        )

    @param.depends("other_plot_select")
    @check_dependencies()
    def do_other_plot(self):
        try:
            if self.other_plot_select == "Galaxy Region":
                fig = self.resolved_galaxy.plot_gal_region(
                    facecolor=self.facecolor
                )
            elif self.other_plot_select == "Fluxes":
                fig = self.resolved_galaxy.pixedfit_plot_map_fluxes()
            elif self.other_plot_select == "Segmentation Map":
                fig = self.resolved_galaxy.plot_seg_stamps()
            elif self.other_plot_select == "Radial SNR":
                fig = self.resolved_galaxy.pixedfit_plot_radial_SNR()
        except Exception as e:
            print(e)
            fig = plt.figure()

        plt.close(fig)
        return pn.pane.Matplotlib(
            fig,
            dpi=144,
            tight=True,
            format="svg",
            max_width=1000,
            sizing_mode="stretch_both",
        )

    @param.depends(
        # "resolved_galaxy",
        "which_map",
        "other_bagpipes_properties",
        "which_run_resolved",
        "norm",
        "total_fit_options",
        "type_select",
    )
    @check_dependencies()
    def other_bagpipes_results_func(self, triggered=None):
        if triggered is not None:
            # print function name and trigger
            print("other_bagpipes_results_func triggered by", triggered)

        param_property = self.other_bagpipes_properties
        p_run_name = self.which_run_resolved
        norm_param = self.norm
        total_params = self.total_fit_options
        weight_mass_sfr = self.upscale_select
        outtype = self.type_select

        if "bagpipes" not in self.resolved_galaxy.sed_fitting_table.keys():
            return pn.pane.Markdown("No Bagpipes results found.")

        options = list(
            self.resolved_galaxy.sed_fitting_table["bagpipes"][
                p_run_name
            ].keys()
        )
        options = [
            i
            for i in options
            if not (i.startswith("#") or i.endswith("16") or i.endswith("84"))
        ]
        actual_options = [i.replace("_50", "") for i in options]
        dist_options = [i for i in options if i.endswith("50")]
        self.pdf_param_property_widget.options = [
            i.replace("_50", "") for i in dist_options
        ]
        self.other_bagpipes_properties_widget.options = actual_options
        self.other_bagpipes_properties_widget.value = param_property

        no_log = ["ssfr"]
        if param_property in no_log:
            self.norm_widget.value = "linear"
            self.norm_widget.disabled = True
        else:
            self.norm_widget.disabled = False

        if f"{param_property}_50" not in dist_options:
            if not param_property.endswith("-"):
                param_property = f"{param_property}-"

        if outtype == "gif":
            logmap = False
            if param_property in ["stellar_mass", "sfr"]:
                if norm_param == "log":
                    logmap = True
            else:
                weight_mass_sfr = False

            fig = self.resolved_galaxy.plot_bagpipes_map_gif(
                parameter=param_property,
                run_name=p_run_name,
                weight_mass_sfr=weight_mass_sfr,
                logmap=logmap,
                path="temp",
                facecolor=self.facecolor,
                cmap="magma",
            )
        elif outtype == "median":
            fig = self.resolved_galaxy.plot_bagpipes_results(
                facecolor=self.facecolor,
                parameters=[param_property],
                max_on_row=3,
                run_name=p_run_name,
                norm=norm_param,
                total_params=total_params,
            )

        if fig is not None:
            if isinstance(fig, plt.Figure):
                plt.close(fig)
                return pn.pane.Matplotlib(
                    fig,
                    dpi=144,
                    tight=True,
                    format="svg",
                    width=400,
                    height=400,
                )
            else:
                return pn.pane.GIF(fig, width=400, height=400)
        else:
            return pn.pane.Markdown("No Bagpipes results found.")

    @param.depends()
    def create_photometric_properties_tab(self):
        phot_prop_page = pn.Column(
            loading=True,
            sizing_mode="stretch_both",
            min_height=400,
            min_width=400,
        )

        options_phot = ["Beta", "MUV", "SFR_UV"]  # Add more options as needed
        drop_down1 = pn.widgets.Select(
            options=options_phot, value="Beta", width=100
        )
        cmap_1 = pn.widgets.Select(
            options=plt.colormaps(), value="viridis", width=100
        )

        plot1 = pn.param.ParamFunction(
            pn.bind(
                self.plot_phot_property,
                drop_down1.param.value,
                cmap_1.param.value,
                watch=False,
            ),
            loading_indicator=True,
        )

        phot_prop_page.append(
            pn.Column(
                pn.Row(drop_down1, cmap_1), plot1, sizing_mode="stretch_width"
            )
        )

        return phot_prop_page

    @param.depends()
    def create_interactive_tab(self):
        interactive_plot = self.create_interactive_plot()
        controls = pn.Column(
            pn.pane.Markdown(
                """
            ### How to use

            - Choose one of the tools on the right side of the plot

            - There are three tools available:
                - Box Edit: Draw rectangles
                - Poly Draw: Draw polygons
                - Point Draw: Draw points

            - For box edit and poly draw, double click to start and finish a shape.
            - For point draw, click to draw a point.
            - To remove a point or shape, click on it, then hit the 'Backspace' key.

            - When finished, click the 'Save' button to save the drawn shapes.
            - The photometry for the drawn shapes will be displayed below the plot.
            - If you want to perform SED fitting on the drawn shapes, configure the SED fitting options and click 'Run SED Fitting'.
            - SED fitting results will be displayed in the SED Results tab, and saved to the .h5 file.
            """,
                width=400,
            ),
        )

        # Link the button to the
        self.interactive_photometry_button.on_click(
            self.photometry_from_shapes
        )

        def clear_shapes(event):
            if event:
                self.drawn_shapes = {}
                self.interactive_poly = hv.Polygons([])
                self.interactive_boxes = hv.Rectangles([])
                self.interactive_points = hv.Points([])

        self.clear_interactive_button.on_click(clear_shapes)

        self.save_shape_photometry_button.on_click(self.photometry_from_shapes)

        self.interactive_sed_fitting_button.on_click(
            self.photometry_from_shapes
        )

        self.interactive_sed_plot = pn.pane.Matplotlib(
            plt.figure(),
            dpi=144,
            tight=True,
            format="svg",
            sizing_mode="stretch_width",
            max_height=400,
        )
        self.interactive_sed_plot_pdf = pn.pane.Matplotlib(
            plt.figure(),
            dpi=144,
            tight=True,
            format="svg",
            sizing_mode="stretch_width",
            max_height=400,
        )

        return pn.Row(
            pn.Column(interactive_plot, controls),
            pn.Column(
                self.interactive_sed_plot, self.interactive_sed_plot_pdf
            ),
        )

    @param.depends("interactive_band", "drawn_shapes")
    def create_interactive_plot(self):
        image_data = self.resolved_galaxy.phot_imgs[self.interactive_band]
        # Flip the image data so that it is displayed correctly
        image_data = np.flipud(image_data)

        image = hv.Image(
            image_data, bounds=(0, 0, image_data.shape[1], image_data.shape[0])
        )

        poly = self.interactive_poly
        boxes = self.interactive_boxes
        points = self.interactive_points

        # paths = hv.Path([])

        lower, upper = np.nanpercentile(image_data, [5, 99])
        if lower <= 0:
            lower = upper / 100

        image.opts(
            aspect="equal",
            xaxis=None,
            yaxis=None,
            width=500,
            height=500,
            cnorm="log",
            clim=(lower, upper),
        )

        poly_colors_cmap = cm.get_cmap("nipy_spectral")
        # draw 30 colors from the viridis colormap
        poly_colors = [poly_colors_cmap(i) for i in np.linspace(0, 1, 10)]

        poly_colors = [
            "#%02x%02x%02x" % tuple(int(255 * x) for x in color[:3])
            for color in poly_colors
        ]

        poly_stream = streams.PolyDraw(
            source=poly,
            drag=True,
            num_objects=10,
            show_vertices=False,
            styles={"fill_color": poly_colors},
        )

        box_colors_cmap = cm.get_cmap("tab20")
        # draw 30 colors from the tab20 colormap
        box_colors = [box_colors_cmap(i) for i in np.linspace(0, 1, 10)]

        box_colors = [
            "#%02x%02x%02x" % tuple(int(255 * x) for x in color[:3])
            for color in box_colors
        ]

        box_stream = streams.BoxEdit(
            source=boxes, num_objects=10, styles={"fill_color": box_colors}
        )

        points_colors_cmap = cm.get_cmap("hsv")
        # draw 30 colors from the inferno colormap
        points_colors = [points_colors_cmap(i) for i in np.linspace(0, 1, 10)]

        points_colors = [
            "#%02x%02x%02x" % tuple(int(255 * x) for x in color[:3])
            for color in points_colors
        ]

        points_stream = streams.PointDraw(
            source=points, num_objects=10, styles={"line_color": points_colors}
        )

        def on_draw_box(data):
            self.drawn_shapes["boxes"] = data

        def on_draw_poly(data):
            self.drawn_shapes["polygons"] = data

        def on_draw_points(data):
            self.drawn_shapes["points"] = data

        box_stream.add_subscriber(on_draw_box)
        poly_stream.add_subscriber(on_draw_poly)
        points_stream.add_subscriber(on_draw_points)

        plot = (image * poly * boxes * points).opts(
            opts.Image(cmap="gray"),
            opts.Rectangles(fill_alpha=0.3, active_tools=["box_edit"]),
            opts.Polygons(fill_alpha=0.3, active_tools=["poly_draw"]),
            opts.Points(
                active_tools=["point_draw"],
                size=45,
                fill_color="none",
                line_width=4,
            ),
            # opts.Path(active_tools=['freehand_draw'])
        )
        self.interactive_plot = pn.pane.HoloViews(plot, height=500, width=500)

        return self.interactive_plot

    @param.depends("which_interactive_sed_fitter")
    def interactive_sed_fitting_options(self):
        if self.which_interactive_sed_fitter == "EAZY-py":
            # options - min percentage error,
            # redshift range
            # template(s)

            # Make a accordion with the options
            # Each option should be a widget
            # The widget should be linked to the appropriate parameter

            # The accordion should be returned
            # The accordion should be added to the interactive tab

            # The accordion should be updated when the which_interactive_sed_fitter changes

            min_percent_error_widget = pn.widgets.FloatInput(
                name="Min Percent Error", value=10, step=1, start=0, end=100
            )
            min_percent_error_widget.link(self, value="min_percent_error")

            redshift_range_widget = pn.widgets.RangeSlider(
                name="Redshift Range", start=0, end=10, value=(0, 25), step=0.1
            )

            redshift_range_widget.link(self, value="redshift_range")

            template_widget = pn.widgets.Select(
                name="Template",
                options=[
                    "FSPS",
                    "FSPS+Larson",
                    "BC03",
                    "JADES",
                    "ARES",
                ],
                value="BC03",
            )

    @param.depends()
    def photometry_from_shapes(self, event, flux_unit=u.ABmag):
        # Label the drawn shapes on the bokeh plot

        if event:
            regions, region_labels = self.convert_shapes_to_regions()
            regions, region_labels = self.convert_shapes_to_regions()
            num_regions = len(regions)

            # check if same event and if regions are the same
            if (
                hasattr(self, "int_event")
                and self.int_event == event
                and hasattr(self, "int_regions")
                and regions == self.int_regions
            ):
                print("Regions are the same")
                return

            self.int_regions = regions
            self.int_event = event

            for reg in region_labels:
                # label = Label(x=reg['x'], y=reg['y'], text=reg['name'], text_color=reg['color'] if type(reg['color']) == str else 'black')

                text = hv.Text(
                    x=reg["x"],
                    y=reg["y"] - 1.4 * self.interactive_aperture_radius,
                    text=reg["name"],
                )  # color=reg['color'] if type(reg['color']) == str else 'black')
                # Draw the text on the plot without removing the drawn shapes
                # white text
                text.opts(color="white")
                self.interactive_plot.object = (
                    self.interactive_plot.object * text
                )

                # Add

            # Readd the drawn shapes to the plot

            for key in self.drawn_shapes:
                if key == "boxes":
                    color = self.drawn_shapes[key]["fill_color"]

                    color = [i if type(i) == str else "black" for i in color]

                    shape = hv.Rectangles(
                        self.drawn_shapes[key], vdims=["fill_color"]
                    ).opts(
                        alpha=0.3,
                        color="fill_color",
                    )
                    self.interactive_boxes = shape

                elif key == "polygons":
                    # color = self.drawn_shapes[key]['fill_color']
                    # color = [i if type(i) == str else 'black' for i in color]
                    # print(self.drawn_shapes[key])
                    xs = self.drawn_shapes[key]["xs"]
                    ys = self.drawn_shapes[key]["ys"]
                    colors = self.drawn_shapes[key]["fill_color"]
                    list_of_polys = []
                    for i in range(len(xs)):
                        poly = {
                            "x": xs[i],
                            "y": ys[i],
                            "fill_color": colors[i],
                        }
                        list_of_polys.append(poly)

                    shape = hv.Polygons(
                        list_of_polys, vdims=["fill_color"]
                    ).opts(
                        alpha=0.3,
                        color="fill_color",
                    )
                    self.interactive_poly = shape

                elif key == "points":
                    # Draw ellipses for the points
                    for j, (posx, posy, color) in enumerate(
                        zip(
                            self.drawn_shapes[key]["x"],
                            self.drawn_shapes[key]["y"],
                            self.drawn_shapes[key]["line_color"],
                        )
                    ):
                        if type(color) != str:
                            color = "black"

                        i = hv.Ellipse(
                            posx, posy, 2 * self.interactive_aperture_radius
                        ).opts(color=color, line_width=2)

                        if j == 0:
                            shape = i
                        else:
                            shape *= i

                    self.interactive_points = shape

                self.interactive_plot.object = (
                    self.interactive_plot.object * shape
                )

            if num_regions == 0:
                print("No regions drawn")
                return

            fluxes, flux_errs, flux_unit_arr = [], [], []
            debug_fig = plt.figure(figsize=(8, 5), constrained_layout=True)
            # Make two subfigures
            # One for the SED plot
            # One for the region cutout plot
            subfigs = debug_fig.subfigures(
                1, 2, wspace=0.07, width_ratios=[1.5, 2]
            )

            plot_ax = subfigs[0].add_subplot(111)
            plot_ax.set_xticks([])
            plot_ax.set_yticks([])
            plot_ax.set_frame_on(False)

            # Add two subaxis to the 2nd subfigure of the main figure
            # One above the other
            ax_save_sed = subfigs[1].add_subplot(211)
            ax_save_pdf = subfigs[1].add_subplot(212)

            ax_save_sed.set_xlabel("Wavelength (microns)", fontsize=12)
            ax_save_sed.set_ylabel("Flux (AB Mag)")
            ax_save_sed.set_title("SED Plot")
            # ax_save_pdf.set_title("PDF Plot")
            ax_save_pdf.set_xlabel("Redshift (z)", fontsize=12)

            newfig = plt.figure(figsize=(8, 6))
            ax = newfig.add_subplot(111)
            ax.set_xlabel("Wavelength (microns)")

            for pos, region in enumerate(regions):
                self.resolved_galaxy.plot_photometry_from_region(
                    region,
                    facecolor=self.facecolor,
                    ax=ax,
                    fig=newfig,
                    flux_unit=flux_unit,
                    label=region_labels[pos]["name"],
                )

                self.resolved_galaxy.plot_photometry_from_region(
                    region,
                    facecolor=self.facecolor,
                    ax=ax_save_sed,
                    fig=subfigs[1],
                    flux_unit=flux_unit,
                    label=region_labels[pos]["name"],
                )

            if flux_unit == u.ABmag:
                ax.set_ylabel("AB Mag")
                ax_save_sed.set_ylabel("AB Mag", fontsize=12)
                # Invert y axis

                ax.invert_yaxis()
                ax_save_sed.invert_yaxis()
            else:
                ax.set_ylabel(f"Flux ({flux_unit})")
                ax_save_sed.set_ylabel(f"Flux ({flux_unit})")

            # Fix x axis range
            ax.set_xlim(ax.get_xlim())
            ax.set_ylim(ax.get_ylim())
            ax_save_sed.set_xlim(ax.get_xlim())
            ax_save_sed.set_ylim(ax.get_ylim())

            if event.obj.name == "Measure Photometry":
                ax.legend(frameon=False)

            if (
                event.obj.name == "Run SED Fitting"
                or event.obj.name == "Save regions"
            ):
                for region in regions:
                    flux, flux_err = (
                        self.resolved_galaxy.get_photometry_from_region(
                            region,
                            return_array=True,
                            save_debug_plot=False,
                        )
                    )

                    self.resolved_galaxy.get_photometry_from_region(
                        region,
                        return_array=False,
                        save_debug_plot=False,
                        debug_fig=subfigs[0],
                    )

                    fluxes.append(flux)
                    flux_errs.append(flux_err)
                    flux_unit_arr.append(flux.unit)
                assert all(
                    [unit == flux_unit_arr[0] for unit in flux_unit_arr]
                ), "All fluxes must have the same units"

                fluxes = np.array(fluxes) * flux_unit_arr[0]
                flux_errs = np.array(flux_errs) * flux_unit_arr[0]

                if self.which_interactive_sed_fitter == "EAZY-py":
                    pdf_fig = plt.figure(figsize=(8, 6))
                    ax_pdf = pdf_fig.add_subplot(111)
                    ax_pdf.set_xlabel("Redshift (z)")
                    ax_pdf.set_ylabel("Probability Density")
                    # Hide the y-axis labels
                    ax_pdf.yaxis.set_tick_params(labelleft=False)
                    ax_save_pdf.yaxis.set_tick_params(labelleft=False)

                    ez = self.resolved_galaxy.fit_eazy_photometry(
                        run_name="app", fluxes=fluxes, flux_errs=flux_errs
                    )
                    datas = []
                    for i, region in enumerate(regions):
                        data = self.resolved_galaxy.plot_eazy_fit(
                            ez,
                            i,
                            ax_sed=ax,
                            fig=newfig,
                            flux_units=flux_unit,
                            ax_pz=ax_pdf,
                            label=True,
                            color=region_labels[i]["color"]
                            if type(region_labels[i]["color"]) != int
                            else "black",
                        )
                        self.resolved_galaxy.plot_eazy_fit(
                            ez,
                            i,
                            ax_sed=ax_save_sed,
                            fig=subfigs[1],
                            flux_units=flux_unit,
                            ax_pz=ax_save_pdf,
                            label=True,
                            color=region_labels[i]["color"]
                            if type(region_labels[i]["color"]) != int
                            else "black",
                        )

                        datas.append(data)

                    # Get all lines on the ax_pdf plot, sum them, and set the x-axis limit based on percentiles of the y-axis peaks
                    lines = ax_pdf.get_lines()
                    y_data = np.array([line.get_ydata() for line in lines])
                    y_data = np.sum(y_data, axis=0)

                    print("shap", np.shape(y_data))

                    norm = np.cumsum(y_data)
                    norm = norm / np.max(norm)
                    x_data = lines[0].get_xdata()
                    lowz = x_data[np.argmin(np.abs(norm - 0.02))] - 0.3
                    highz = x_data[np.argmin(np.abs(norm - 0.98))] + 0.3
                    ax_pdf.set_xlim(lowz, highz)

                    print("lowz", lowz, "highz", highz)

                    self.interactive_sed_plot_pdf.object = pdf_fig

                    ax_save_sed.legend(frameon=False)

                    ax.legend(frameon=False)

                    # debug_fig.savefig(f"{}"
                    try:
                        debug_fig.savefig(
                            f"/nvme/scratch/work/tharvey/EXPANSE/galaxies/diagnostic_plots/region_cutouts.png"
                        )
                    except FileNotFoundError:
                        pass

                elif self.which_interactive_sed_fitter == "Dense Basis":
                    raise NotImplementedError(
                        "Dense Basis SED fitting is not implemented yet."
                    )

            if event.obj.name == "Save regions":
                if self.which_interactive_sed_fitter == "EAZY-py":
                    self.resolved_galaxy.save_eazy_outputs(
                        ez, regions, fluxes, flux_errs, save_txt=True
                    )

            self.interactive_sed_plot.object = newfig

    @param.depends()
    def convert_regions_to_shapes(self, regions):
        from regions import (
            PixCoord,
            RectanglePixelRegion,
            PolygonPixelRegion,
            CirclePixelRegion,
        )

        # Convert regions to shapes
        shape_convert_dict = {
            RectanglePixelRegion: "boxes",
            PolygonPixelRegion: "polygons",
            CirclePixelRegion: "points",
        }

        shapes = {}
        for region in regions:
            if type(region) in shape_convert_dict:
                if shape_convert_dict[type(region)] not in shapes:
                    shapes[shape_convert_dict[type(region)]] = {
                        "x0": [],
                        "y0": [],
                        "x1": [],
                        "y1": [],
                        "xs": [],
                        "ys": [],
                    }

                if type(region) == RectanglePixelRegion:
                    shapes[shape_convert_dict[type(region)]]["x0"].append(
                        region.center.x - region.width / 2
                    )
                    shapes[shape_convert_dict[type(region)]]["y0"].append(
                        region.center.y - region.height / 2
                    )
                    shapes[shape_convert_dict[type(region)]]["x1"].append(
                        region.center.x + region.width / 2
                    )
                    shapes[shape_convert_dict[type(region)]]["y1"].append(
                        region.center.y + region.height / 2
                    )
                elif type(region) == PolygonPixelRegion:
                    shapes[shape_convert_dict[type(region)]]["xs"].append(
                        region.vertices.x
                    )
                    shapes[shape_convert_dict[type(region)]]["ys"].append(
                        region.vertices.y
                    )
                elif type(region) == CirclePixelRegion:
                    shapes[shape_convert_dict[type(region)]]["x"].append(
                        region.center.x
                    )
                    shapes[shape_convert_dict[type(region)]]["y"].append(
                        region.center.y
                    )

        return shapes

    @param.depends()
    def convert_shapes_to_regions(self):
        from regions import (
            PixCoord,
            RectanglePixelRegion,
            PolygonPixelRegion,
            CirclePixelRegion,
        )

        regions = []

        region_strings = []
        num_regions = 0
        region_labels = []
        # Convert boxes to RectanglePixelRegion
        if "boxes" in self.drawn_shapes:
            if (
                "color" in self.drawn_shapes["boxes"]
                or "fill_color" in self.drawn_shapes["boxes"]
            ):
                colors = self.drawn_shapes["boxes"].get(
                    "color", self.drawn_shapes["boxes"].get("fill_color")
                )

            for (
                x0,
                y0,
                x1,
                y1,
                color,
            ) in zip(
                self.drawn_shapes["boxes"]["x0"],
                self.drawn_shapes["boxes"]["y0"],
                self.drawn_shapes["boxes"]["x1"],
                self.drawn_shapes["boxes"]["y1"],
                colors,
            ):
                x = (x0 + x1) / 2
                y = (y0 + y1) / 2
                width = abs(x1 - x0)
                height = abs(y1 - y0)

                center = PixCoord(x, y)
                reg = RectanglePixelRegion(
                    center, width, height, visual={"color": color}
                )
                regions.append(reg)
                region_labels.append(
                    {
                        "name": f"Box {num_regions}",
                        "x": x,
                        "y": y,
                        "color": color,
                    }
                )
                num_regions += 1

        # Convert polygons to PolygonPixelRegion

        if "polygons" in self.drawn_shapes:
            if (
                "color" in self.drawn_shapes["polygons"]
                or "fill_color" in self.drawn_shapes["polygons"]
            ):
                colors = self.drawn_shapes["polygons"].get(
                    "color", self.drawn_shapes["polygons"].get("fill_color")
                )

            for poly_xs, poly_ys, color in zip(
                self.drawn_shapes["polygons"]["xs"],
                self.drawn_shapes["polygons"]["ys"],
                colors,
            ):
                if len(poly_xs) < 3:
                    continue
                vertices = PixCoord(poly_xs, poly_ys)
                reg = PolygonPixelRegion(vertices, visual={"color": color})
                regions.append(reg)
                x = np.mean(poly_xs)
                y = np.mean(poly_ys)
                region_labels.append(
                    {
                        "name": f"Polygon {num_regions}",
                        "x": x,
                        "y": y,
                        "color": color,
                    }
                )
                num_regions += 1

        # Convert points to CirclePixelRegion
        if "points" in self.drawn_shapes:
            if (
                "line_color" in self.drawn_shapes["points"]
                or "fill_color" in self.drawn_shapes["points"]
            ):
                colors = self.drawn_shapes["points"].get(
                    "line_color", self.drawn_shapes["points"].get("fill_color")
                )
            for point_x, point_y, color in zip(
                self.drawn_shapes["points"]["x"],
                self.drawn_shapes["points"]["y"],
                colors,
            ):
                center = PixCoord(point_x, point_y)
                # You might want to adjust the radius based on your needs
                radius = self.interactive_aperture_radius
                regions.append(
                    CirclePixelRegion(center, radius, visual={"color": color})
                )
                region_labels.append(
                    {
                        "name": f"Aperture {num_regions}",
                        "x": point_x,
                        "y": point_y,
                        "color": color,
                    }
                )
                num_regions += 1

        return regions, region_labels

    # @lru_cache(maxsize=None)

    @param.depends()
    def update_sidebar(self):
        sidebar = pn.Column(
            pn.layout.Divider(), "### Settings", name="settings_sidebar"
        )

        if self.active_tab == 0:  # Cutouts
            sidebar.extend(
                [
                    self.psf_mode_widget,
                    "#### RGB Image",
                    self.red_channel_widget,
                    self.green_channel_widget,
                    self.blue_channel_widget,
                    self.stretch_widget,
                    self.q_widget,
                ]
            )
        elif self.active_tab == 1:  # SED Results
            sidebar.extend(
                [
                    "#### Pixel Binning",
                    self.which_map_widget,
                    pn.Row(self.show_galaxy_widget, self.show_kron_widget),
                    self.scale_alpha_widget,
                    self.choose_show_band_widget,
                    "#### SED Fitting Tool",
                    self.which_sed_fitter_widget,
                    pn.bind(
                        self.possible_runs_select,
                        self.which_sed_fitter_widget.param.value,
                        watch=True,
                    ),
                    pn.Row(
                        self.upscale_select_widget,
                        self.show_sed_photometry_widget,
                    ),
                    "#### SED Plot Config",
                    self.which_flux_unit_widget,
                    self.multi_choice_bins_widget,
                    self.total_fit_options_widget,
                    self.sed_plot_controls,
                ]
            )
        elif self.active_tab == 2:  # Photometric Properties
            sidebar.extend(
                [
                    "#### Pixel Binning",
                    self.which_map_widget,
                ]
            )

        if self.active_tab == 3:  # Interactive tab
            sidebar.extend(
                [
                    "#### Interactive Plot Controls",
                    self.interactive_band_widget,
                    self.interactive_aperture_radius_widget,
                    self.interactive_photometry_button,
                    self.interactive_sed_fitting_dropdown,
                    self.interactive_sed_fitting_button,
                    self.save_shape_photometry_button,
                    self.clear_interactive_button,
                ]
            )

            # Add Synthesizer tab options if it's a MockResolvedGalaxy
        if isinstance(self.resolved_galaxy, MockResolvedGalaxy):
            if (
                self.active_tab == 3
            ):  # Assuming Synthesizer is the 4th tab (index 3)
                sidebar.append(pn.layout.Divider())
                sidebar.append("### Synthesizer Options")
                # Add any specific Synthesizer options here

        return sidebar

    @param.depends(
        # "resolved_galaxy",
        "red_channel",
        "green_channel",
        "blue_channel",
        "stretch",
        "q",
        "psf_mode",
    )
    @check_dependencies()
    def plot_rgb(self):
        """
        Plot an RGB image of the galaxy using specified bands and parameters.

        Args:
            red (list): List of bands for the red channel.
            green (list): List of bands for the green channel.
            blue (list): List of bands for the blue channel.
            scale (float): Stretch parameter for Lupton RGB.
            q (float): Q parameter for Lupton RGB.
            psf_mode (str): 'Original' or 'PSF Matched'.

        Returns:
            pn.pane.HoloViews: A HoloViews pane containing the RGB image.
        """
        use_psf_matched = self.psf_mode == "PSF Matched"
        rgb = self.resolved_galaxy.plot_lupton_rgb(
            self.red_channel,
            self.green_channel,
            self.blue_channel,
            self.stretch,
            self.q,
            use_psf_matched=use_psf_matched,
        )

        rgb = np.flipud(rgb)  # Flip y axis to match image orientation

        rgb_img = hv.RGB(
            rgb,
            bounds=(
                0,
                0,
                self.resolved_galaxy.cutout_size,
                self.resolved_galaxy.cutout_size,
            ),
        ).opts(xaxis=None, yaxis=None)

        any_band = self.resolved_galaxy.bands[0]
        scale_bar_label = hv.Text(17.5, 10, "1'", fontsize=15).opts(
            color="white"
        )
        scale_bar_size = (
            self.resolved_galaxy.im_pixel_scales[any_band].to(u.arcsec).value
        )
        scale_bar_10as = hv.Rectangles(
            [(3, 3, 3 + 1 / scale_bar_size, 4)]
        ).opts(color="white", line_color="white")

        rgb_img = rgb_img * scale_bar_label * scale_bar_10as

        if use_psf_matched:
            circle = hv.Ellipse(7, 15, 5).opts(
                color="white", line_color="white"
            )
            rgb_img = rgb_img * circle

        center = self.resolved_galaxy.cutout_size / 2
        a, b, theta = self.resolved_galaxy.plot_kron_ellipse(
            None, None, return_params=True
        )
        kron = hv.Ellipse(
            center, center, (a, b), orientation=theta * np.pi / 180
        ).opts(color="white", line_color="white")
        rgb_img = rgb_img * kron

        return pn.pane.HoloViews(rgb_img, height=400, width=430)

    # @lru_cache(maxsize=None)

    @param.depends(
        "which_map",
        "show_galaxy",
        "show_kron",
        "scale_alpha",
        "choose_show_band",
    )
    @check_dependencies()
    def plot_bins(
        self,
        cmap="nipy_spectral_r",
    ):
        """
        Plot the binning map for the galaxy.

        Args:
            bin_type (str): Type of binning ('pixedfit' or 'voronoi').
            cmap (str): Colormap to use for the plot.
            scale_alpha (float): Alpha value for scaling.
            show_galaxy (bool): Whether to show the galaxy image.
            show_kron (bool): Whether to show the Kron ellipse.
            psf_matched (bool): Whether to use PSF-matched data.
            band (str): Band to use for showing the galaxy image.

        Returns:
            pn.pane.HoloViews: A HoloViews pane containing the binning map.
        """

        print("bins triggered")

        bin_type = self.which_map
        scale_alpha = self.scale_alpha
        show_galaxy = self.show_galaxy
        show_kron = self.show_kron
        psf_matched = self.psf_mode
        band = self.choose_show_band

        array = getattr(self.resolved_galaxy, f"{bin_type}_map")
        if array is None:
            return None

        array[array == 0] = np.nan
        dimensions = np.linspace(
            0,
            self.resolved_galaxy.cutout_size,
            self.resolved_galaxy.cutout_size,
        )
        im_array = xr.DataArray(
            array,
            dims=["y", "x"],
            name=f"{bin_type} bins",
            coords={"x": dimensions, "y": dimensions},
        )
        hvplot = im_array.hvplot("x", "y").opts(
            cmap=cmap,
            xaxis=None,
            yaxis=None,
            clabel=f"{bin_type} Bin",
            alpha=scale_alpha,
            shared_axes=False,
            # tools=['tap'],
        )

        opts = np.unique(array)
        # Make ints
        opts = opts[~np.isnan(opts)].astype(int)
        # Update options for the multi-choice widget
        if bin_type == "pixedfit":
            new_opts = ["RESOLVED"] + list(opts)
            update = any(
                [
                    i not in self.multi_choice_bins_widget.options
                    for i in new_opts
                ]
            )
            if update:
                print("Updating options")
                print(new_opts, self.multi_choice_bins_widget.options)
                self.multi_choice_bins_widget.options = new_opts

        if show_kron:
            center = self.resolved_galaxy.cutout_size / 2
            a, b, theta = self.resolved_galaxy.plot_kron_ellipse(
                None, None, return_params=True
            )
            kron = hv.Ellipse(
                center, center, (a, b), orientation=theta * np.pi / 180
            ).opts(color="red", line_color="red")
            hvplot = hvplot * kron

        if show_galaxy:
            if psf_matched:
                data = self.resolved_galaxy.psf_matched_data["star_stack"][
                    band
                ]
            else:
                data = self.resolved_galaxy.phot_imgs[band]
            data = np.flipud(data)
            lower, upper = np.nanpercentile(data, [5, 99])
            if lower <= 0:
                lower = upper / 1000

            new_hvplot = hv.Image(
                data,
                bounds=(
                    0,
                    0,
                    self.resolved_galaxy.cutout_size,
                    self.resolved_galaxy.cutout_size,
                ),
            ).opts(
                xaxis=None,
                yaxis=None,
                cmap="gray",
                cnorm="log",
                clim=(lower, upper),
                shared_axes=False,
            )
            hvplot = new_hvplot * hvplot

        bin_map = pn.pane.HoloViews(
            hvplot, height=300, width=400, aspect_ratio=1
        )
        self.point_selector.source = bin_map.object

        # Update the point selector stream
        # self.point_selector.source = hvplot

        return bin_map

    # @lru_cache(maxsize=None)
    @param.depends("psf_mode")
    @check_dependencies()
    def plot_cutouts(self, psf_type="star_stack"):
        """
        Plot cutouts of the galaxy in different bands.

        Args:
            psf_matched_param (str): 'Original' or 'PSF Matched'.
            psf_type_param (str): Type of PSF to use.

        Returns:
            pn.pane.Matplotlib: A Matplotlib pane containing the cutout plots.
        """

        fig = self.resolved_galaxy.plot_cutouts(
            facecolor=self.facecolor,
            psf_matched=self.psf_mode == "PSF Matched",
            psf_type=psf_type,
        )
        plt.close(fig)
        return pn.pane.Matplotlib(
            fig,
            dpi=144,
            max_width=950,
            tight=True,
            format="svg",
            sizing_mode="scale_width",
        )

    def possible_runs_select(self, sed_fitting_tool):
        """
        Update the available SED fitting runs based on the selected tool.

        Args:
            sed_fitting_tool (str): The selected SED fitting tool.

        Returns:
            pn.Column or None: A column with run selection widgets, or None if no runs are found.
        """
        if sed_fitting_tool is None:
            return None

        options = self.resolved_galaxy.sed_fitting_table.get(
            sed_fitting_tool, None
        )
        options_resolved = []
        options_aperture = []
        if options is None:
            options = ["No runs found"]
        else:
            for option in options.keys():
                ids = options[option]["#ID"]
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

        self.which_run_resolved_widget.options = options_resolved
        self.which_run_aperture_widget.options = options_aperture
        if len(options_resolved) > 0:
            self.which_run_resolved = options_resolved[0]
        if len(options_aperture) > 0:
            self.which_run_aperture = options_aperture[0]

        return pn.Column(
            self.which_run_resolved_widget, self.which_run_aperture_widget
        )

    @param.depends(
        # "resolved_galaxy",
        "which_map",
        "which_sed_fitter",
        "which_run_resolved",
        "which_run_aperture",
        "multi_choice_bins",
        "total_fit_options",
        "pdf_param_property",
    )
    @check_dependencies()
    def plot_bagpipes_pdf(self, cmap="nipy_spectral_r"):
        if self.which_sed_fitter == "bagpipes":
            multi_choice_bins_param_safe = [
                int(i) if i != "RESOLVED" and i != np.nan else i
                for i in self.multi_choice_bins
            ]

            if (
                self.which_run_resolved
                and self.which_run_resolved
                in self.resolved_galaxy.sed_fitting_table["bagpipes"].keys()
            ):
                if self.which_map == "pixedfit":
                    map = self.resolved_galaxy.pixedfit_map

                norm = Normalize(vmin=np.nanmin(map), vmax=np.nanmax(map))

                cmap = cm.get_cmap(cmap)
                colors_bins = [
                    cmap(norm(rbin)) if rbin != "RESOLVED" else "black"
                    for pos, rbin in enumerate(multi_choice_bins_param_safe)
                ]
                fig, _ = (
                    self.resolved_galaxy.plot_bagpipes_component_comparison(
                        parameter=self.pdf_param_property,
                        run_name=self.which_run_resolved,
                        bins_to_show=multi_choice_bins_param_safe,
                        save=False,
                        run_dir="pipes/",
                        facecolor=self.facecolor,
                        colors=colors_bins,
                    )
                )

                ax = fig.get_axes()[0]
            else:
                fig = None
                ax = None

            if (
                self.which_run_aperture
                and self.which_run_aperture
                in self.resolved_galaxy.sed_fitting_table["bagpipes"].keys()
            ):
                colors_total = [
                    self.TOTAL_FIT_COLORS[rbin]
                    for rbin in self.total_fit_options
                ]

                fig, _ = (
                    self.resolved_galaxy.plot_bagpipes_component_comparison(
                        parameter=self.pdf_param_property,
                        run_name=self.which_run_aperture,
                        bins_to_show=self.total_fit_options,
                        save=False,
                        run_dir="pipes/",
                        facecolor=self.facecolor,
                        fig=fig,
                        axes=ax,
                        colors=colors_total,
                    )
                )

            plt.close(fig)
            return pn.pane.Matplotlib(
                fig,
                dpi=144,
                tight=True,
                format="svg",
                sizing_mode="scale_both",
                min_width=500,
                min_height=400,
            )
        else:
            return pn.pane.Markdown(
                f"Not implemented for {self.which_sed_fitter}."
            )

    @param.depends(
        # "resolved_galaxy",
        "which_map",
        "which_sed_fitter",
        "which_flux_unit",
        "multi_choice_bins",
        "which_run_aperture",
        "which_run_resolved",
        "total_fit_options",
        "show_sed_photometry",
        "sed_log_x",
        "sed_log_y",
        "sed_x_min",
        "sed_x_max",
        "sed_y_min",
        "sed_y_max",
    )
    @check_dependencies()
    def plot_sed(self):
        """
        import inspect

        current_function =  inspect.stack()[0][3]


        print('SED triggered')
        # print all the parameters
        # Get list of triggers
        dependencies = [o.name for o in self.param.method_dependencies(current_function)]

        print(f'Current function: {current_function}, Dependencies: {dependencies}')

        args = [copy.deepcopy(getattr(self, dep)) for dep in dependencies]


        if getattr(self, f'{current_function}_previous_args', None) == None:
            setattr(self, f'{current_function}_previous_args', args)
            print('First time running')
        else:
            # Check if any of the arguments have changed
            if any([args[i] != getattr(self, f'{current_function}_previous_args')[i] for i in range(len(args))]):
                print('Arguments have changed')
                setattr(self, f'{current_function}_previous_args', args)
            else:
                print('Arguments have not changed')
                return self.sed_plot
        """

        multi_choice_bins_param_safe = [
            int(i) if i != "RESOLVED" and i != np.nan else i
            for i in self.multi_choice_bins
        ]

        fig, ax = plt.subplots(
            figsize=(6, 3), constrained_layout=True, facecolor=self.facecolor
        )

        x_unit = u.micron
        y_unit = {
            "uJy": u.uJy,
            "ABmag": u.ABmag,
            "ergscma": u.erg / u.s / u.cm**2 / u.AA,
        }[self.which_flux_unit]

        psf_type = self.resolved_galaxy.use_psf_type
        table = self.resolved_galaxy.photometry_table[psf_type][self.which_map]

        self.resolved_galaxy.get_filter_wavs()
        wavs = self.resolved_galaxy.filter_wavs

        show_sed_photometry = self.show_sed_photometry == "Show"

        list_of_markers = [
            "o",
            "s",
            "v",
            "^",
            "<",
            ">",
            "1",
            "2",
            "3",
            "4",
            "8",
            "p",
            "P",
            "*",
            "h",
            "H",
            "+",
            "x",
            "X",
            "D",
            "d",
            "|",
            "_",
        ]
        if show_sed_photometry:
            list_of_markers = list_of_markers[1:]

        colors = []
        colors_bin = []
        colors_total = []
        cmap = plt.get_cmap("nipy_spectral_r")

        for bin_pos, rbin in enumerate(
            multi_choice_bins_param_safe + self.total_fit_options
        ):
            if isinstance(rbin, str) and rbin != "RESOLVED":
                color = self.TOTAL_FIT_COLORS[rbin]
                colors_total.append(color)
            else:
                if rbin == "RESOLVED":
                    color = self.TOTAL_FIT_COLORS[rbin]
                else:
                    color = cmap(
                        plt.Normalize(
                            vmin=np.nanmin(self.resolved_galaxy.pixedfit_map),
                            vmax=np.nanmax(self.resolved_galaxy.pixedfit_map),
                        )(rbin)
                    )
                colors_bin.append(color)
            colors.append(color)

            mask = np.array([str(i) == str(rbin) for i in table["ID"]])
            table_row = table[mask]

            if len(table_row) == 0:
                continue

            marker = list_of_markers[bin_pos % len(list_of_markers)]

            max_flux = 0 * u.uJy
            for pos, band in enumerate(self.resolved_galaxy.bands):
                wav = wavs[band]
                flux = table_row[band]
                flux_err = table_row[f"{band}_err"]

                if not flux.isscalar:
                    flux = flux[0]
                    flux_err = flux_err[0]

                if flux > 0:
                    if flux_err / flux < 0.1:
                        flux_err = 0.1 * flux

                if y_unit == u.ABmag:
                    yerr = [
                        [
                            np.abs(
                                2.5
                                * abs(
                                    np.log10(
                                        flux.value
                                        / (flux.value - flux_err.value)
                                    )
                                )
                            )
                        ],
                        [
                            np.abs(
                                2.5 * np.log10(1 + flux_err.value / flux.value)
                            )
                        ],
                    ]
                    yerr[0][0], yerr[1][0] = yerr[1][0], yerr[0][0]
                else:
                    yerr = flux_err.to(
                        y_unit, equivalencies=u.spectral_density(wav)
                    ).value

                lab = int(rbin) if isinstance(rbin, float) else rbin
                lab = lab if pos == 0 else ""

                if flux > max_flux.to(
                    y_unit, equivalencies=u.spectral_density(wav)
                ):
                    max_flux = flux.to(
                        y_unit, equivalencies=u.spectral_density(wav)
                    )

                ax.errorbar(
                    wav.to(x_unit).value,
                    flux.to(
                        y_unit, equivalencies=u.spectral_density(wav)
                    ).value,
                    yerr=yerr,
                    fmt=marker,
                    linestyle="none",
                    color=color,
                    label=lab,
                    markeredgecolor="black" if show_sed_photometry else color,
                )

        if y_unit != u.ABmag:
            ax.set_ylim(None, max_flux.value * 1.5)

        if self.which_sed_fitter == "bagpipes":
            if self.which_run_resolved:
                fig, _ = self.resolved_galaxy.plot_bagpipes_fit(
                    self.which_run_resolved,
                    ax,
                    fig,
                    bins_to_show=multi_choice_bins_param_safe,
                    marker_colors=colors_bin,
                    wav_units=x_unit,
                    flux_units=y_unit,
                    show_photometry=show_sed_photometry,
                )

            if self.which_run_aperture:
                fig, _ = self.resolved_galaxy.plot_bagpipes_fit(
                    self.which_run_aperture,
                    ax,
                    fig,
                    bins_to_show=self.total_fit_options,
                    marker_colors=colors_total,
                    wav_units=x_unit,
                    flux_units=y_unit,
                    show_photometry=show_sed_photometry,
                )

        if y_unit == u.ABmag:
            ax.invert_yaxis()
        ax.set_xlabel(
            rf"$\rm{{Wavelength}}$ ({x_unit:latex})", fontsize="large"
        )
        ax.set_ylabel(
            rf"$\rm{{Flux \ Density}}$ ({y_unit:latex})", fontsize="large"
        )

        # if len(ax.lines) > 0:
        #    ax.set_xlim(ax.get_xlim())
        #    ax.set_ylim(ax.get_ylim())

        if self.sed_log_x:
            ax.set_xscale("log")
        if self.sed_log_y:
            ax.set_yscale("log")

        ax.set_xlim(self.sed_x_min, self.sed_x_max)

        if self.sed_y_min is not None or self.sed_y_max is not None:
            ax.set_ylim(self.sed_y_min, self.sed_y_max)

        ax.legend(loc="upper left", frameon=False, fontsize="small")

        self.sed_plot = pn.pane.Matplotlib(
            fig, dpi=144, tight=True, format="svg", sizing_mode="scale_both"
        )

        # self.key_sed_plot_params = {'x_unit': copy.copy(x_unit), 'y_unit': copy.copy(y_unit), 'show_sed_photometry': copy.copy(show_sed_photometry)},

        # plt.close(fig)
        return self.sed_plot

    @param.depends(
        # "resolved_galaxy",
        "which_map",
        "which_sed_fitter",
        "multi_choice_bins",
        "which_run_aperture",
        "which_run_resolved",
        "total_fit_options",
    )
    def plot_corner(self):
        """
        Plot the corner plot for the selected bins and runs.

        Returns:
            pn.pane.Matplotlib: A Matplotlib pane containing the corner plot.
        """
        multi_choice_bins_param_safe = [
            int(i) if i != "RESOLVED" and i != np.nan else i
            for i in self.multi_choice_bins
        ]

        if "RESOLVED" in multi_choice_bins_param_safe:
            multi_choice_bins_param_safe.remove("RESOLVED")

        colors_bins = [
            plt.get_cmap("nipy_spectral_r")(
                plt.Normalize(
                    vmin=np.nanmin(self.resolved_galaxy.pixedfit_map),
                    vmax=np.nanmax(self.resolved_galaxy.pixedfit_map),
                )(rbin)
            )
            for rbin in multi_choice_bins_param_safe
        ]
        colors_total = [
            self.TOTAL_FIT_COLORS[rbin] for rbin in self.total_fit_options
        ]

        if self.which_sed_fitter == "bagpipes":
            fig = None

            if self.which_run_aperture:
                bins_to_show = self.total_fit_options
                fig, _ = self.resolved_galaxy.plot_bagpipes_corner(
                    run_name=self.which_run_aperture,
                    bins_to_show=bins_to_show,
                    save=False,
                    facecolor=self.facecolor,
                    colors=colors_total,
                    run_dir="pipes/",
                )
            """

            if self.which_run_resolved:
                bins_to_show = multi_choice_bins_param_safe
                fig, _ = self.resolved_galaxy.plot_bagpipes_corner(
                    run_name=self.which_run_resolved,
                    bins_to_show=bins_to_show,
                    save=False,
                    facecolor=self.facecolor,
                    colors=colors_bins,
                    run_dir="pipes/",
                    fig=fig,
                )
            """

            if fig is None:
                return pn.pane.Markdown(
                    "No Bagpipes results found for corner plot."
                )

            plt.close(fig)
            return pn.pane.Matplotlib(
                fig,
                dpi=144,
                tight=True,
                format="svg",
                sizing_mode="stretch_both",
                max_width=800,
                min_width=500,
            )
        else:
            return pn.pane.Markdown(
                f"Corner plot not implemented for {self.which_sed_fitter}."
            )

    @param.depends(
        "which_map",
        "which_sed_fitter",
        "multi_choice_bins",
        "which_run_aperture",
        "which_run_resolved",
        "total_fit_options",
    )
    def plot_sfh(self):
        """
        Plot the star formation history (SFH) for the selected bins and runs.

        Returns:
            pn.pane.Matplotlib: A Matplotlib pane containing the SFH plot.
        """
        multi_choice_bins_param_safe = [
            int(i) if i != "RESOLVED" and i != np.nan else i
            for i in self.multi_choice_bins
        ]

        if self.which_sed_fitter == "bagpipes":
            cmap = plt.get_cmap("nipy_spectral_r")
            norm = plt.Normalize(
                vmin=np.nanmin(self.resolved_galaxy.pixedfit_map),
                vmax=np.nanmax(self.resolved_galaxy.pixedfit_map),
            )

            colors_bins = [
                cmap(norm(rbin)) if rbin != "RESOLVED" else "black"
                for rbin in multi_choice_bins_param_safe
            ]
            colors_total = [
                self.TOTAL_FIT_COLORS[rbin] for rbin in self.total_fit_options
            ]

            fig, ax = plt.subplots(
                figsize=(6, 3),
                constrained_layout=True,
                facecolor=self.facecolor,
            )

            if self.which_run_resolved:
                fig, _ = self.resolved_galaxy.plot_bagpipes_sfh(
                    run_name=self.which_run_resolved,
                    bins_to_show=multi_choice_bins_param_safe,
                    save=False,
                    facecolor=self.facecolor,
                    marker_colors=colors_bins,
                    time_unit="Gyr",
                    run_dir="pipes/",
                    fig=fig,
                    axes=ax,
                )

            if self.which_run_aperture:
                fig, _ = self.resolved_galaxy.plot_bagpipes_sfh(
                    run_name=self.which_run_aperture,
                    bins_to_show=self.total_fit_options,
                    save=False,
                    facecolor=self.facecolor,
                    marker_colors=colors_total,
                    time_unit="Gyr",
                    run_dir="pipes/",
                    fig=fig,
                    axes=ax,
                )

            # Check if any data was plotted
            if len(ax.lines) == 0:
                plt.close(fig)
                return pn.pane.Markdown(
                    "No Bagpipes results found for SFH plot."
                )

            ax.set_xlabel("Lookback Time (Gyr)", fontsize="large")
            ax.set_ylabel(r"SFR (M$_{\odot}$ yr$^{-1}$)", fontsize="large")
            # set tick label size
            ax.tick_params(axis="both", which="major", labelsize="medium")
            ax.set_yscale("log")
            ax.set_ylim(0.1, None)
            ax.legend(loc="upper right", fontsize="medium")

            plt.close(fig)
            return pn.pane.Matplotlib(
                fig,
                dpi=144,
                tight=True,
                format="svg",
                sizing_mode="scale_both",
            )
        else:
            return pn.pane.Markdown(
                f"SFH plotting not implemented for {self.which_sed_fitter}."
            )

    @param.depends()
    def create_synthesizer_tab(self):
        synthesizer_page = pn.Column()

        synthesizer_page.append("### Synthesizer Properties")

        top_row = pn.Row()
        middle_row = pn.Row()

        mass_map = self.plot_map_with_controls(
            self.resolved_galaxy.property_images["stellar_mass"],
            label="Stellar Mass",
            unit=r"$$M_{\odot}$$",
        )

        age_map = self.plot_map_with_controls(
            self.resolved_galaxy.property_images["stellar_age"],
            label="Age",
            unit="Gyr",
        )

        metallicity_map = self.plot_map_with_controls(
            self.resolved_galaxy.property_images["stellar_metallicity"],
            label="Metallicity",
            unit=r"$$Z_{\odot}$$",
        )

        top_row.append(mass_map)
        middle_row.append(age_map)
        middle_row.append(metallicity_map)

        synthesizer_page.append(top_row)
        synthesizer_page.append(middle_row)

        synthesizer_page.append(pn.layout.Divider())

        synthesizer_page.append("### Spectra")
        bottom_row = pn.Row()

        fig = pn.pane.Matplotlib(
            self.plot_mock_spectra(components=["det_segmap_fnu"]),
            dpi=144,
            tight=True,
            format="svg",
            sizing_mode="stretch_both",
            max_height=500,
        )

        bottom_row.append(fig)
        synthesizer_page.append(bottom_row)

        return synthesizer_page

    @param.depends()
    def plot_map_with_controls(self, map_array, label="", unit=""):
        range_slider = pn.widgets.RangeSlider(
            start=np.nanmin(map_array),
            end=np.nanmax(map_array),
            value=(np.nanmin(map_array), np.nanmax(map_array)),
            step=0.01 * (np.nanmax(map_array) - np.nanmin(map_array)),
            name="Colorbar Range",
            format=PrintfTickFormatter(format="%1.1e"),
        )
        colormap_choice = pn.widgets.Select(
            options=plt.colormaps(), value="viridis", name="Colormap"
        )
        log_scale = pn.widgets.Checkbox(name="Log Scale", value=True)

        @pn.depends(
            range_slider.param.value,
            colormap_choice.param.value,
            log_scale.param.value,
        )
        def create_plot(range_value, colormap, use_log_scale):
            if use_log_scale:
                clim = (np.log10(range_value[0]), np.log10(range_value[1]))
                data = np.log10(map_array)
            else:
                clim = range_value
                data = map_array

            if clim[0] == clim[1]:
                clim = (clim[0] - 1, clim[1] + 1)

            logtext = r"$$\log_{10}$$ " if log_scale else ""

            return hv.Image(
                data,
                bounds=(
                    0,
                    0,
                    self.resolved_galaxy.cutout_size,
                    self.resolved_galaxy.cutout_size,
                ),
            ).opts(
                cmap=colormap,
                colorbar=True,
                clabel=f"{label} ({logtext}{unit})",
                xaxis=None,
                yaxis=None,
                colorbar_position="top",
                clim=clim,
                logz=use_log_scale,
            )

        accordion = pn.Accordion(
            ("Options", pn.Column(range_slider, colormap_choice, log_scale)),
            width=300,
        )
        plot_pane = pn.pane.HoloViews(create_plot)

        return pn.Column(plot_pane, accordion)

    @param.depends()
    def plot_mock_spectra(self, components):
        fig, ax = plt.subplots(figsize=(8, 4.5), facecolor="white")

        if isinstance(components, str):
            components = [components]

        wav = self.resolved_galaxy.seds["wav"] * u.Angstrom

        for component in components:
            flux = self.resolved_galaxy.seds[component]["total"] * u.uJy
            ax.plot(
                wav.to(u.um),
                flux.to(u.uJy, equivalencies=u.spectral_density(wav)),
                label=component,
            )

        ax.set_xlim(0.5, 5)
        ax.set_ylim(
            0,
            1.1
            * np.max(flux[(wav > 0.5 * u.um) & (wav < 5 * u.um)])
            .to(u.uJy)
            .value,
        )
        ax.set_xlabel("Wavelength (μm)")
        ax.set_ylabel("Flux (μJy)")
        ax.legend()

        return fig


class ResolvedSEDApp(param.Parameterized):
    galaxy_tabs = param.List()
    galaxies_dir = galaxies_dir
    active_galaxy_tab = param.Integer(0)

    def __init__(self, **params):
        super().__init__(**params)
        self.sidebar = self.create_sidebar()
        self.tabs = pn.Tabs(
            closable=True, dynamic=True, scroll=False, min_height=2000
        )
        self.tabs.param.watch(self.update_active_galaxy_tab, "active")

    def create_sidebar(self):
        self.file_input = pn.widgets.FileInput(accept=".h5")
        self.choose_file_input = pn.widgets.Select(
            name="Select Remote File",
            options=[None],
            value=None,
        )

        if os.path.exists(galaxies_dir):
            self.choose_file_input.options = [None] + sorted(
                [f for f in os.listdir(self.galaxies_dir) if f.endswith(".h5")]
            )
        else:
            self.choose_file_input.options = ["No .h5 files found"]

        return pn.Column(
            "### Upload .h5", self.file_input, "### or", self.choose_file_input
        )

    def handle_file_upload(self, value):
        # Process uploaded file and create new GalaxyTab
        file = BytesIO(value)
        hfile = h5.File(file, "r")

        # ... Process file and create ResolvedGalaxy or MockResolvedGalaxy ...
        file = BytesIO(value)

        hfile = h5.File(file, "r")
        # what is the filename
        mtype = "mock" if "mock_galaxy" in hfile.keys() else "resolved"
        hfile.close()

        if mtype == "mock":
            resolved_galaxy = MockResolvedGalaxy.init_from_h5(file)
        else:
            resolved_galaxy = ResolvedGalaxy.init_from_h5(file)

        galaxy_tab = GalaxyTab(resolved_galaxy)

        galaxy_tab.app = self

        self.galaxy_tabs.append(galaxy_tab)
        self.tabs.append(
            (
                f"{galaxy_tab.galaxy_id} ({galaxy_tab.survey})",
                galaxy_tab.info_tabs,
            )
        )
        self.update_sidebar()

    def choose_file(self, value):
        # Load file from galaxies_dir and call handle_file_upload
        path = f"{galaxies_dir}/{value}"
        if os.path.exists(path):
            with open(path, "rb") as f:
                value = f.read()
            self.handle_file_upload(value)
        else:
            raise FileNotFoundError(f"File {path} not found.")

    def get_panel(self):
        return pn.template.FastListTemplate(
            title="EXPANSE - Resolved SED Viewer",
            sidebar=[self.sidebar],
            main=[self.tabs],
            accent=ACCENT,
        )

    def update_active_galaxy_tab(self, event):
        self.update_sidebar()
        self.active_galaxy_tab = event.new

    def update_sidebar(self):
        if self.galaxy_tabs:
            active_galaxy = self.galaxy_tabs[self.active_galaxy_tab]
            sidebar_content = active_galaxy.update_sidebar()
            self.sidebar.clear()
            self.sidebar.extend(
                [
                    "### Upload .h5",
                    self.file_input,
                    "### or",
                    self.choose_file_input,
                    pn.layout.Divider(),
                    sidebar_content,
                ]
            )


# Main function to create and serve the app
def resolved_sed_interface():
    app = ResolvedSEDApp()

    pn.bind(app.handle_file_upload, app.file_input, watch=True)
    pn.bind(app.choose_file, app.choose_file_input, watch=True)

    if len(initial_galaxy) > 0:
        app.choose_file_input.value = initial_galaxy

    return app.get_panel()


# CLI command remains largely unchanged
@click.command()
@click.option("--port", default=5006, help="Port to run the server on.")
@click.option(
    "--gal_dir", default="internal", help="Directory containing galaxy files."
)
@click.option("--galaxy", default="", help="List of galaxies to load.")
@click.option("--tab", default="Cutouts", help="Tab to load.")
@click.option("--test_mode", default=False, help="Run in test mode.")
def expanse_viewer_class(
    port=5004, gal_dir="internal", galaxy=None, tab="Cutouts", test_mode=False
):
    global initial_galaxy
    global galaxies_dir
    global initial_tab

    hv.extension("bokeh", logo=False)
    pn.extension("mathjax")
    pn.extension("tabulator")

    initial_galaxy = copy.copy(galaxy)
    initial_tab = copy.copy(tab)

    if test_mode:
        for key in logging.Logger.manager.loggerDict:
            logging.getLogger(key).setLevel(logging.ERROR)
            logging.getLogger(key).propagate = True

    if gal_dir != "internal":
        galaxies_dir = gal_dir

    pn.serve(
        resolved_sed_interface,
        websocket_max_message_size=MAX_SIZE_MB * 1024 * 1024,
        http_server_kwargs={"max_buffer_size": MAX_SIZE_MB * 1024 * 1024},
        port=port,
    )


if __name__ == "__main__":
    expanse_viewer_class()
