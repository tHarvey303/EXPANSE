import panel as pn
import param
import holoviews as hv
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
import h5py as h5
from astropy.io import fits
from astropy.wcs import WCS
import click
import copy
from functools import lru_cache
import os
from .ResolvedGalaxy import ResolvedGalaxy, MockResolvedGalaxy
from astropy import units as u
import xarray as xr


MAX_SIZE_MB = 150
ACCENT = "goldenrod"
galaxies_dir = os.path.join(
    os.path.dirname(
        os.path.dirname((os.path.dirname(os.path.abspath(__file__))))
    ),
    "galaxies",
)


class GalaxyTab(param.Parameterized):
    # Parameters for SED results tab
    galaxy_id = param.String()
    survey = param.String()
    resolved_galaxy = param.Parameter()
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
    other_bagpipes_properties = param.Selector(default="constant:age_max")
    norm = param.Selector(default="linear", objects=["linear", "log"])
    type_select = param.Selector(default="median", objects=["median", "gif"])
    pdf_param_property = param.Selector(
        default="stellar_mass", objects=["stellar_mass", "sfr"]
    )
    upscale_select = param.Boolean(default=False)

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

    TOTAL_FIT_COLORS = {
        "TOTAL_BIN": "springgreen",
        "MAG_AUTO": "blue",
        "MAG_APER": "sandybrown",
        "MAG_ISO": "purple",
        "MAG_APER_TOTAL": "orange",
        "MAG_BEST": "cyan",
        "RESOLVED": "red",
    }

    def __init__(self, resolved_galaxy, facecolor="#f7f7f7", **params):
        galaxy_id = resolved_galaxy.galaxy_id
        survey = resolved_galaxy.survey

        super().__init__(
            galaxy_id=galaxy_id,
            survey=survey,
            resolved_galaxy=resolved_galaxy,
            **params,
        )

        self.facecolor = facecolor
        self.setup_widgets()
        self.setup_streams()
        self.setup_tabs()

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
        self.multi_choice_bins_widget.link(self, value="multi_choice_bins")

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
            options=["constant:age_max"],
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

    def create_sed_results_tab(self):
        sed_results_grid = pn.Column(sizing_mode="stretch_both")

        bin_map = pn.bind(
            self.plot_bins,
            self.param.which_map,
            "nipy_spectral_r",
            self.param.scale_alpha,
            self.param.show_galaxy,
            self.param.show_kron,
            self.param.psf_mode,
            self.param.choose_show_band,
        )

        top_row = pn.Row(
            bin_map, self.plot_sed, sizing_mode="stretch_width", height=350
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
            pdf_controls,
            self.plot_bagpipes_pdf,
            # sizing_mode='stretch_both'
            max_height=400,
            max_width=400,
        )

        sed_results_grid.extend([top_row, mid_row, bot_row, even_lower])

        return sed_results_grid

    def update_active_tab(self, event):
        self.active_tab = event.new
        if hasattr(self, "app"):
            self.app.update_sidebar()

    def setup_streams(self):
        self.stream = hv.streams.Tap(transient=True)

    def setup_tabs(self):
        self.cutouts_tab = pn.param.ParamMethod(
            self.create_cutouts_tab, loading_indicator=True, lazy=True
        )
        self.sed_results_tab = pn.param.ParamMethod(
            self.create_sed_results_tab, loading_indicator=True, lazy=True
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

        if isinstance(self.resolved_galaxy, MockResolvedGalaxy):
            self.synthesizer_tab = pn.panel(
                self.create_synthesizer_tab, loading_indicator=True
            )
            self.info_tabs.append(("Synthesizer", self.synthesizer_tab))

        self.info_tabs.param.watch(self.update_active_tab, "active")

    def create_cutouts_tab(self):
        cutout_grid = pn.Column(scroll=True, sizing_mode="stretch_width")

        # Cutouts row
        cutouts_row = pn.Row(
            pn.bind(self.plot_cutouts, self.param.psf_mode, "star_stack"),
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

    @param.depends("resolved_galaxy", "band_select")
    def do_snr_plot(self):
        fig = self.resolved_galaxy.plot_snr_map(
            band=self.band_select, facecolor=self.facecolor
        )
        plt.close(fig)
        return pn.pane.Matplotlib(
            fig, dpi=144, tight=True, format="svg", width=300, height=300
        )

    @param.depends("resolved_galaxy", "other_plot_select")
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
        "resolved_galaxy",
        "which_map",
        "other_bagpipes_properties",
        "which_run_resolved",
        "norm",
        "total_fit_options",
        "upscale_select",
        "type_select",
    )
    def other_bagpipes_results_func(self):
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
        self.pdf_param_property.options = [
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

    # @lru_cache(maxsize=None)
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
                ]
            )
        elif self.active_tab == 2:  # Photometric Properties
            sidebar.extend(
                [
                    "#### Pixel Binning",
                    self.which_map_widget,
                ]
            )

        # Add Synthesizer tab options if it's a MockResolvedGalaxy
        if isinstance(active_galaxy.resolved_galaxy, MockResolvedGalaxy):
            if (
                active_galaxy.info_tabs.active == 3
            ):  # Assuming Synthesizer is the 4th tab (index 3)
                self.sidebar.append(pn.layout.Divider())
                self.sidebar.append("### Synthesizer Options")
                # Add any specific Synthesizer options here

        return sidebar

    @param.depends(
        "resolved_galaxy",
        "red_channel",
        "green_channel",
        "blue_channel",
        "stretch",
        "q",
        "psf_mode",
    )
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
    def plot_bins(
        self,
        bin_type,
        cmap,
        scale_alpha,
        show_galaxy,
        show_kron,
        psf_matched,
        band,
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
        )

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
            )
            hvplot = new_hvplot * hvplot

        bin_map = pn.pane.HoloViews(
            hvplot, height=300, width=400, aspect_ratio=1
        )
        self.stream.source = bin_map.object

        return bin_map

    # @lru_cache(maxsize=None)
    def plot_cutouts(self, psf_matched_param, psf_type_param):
        """
        Plot cutouts of the galaxy in different bands.

        Args:
            psf_matched_param (str): 'Original' or 'PSF Matched'.
            psf_type_param (str): Type of PSF to use.

        Returns:
            pn.pane.Matplotlib: A Matplotlib pane containing the cutout plots.
        """
        psf_matched = psf_matched_param == "PSF Matched"
        fig = self.resolved_galaxy.plot_cutouts(
            facecolor=self.facecolor,
            psf_matched=psf_matched,
            psf_type=psf_type_param,
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

    # @lru_cache(maxsize=None)
    def plot_phot_property(self, property, cmap, **kwargs):
        """
        Plot a photometric property map for the galaxy.

        Args:
            property (str): The photometric property to plot.
            cmap (str): Colormap to use for the plot.
            **kwargs: Additional keyword arguments for the property calculation.

        Returns:
            pn.pane.Matplotlib: A Matplotlib pane containing the property map.
        """
        options_direct = {
            "Beta": "beta_phot",
            "mUV": "mUV_phot",
            "MUV": "MUV_phot",
            "Lobs_UV": "LUV_phot",
            "AUV": "AUV_from_beta_phot",
            "SFR_UV": "SFR_UV_phot",
            "fesc": "fesc_from_beta_phot",
            "EW_rest_optical": "EW_rest_optical",
            "line_flux_rest_optical": "line_flux_rest_optical",
            "xi_ion": "xi_ion",
        }

        default_options = {"SFR_UV": {"density": True, "logmap": True}}
        if property in default_options:
            kwargs.update(default_options[property])

        if options_direct[property] in kwargs:
            kwargs.update(kwargs[options_direct[property]])

        fig = self.resolved_galaxy.galfind_phot_property_map(
            options_direct[property],
            cmap=cmap,
            facecolor=self.facecolor,
            **kwargs,
        )
        plt.close(fig)

        return pn.pane.Matplotlib(
            fig,
            dpi=144,
            tight=True,
            format="svg",
            sizing_mode="scale_width",
            min_width=200,
        )

    def handle_map_click(self, x, y, mode):
        """
        Handle clicks on the bin map and update relevant plots.

        Args:
            x (float): X-coordinate of the click.
            y (float): Y-coordinate of the click.
            mode (str): The type of plot to update ('sed', 'sfh', 'corner', or 'pdf').

        Returns:
            pn.pane.Matplotlib or None: Updated plot based on the click, or None if click is invalid.
        """
        use = True
        if x is None or y is None:
            use = False
        if use:
            if not (
                0 < x < self.resolved_galaxy.cutout_size
                and 0 < y < self.resolved_galaxy.cutout_size
            ):
                use = False

        map = self.resolved_galaxy.pixedfit_map
        if use:
            rbin = map[int(np.ceil(y)), int(np.ceil(x))]
            if rbin not in self.multi_choice_bins.value:
                self.multi_choice_bins.value = self.multi_choice_bins.value + [
                    rbin
                ]

        multi_choice_bins_param_safe = [
            int(i) if i != "RESOLVED" and i != np.nan else i
            for i in self.multi_choice_bins.value
        ]

        if mode == "sed":
            return self.plot_sed(multi_choice_bins_param_safe)
        elif mode == "sfh":
            return self.plot_sfh(multi_choice_bins_param_safe)
        elif mode == "corner":
            return self.plot_corner(multi_choice_bins_param_safe)
        elif mode == "pdf":
            return self.plot_bagpipes_pdf(multi_choice_bins_param_safe)

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

        return pn.Column(self.which_run_resolved, self.which_run_aperture)

    @param.depends(
        "resolved_galaxy",
        "which_sed_fitter",
        "which_run_resolved",
        "which_run_aperture",
        "multi_choice_bins",
        "total_fit_options",
        "pdf_param_property",
    )
    def plot_bagpipes_pdf(self):
        multi_choice_bins_param_safe = [
            int(i) if i != "RESOLVED" and i != np.nan else i
            for i in self.multi_choice_bins
        ]

        if self.which_sed_fitter == "bagpipes":
            if self.which_run_resolved:
                run_name = self.which_run_resolved
                bins_to_show = multi_choice_bins_param_safe
            elif self.which_run_aperture:
                run_name = self.which_run_aperture
                bins_to_show = self.total_fit_options
            else:
                return pn.pane.Markdown("No Bagpipes results found.")

            fig, _ = self.resolved_galaxy.plot_bagpipes_component_comparison(
                parameter=self.pdf_param_property,
                run_name=run_name,
                bins_to_show=bins_to_show,
                save=False,
                run_dir="pipes/",
                facecolor=self.facecolor,
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
        "resolved_galaxy",
        "which_map",
        "which_sed_fitter",
        "which_flux_unit",
        "multi_choice_bins",
        "which_run_aperture",
        "which_run_resolved",
        "total_fit_options",
        "show_sed_photometry",
    )
    def plot_sed(self):
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

        if len(ax.lines) > 0:
            ax.set_xlim(ax.get_xlim())
            ax.set_ylim(ax.get_ylim())

        ax.legend(loc="upper left", frameon=False, fontsize="small")
        plt.close(fig)
        return pn.pane.Matplotlib(
            fig, dpi=144, tight=True, format="svg", sizing_mode="scale_both"
        )

    @param.depends(
        "resolved_galaxy",
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
        "resolved_galaxy",
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
                figsize=(8, 6),
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

            ax.set_xlabel("Lookback Time (Gyr)")
            ax.set_ylabel("SFR (M$_{\odot}$ yr$^{-1}$)")
            ax.set_yscale("log")
            ax.legend(loc="upper left", fontsize="small")

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

        def create_synthesizer_tab(self):
            synthesizer_page = pn.Column()

            synthesizer_page.append("### Synthesizer Properties")

            top_row = pn.Row()
            middle_row = pn.Row()

            mass_map = self.plot_map_with_controls(
                self.resolved_galaxy.property_images["stellar_mass"],
                label="Stellar Mass",
                unit=r"$M_{\odot}$",
            )

            age_map = self.plot_map_with_controls(
                self.resolved_galaxy.property_images["stellar_age"],
                label="Age",
                unit="Gyr",
            )

            metallicity_map = self.plot_map_with_controls(
                self.resolved_galaxy.property_images["stellar_metallicity"],
                label="Metallicity",
                unit=r"$Z_{\odot}$",
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

    def plot_map_with_controls(self, map_array, label="", unit=""):
        range_slider = pn.widgets.RangeSlider(
            start=np.nanmin(map_array),
            end=np.nanmax(map_array),
            value=(np.nanmin(map_array), np.nanmax(map_array)),
            step=0.01 * (np.nanmax(map_array) - np.nanmin(map_array)),
            name="Colorbar Range",
            format=pn.widgets.NumberFormatter(format="0.1e"),
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

            logtext = "$\log_{10}$ " if log_scale else ""

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
        self.active_galaxy_tab = event.new
        self.update_sidebar()

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
@click.option("--port", default=5004, help="Port to run the server on.")
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
