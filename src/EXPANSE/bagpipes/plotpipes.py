import ast
import glob
import os
import shutil
import sys
import time
import warnings
from pathlib import Path

import astropy.constants as c
import astropy.units as u
import h5py
import matplotlib as mpl
import matplotlib.patches as patches
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatLambdaCDM
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.nddata.utils import NoOverlapError
from astropy.table import Table
from astropy.visualization import (
    ImageNormalize,
    LinearStretch,
    LogStretch,
    ManualInterval,
    make_lupton_rgb,
)
from astropy.wcs import WCS
from matplotlib.ticker import AutoMinorLocator
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from tqdm import tqdm
from corner import corner

# Bye warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
# from plot_spectrum_posterior import *
# Work out if on laptop or not

file_path = os.path.abspath(__file__)
if "nvme" in file_path:
    computer = "morgan"
elif "Users" in file_path:
    computer = "mac"
else:
    computer = "unknown"

if computer == "mac":
    bagpipes_dir = "/Users/user/Documents/PhD/bagpipes_dir/"
    prospector_dir = ""
    db_dir = ""
    # print("Running on Mac.")
elif computer == "morgan":
    bagpipes_dir = "/nvme/scratch/work/tharvey/bagpipes/"
    db_dir = "/nvme/scratch/work/tharvey/dense_basis/pregrids/"
    prospector_dir = "/nvme/scratch/work/tharvey/prospector/output"
    # print("Running on Morgan.")

elif computer == "unknown":
    bagpipes_dir = ""
    db_dir = ""
    prospector_dir = ""


bagpipes_filter_dir = f"{bagpipes_dir}/inputs/filters/"


import sys


def calculate_bins(
    redshift,
    redshift_sfr_start=20,
    log_time=True,
    output_unit="yr",
    return_flat=False,
    num_bins=6,
    first_bin=10 * u.Myr,
    second_bin=None,
    cosmo=FlatLambdaCDM(H0=70, Om0=0.3, Tcmb0=2.725),
):
    time_observed = cosmo.lookback_time(redshift)
    time_sfr_start = cosmo.lookback_time(redshift_sfr_start)
    time_dif = abs(time_observed - time_sfr_start)
    if second_bin is not None:
        assert (
            second_bin > first_bin
        ), "Second bin must be greater than first bin"

    if second_bin is None:
        diff = np.linspace(
            np.log10(first_bin.to(output_unit).value),
            np.log10(time_dif.to(output_unit).value),
            num_bins,
        )
    else:
        diff = np.linspace(
            np.log10(second_bin.to(output_unit).value),
            np.log10(time_dif.to(output_unit).value),
            num_bins - 1,
        )

    if not log_time:
        diff = 10**diff

    if return_flat:
        if second_bin is None:
            return np.concatenate(([0], diff))
        else:
            if log_time:
                return np.concatenate(
                    [[0, np.log10(first_bin.to(output_unit).value)], diff]
                )
            else:
                return np.concatenate(
                    [[0, first_bin.to(output_unit).value], diff]
                )
    bins = []
    bins.append(
        [
            0,
            np.log10(first_bin.to("year").value)
            if log_time
            else first_bin.to("year").value,
        ]
    )
    if second_bin is not None:
        bins.append(
            [
                np.log10(first_bin.to("year").value)
                if log_time
                else first_bin.to("year").value,
                np.log10(second_bin.to("year").value)
                if log_time
                else second_bin.to("year").value,
            ]
        )

    for i in range(1, len(diff)):
        bins.append([diff[i - 1], diff[i]])

    return bins


flexoki_colors = [
    "#D14D41",
    "#DA702C",
    "#D0A215",
    "#879A39",
    "#3AA99F",
    "#4385BE",
    "#8B7EC8",
    "#CE5D97",
]
# try:
# plt.style.use('/nvme/scratch/work/tharvey/scripts/paper.mplstyle')
# print('Using custom style')
# except FileNotFoundError:
#   pass

# Use class-based approach to plotting
# Use pathlib to handle paths
# should be able to load multiple runs at once
# Make catalog version work
# os.environ['DYLD_LIBRARY_PATH'] = '/Users/user/Documents/multinest/MultiNest/lib'
# os.environ['']


def combine_bands(bands, image_paths):
    opened_first = False
    for pos, band in enumerate(bands):
        try:
            path, ext = image_paths[band]
            fits_image = fits.open(path)
            header = fits_image[ext].header
            wcs = WCS(header)
            data = np.array(fits_image[ext].data)
            if not opened_first:
                data_combined = data
                opened_first = True
            elif np.shape(data) == np.shape(data_combined):
                data_combined += data
            return data_combined, wcs
        except FileNotFoundError:
            print(f"No data in {band}.")
            continue

    if not opened_first:
        return None, None


def five_sig_depth_to_n_sig_depth(five_sig_depth, n):
    one_sig_flux = 1 / 5 * 10 ** ((five_sig_depth - 28.08) / -2.5)
    n_sig_mag = -2.5 * np.log10(n * one_sig_flux) + 28.08
    return n_sig_mag


def colormap(
    z, z_range=(6, 21), cmap=mpl.colormaps.get_cmap("gist_rainbow_r")
):
    z = z - z_range[0]
    z_top = z_range[1] - z_range[0]
    z_new = z / z_top
    color = cmap(z_new)
    return color


class PipesFitNoLoad:
    """
    This object is used to load a bagpipes fit directly from the .h5 without reinitializing the galaxy object.
    Will only work if Bagpipes was run with my modified version of Bagpipes.
    """

    def __new__(
        cls,
        galaxy_id,
        field,
        h5_path,
        filter_path=bagpipes_filter_dir,
        bands=None,
        data_func=None,
        **kwargs,
    ):
        # Create a temporary instance to check compatibility
        instance = super().__new__(cls)
        instance.galaxy_id = galaxy_id
        instance.field = field
        instance.h5_path = h5_path
        instance.filter_path = filter_path
        instance.bands = bands
        instance.data_func = data_func
        instance.has_advanced_quantities = True

        # Check compatibility
        instance._check_compatibility()

        # If advanced quantities are not present, return a PipesFit instance instead
        if not instance.has_advanced_quantities:
            print("Falling back to old method.")
            return PipesFit(
                galaxy_id,
                field,
                h5_path,
                filter_path=filter_path,
                bands=bands,
                data_func=data_func,
                **kwargs,
            )

        # Otherwise, return the PipesFitNoLoad instance
        return instance

    def __init__(
        self,
        galaxy_id,
        field,
        h5_path,
        filter_path=bagpipes_filter_dir,
        bands=None,
        data_func=None,
        **kwargs,
    ):
        # Only initialize if we're actually creating a PipesFitNoLoad instance
        self.galaxy_id = galaxy_id
        self.field = field
        self.h5_path = h5_path
        self.bands = bands
        self.data_func = data_func
        self.filter_path = filter_path
        self.wav = None
        self.fit_instructions = None
        self.time_grid = None
        self.dof = None
        self.fitted_params = None
        self.has_advanced_quantities = True

        self._get_fit_instructions()
        if (
            "noise" in self.fit_instructions.keys()
            or "veldisp" in self.fit_instructions.keys()
        ):
            self.fitted_type = "spec"
        else:
            self.fitted_type = "phot"

    def _check_compatibility(self):
        with h5py.File(self.h5_path, "r") as data:
            if "fit_instructions" not in data.attrs.keys():
                raise KeyError("fit_instructions not found in h5 file.")

            if "advanced_quantities" not in data.keys():
                self.has_advanced_quantities = False

    def _load_item_from_h5(
        self, items, percentiles=False, perc=(16, 50, 84), transpose=False
    ):
        if type(items) == str:
            items = [items]
        return_items = []
        with h5py.File(self.h5_path, "r") as data:
            print(self.h5_path)
            for item in items:
                if item in data.keys():
                    return_items.append(data[item][()])
                elif item in data["basic_quantities"].keys():
                    return_items.append(data["basic_quantities"][item][()])
                elif (
                    "advanced_quantities" in data.keys()
                    and item in data["advanced_quantities"].keys()
                ):
                    return_items.append(data["advanced_quantities"][item][()])
                else:
                    raise KeyError(f"{item} not found in h5 file.")

        if percentiles:
            return_items = [
                np.percentile(item, perc, axis=0) for item in return_items
            ]

        if transpose:
            return_items = [np.transpose(item) for item in return_items]

        if len(return_items) == 1:
            return return_items[0]
        else:
            return return_items

    def _list_items(self):
        items = []
        with h5py.File(self.h5_path, "r") as data:
            for key in data.keys():
                if type(data[key]) == h5py._hl.group.Group:
                    for sub_key in data[key].keys():
                        items.append(f"{sub_key}")
                else:
                    items.append(key)
        return items

    def _load_spectrum(self):
        with h5py.File(self.h5_path, "r") as data:
            spectrum = data["spectrum"]
            return spectrum

    def plot_corner_plot(
        self,
        show=False,
        save=True,
        bins=25,
        type="fit_params",
        fig=None,
        color="black",
        facecolor="white",
        extra_samples=["sfr"],
    ):
        self.calculate_dof()

        samples = self._load_item_from_h5(self.fitted_params + extra_samples)

        labels = [f"{param}" for param in self.fitted_params + extra_samples]

        samples = np.array(samples).T
        # Make the corner plot
        fig = corner(
            samples,
            labels=labels,
            quantiles=[0.16, 0.5, 0.84],
            show_titles=True,
            title_kwargs={"fontsize": 13},
            smooth=1.0,
            smooth1d=1.0,
            bins=bins,
            fig=fig,
            color=color,
            facecolor=facecolor,
        )

        return fig

    def calculate_dof(self):
        if self.dof is None:
            self.len_photometry = len(self._load_item_from_h5("photometry"))
            # Loop over fit_instructions and count parameters which are length two lists or
            params = []
            for key, value in self.fit_instructions.items():
                if type(value) == dict:
                    for sub_key, sub_value in value.items():
                        if (
                            type(sub_value) in [list, tuple, np.ndarray]
                            and len(sub_value) == 2
                        ):
                            params.append(f"{key}:{sub_key}")
                elif (
                    type(value) in [list, tuple, np.ndarray]
                    and len(value) == 2
                ):
                    params.append(key)

            self.fitted_params = params

            self.dof = self.len_photometry - len(params)

    def calculate_bic(self):
        dof = self.calculate_dof()
        ln_evidence = self._load_item_from_h5("lnz")[()]
        return -2 * ln_evidence + dof * np.log(self.len_photometry)

    def _get_fit_instructions(self):
        with h5py.File(self.h5_path, "r") as data:
            if "fit_instructions" in data.attrs.keys():
                self.fit_instructions = ast.literal_eval(
                    data.attrs["fit_instructions"]
                )
            else:
                raise KeyError("fit_instructions not found in h5 file.")

    def _recalculate_bagpipes_wavelength_array(
        self,
        bands=None,
        bagpipes_filter_dir=bagpipes_filter_dir,
        use_bpass=False,
    ):
        if bands is None:
            bands = self.bands

        if bands is None:
            raise ValueError("No bands provided or stored in object.")

        paths = [
            glob.glob(f"{bagpipes_filter_dir}/*{band}*")[0] for band in bands
        ]
        if use_bpass:
            from bagpipes import config_bpass as config
        else:
            from bagpipes import config

        from bagpipes.filters import filter_set

        ft = filter_set(paths)
        min_wav = ft.min_phot_wav
        max_wav = ft.max_phot_wav
        max_z = config.max_redshift

        max_wavs = [(min_wav / (1.0 + max_z)), 1.01 * max_wav, 10**8]

        x = [1.0]

        R = [config.R_other, config.R_phot, config.R_other]

        for i in range(len(R)):
            if i == len(R) - 1 or R[i] > R[i + 1]:
                while x[-1] < max_wavs[i]:
                    x.append(x[-1] * (1.0 + 0.5 / R[i]))

            else:
                while x[-1] * (1.0 + 0.5 / R[i]) < max_wavs[i]:
                    x.append(x[-1] * (1.0 + 0.5 / R[i]))

        self.wav = np.array(x)

    def _recreate_filter_set(self):
        from bagpipes.filters import filter_set

        filter_paths = [
            glob.glob(f"{self.filter_path}/*{band}*")[0] for band in self.bands
        ]

        self.filter_set = filter_set(filter_paths)

    def plot_best_photometry(
        self,
        ax,
        colour="black",
        wav_units=u.um,
        flux_units=u.ABmag,
        photometry=None,
        zorder=4,
        y_scale=None,
        skip_no_obs=False,
        background_spectrum=False,
        **kwargs,
    ):
        """Plots best-fitting photometry from fitting

        Args:
            ax (_type_): matplotlib axis object to plot onto.
            colour (str, optional): marker color. Defaults to 'black'.
            wav_units (astropy unit, optional): wavelength units. Defaults to u.um.
            flux_units (astropy unit, optional): flux unit. Defaults to u.ABmag.
            zorder (int, optional): zorder to plot markers on. Defaults to 4.
            y_scale (_type_, optional): _description_. Defaults to None.
            skip_no_obs (bool, optional): _description_. Defaults to False.
            background_spectrum (bool, optional): _description_. Defaults to False.
        """
        if self.fitted_type in ["phot", "both"]:
            if photometry is not None:
                mask = photometry[:, 1] > 0.0
                upper_lims = photometry[:, 1] + photometry[:, 2]
                ymax = 1.05 * np.max(upper_lims[mask])

            else:
                photometry_temp = self._load_item_from_h5(
                    "photometry", percentiles=True, transpose=True
                )
                ymax = 1.05 * np.max(photometry_temp[:, 2])

            if not y_scale:
                y_scale = int(np.log10(ymax)) - 1

            redshift = self._get_redshift()

            self._recalculate_bagpipes_wavelength_array()
            self._recreate_filter_set()

            wavs = self.wav * (1.0 + redshift) * u.AA
            eff_wavs = self.filter_set.eff_wavs * u.AA
            # Convert to desired units.
            wavs_plot = wavs.to(wav_units).value
            eff_wavs.to(wav_units).value

            if background_spectrum:
                spectrum_full = self._load_item_from_h5("spectrum_full")[:]

                spec_post = (
                    np.percentile(
                        spectrum_full,
                        (16, 50, 84),
                        axis=0,
                    ).T
                    * u.erg
                    / (u.cm**2 * u.s * u.AA)
                )

                spec_post = spec_post.astype(
                    float
                )  # fixes weird isfinite error

                flux_nu = spec_post.to(
                    flux_units, equivalencies=u.spectral_density(wavs)
                )

                ax.plot(
                    wavs_plot, flux_nu[:, 1], color="black", zorder=zorder - 1
                )

                ax.fill_between(
                    wavs_plot,
                    flux_nu[:, 0],
                    flux_nu[:, 2],
                    zorder=zorder - 1,
                    color="navajowhite",
                    linewidth=0,
                )

            bestfit_photometry = self._load_item_from_h5(
                "photometry", percentiles=True, transpose=True
            )
            bestfit_photometry *= u.erg / (u.cm**2 * u.s * u.AA)

            bestfit_photometry = bestfit_photometry[:, 1].to(
                flux_units, equivalencies=u.spectral_density(eff_wavs)
            )

            ax.scatter(
                eff_wavs.to(wav_units).value,
                bestfit_photometry.value,
                color=colour,
                zorder=zorder,
                alpha=0.05,
                s=40,
                rasterized=True,
            )

    def _get_redshift(self):
        if "redshift" in self._list_items():
            return np.median(self._load_item_from_h5("redshift"))
        elif "redshift" in self.fit_instructions.keys():
            return self.fit_instructions["redshift"]

    def plot_best_fit(
        self,
        ax,
        colour="black",
        wav_units=u.um,
        flux_units=u.ABmag,
        lw=1,
        fill_uncertainty=False,
        zorder=5,
        label=None,
        return_flux=False,
        **kwargs,
    ):
        """Plot the best-fit SED with optional uncertainties.

        Args:
            ax: Matplotlib axis to plot on
            colour: Color of the line
            wav_units: Units for wavelength axis
            flux_units: Units for flux axis
            lw: Line width
            fill_uncertainty: Whether to show uncertainty bands
            zorder: Plot z-order
            label: Legend label
            return_flux: If True, return wavelength and flux arrays instead of plotting
            **kwargs: Additional keywords passed to plot
        """
        # Get wavelength array and best-fit spectrum samples
        # wavelengths = self._load_item_from_h5('wavelengths')[()]
        spectrum_samples = self._load_item_from_h5("spectrum_full")[:]

        if self.wav is None:
            self._recalculate_bagpipes_wavelength_array()

        wavelengths = self.wav

        redshift = self._get_redshift()
        print(redshift)

        # Get observer frame wavelengths
        wavs = wavelengths * (1 + redshift) * u.AA

        # Calculate percentiles of spectrum samples
        spec_percentiles = np.percentile(
            spectrum_samples, [16, 50, 84], axis=0
        ).T

        # Convert to flux units
        flux_lambda = spec_percentiles * u.erg / (u.s * u.cm**2 * u.AA)
        wavs_3 = np.vstack([wavs, wavs, wavs]).T
        flux = flux_lambda.to(
            flux_units, equivalencies=u.spectral_density(wavs_3)
        )

        if return_flux:
            return wavs.to(wav_units).value, flux[:, 1]

        # Plot median spectrum
        ax.plot(
            wavs.to(wav_units).value,
            flux[:, 1],
            lw=lw,
            color=colour,
            alpha=0.7,
            zorder=zorder,
            label=label if label else self.galaxy_id,
            **kwargs,
        )

        if fill_uncertainty:
            ax.fill_between(
                wavs.to(wav_units).value,
                flux[:, 0],
                flux[:, 2],
                alpha=0.5,
                color=colour,
                lw=0,
                zorder=zorder,
            )

    def _recrate_bagpipes_time_grid(self, log_sampling=0.0025, cosmo=None):
        if cosmo is None:
            cosmo = FlatLambdaCDM(H0=70.0, Om0=0.3)

        self.cosmo = cosmo

        self.z_array = np.arange(0.0, 100.0, 0.01)
        self.age_at_z = cosmo.age(self.z_array).value
        # ldist_at_z = cosmo.luminosity_distance(z_array).value

        self.hubble_time = self.age_at_z[self.z_array == 0.0]

        # Set up the age sampling for internal SFH calculations.
        log_age_max = np.log10(self.hubble_time) + 9.0 + 2 * log_sampling
        self.ages = np.arange(6.0, log_age_max, log_sampling)
        self.ages = 10**self.ages * u.yr

    def plot_sfh(
        self,
        ax,
        colour="black",
        modify_ax=True,
        add_zaxis=True,
        timescale="Myr",
        plottype="lookback",
        logify=False,
        cosmo=None,
        return_sfh=False,
        **kwargs,
    ):
        """Plot star formation history from posterior samples.

        Args:
            ax: Matplotlib axis
            colour: Line color
            modify_ax: Whether to modify axis labels/ticks
            add_zaxis: Add redshift axis
            timescale: Time unit for x-axis
            plottype: 'lookback' or 'absolute' time
            logify: Use log scale for y-axis
            cosmo: Astropy cosmology object
            return_sfh: Return SFH arrays instead of plotting
        """
        # Load SFH data
        sfh_samples = self._load_item_from_h5("sfh")[:]

        self._recrate_bagpipes_time_grid()
        redshift = self._get_redshift()

        age_of_universe = (
            np.interp(redshift, self.z_array, self.age_at_z) * u.yr
        )

        if cosmo is None:
            cosmo = self.cosmo

        if plottype == "lookback":
            times = self.ages
        elif plottype == "absolute":
            times = age_of_universe - self.ages
        # Convert times based on plottype

        # Calculate percentiles
        sfh_percentiles = np.percentile(sfh_samples, [16, 50, 84], axis=0)

        if return_sfh:
            return np.column_stack(
                (times.to(timescale).value, *sfh_percentiles)
            )

        # Plot median SFH
        ax.plot(
            times.to(timescale).value,
            sfh_percentiles[1],
            color=colour,
            zorder=3,
        )

        ax.fill_between(
            times.to(timescale).value,
            sfh_percentiles[0],
            sfh_percentiles[2],
            color=colour,
            alpha=0.3,
            zorder=2,
        )

        if modify_ax:
            ax.set_xlabel(
                f"{'Lookback ' if plottype=='lookback' else ''} Time ({timescale})"
            )
            ax.set_ylabel(r"SFR (M$_{\odot}$ yr$^{-1}$)")

        if logify:
            ax.set_yscale("log")

        if add_zaxis and plottype == "absolute":
            # Add redshift axis based on lookback times
            ax2 = ax.twiny()
            # Calculate redshifts corresponding to lookback times
            # Set ticks and labels
            time_range = ax.get_xlim()
            # Convert time_range to match unit of age_at_z
            z_times = np.linspace(*time_range, 6) * u.Unit(timescale)
            ax2.set_xticks(
                np.interp(z_times.to(u.yr).value, self.z_array, self.age_at_z)
            )
            ax2.set_xlim(time_range)
            ax2.set_xlabel("Redshift", fontsize="small")

    def plot_sed(
        self,
        ax,
        colour="black",
        wav_units=u.um,
        flux_units=u.ABmag,
        x_ticks=None,
        zorder=4,
        ptsize=40,
        y_scale=None,
        lw=1.0,
        skip_no_obs=False,
        fcolour="blue",
        label=None,
        marker="o",
        rerun_fluxes=False,
        **kwargs,
    ):
        """Plot observed photometry - SPEC not working yet.

        Args:
            ax: Matplotlib axis
            colour: Marker edge color
            wav_units: Wavelength axis units
            flux_units: Flux axis units
            zorder: Plot z-order
            ptsize: Point size
            fcolour: Marker face color
            label: Legend label
            marker: Marker style
        """
        # Load observed photometry
        wavelengths = self._load_item_from_h5("obs_wavelengths")[:]
        fluxes = self._load_item_from_h5("obs_fluxes")[:]
        flux_errors = self._load_item_from_h5("obs_flux_errors")[:]

        # Convert to desired units
        wav = wavelengths * u.AA
        flambda = (
            np.vstack([fluxes, flux_errors]).T * u.erg / (u.s * u.cm**2 * u.AA)
        )

        # Convert to f_nu
        wavs_2 = np.vstack([wav, wav]).T
        fnu = flambda * wavs_2**2 / c.c
        fnu_jy = fnu[:, 0].to(u.Jy)
        fnu_jy_err = fnu[:, 1].to(u.Jy)

        # Convert to requested flux units
        if flux_units == u.ABmag:
            plot_fnu = fnu[:, 0].to(u.ABmag).value
            # Calculate asymmetric errors in magnitudes
            inner = fnu_jy / (fnu_jy - fnu_jy_err)
            err_up = 2.5 * np.log10(inner)
            err_low = 2.5 * np.log10(1 + (fnu_jy_err / fnu_jy))
        else:
            plot_fnu = fnu[:, 0].to(flux_units).value
            err_up = fnu_jy_err.to(flux_units).value
            err_low = fnu_jy_err.to(flux_units).value

        wav = wav.to(wav_units).value

        # Plot photometry points
        mask = ~np.isnan(plot_fnu)
        if skip_no_obs:
            mask &= plot_fnu != 0

        ax.errorbar(
            wav[mask],
            plot_fnu[mask],
            yerr=[err_low[mask], err_up[mask]],
            lw=lw,
            linestyle=" ",
            capsize=3,
            capthick=lw,
            zorder=zorder - 1,
            color=colour,
        )

        ax.scatter(
            wav[mask],
            plot_fnu[mask],
            color=colour,
            s=ptsize,
            zorder=zorder,
            linewidth=lw,
            facecolor=fcolour,
            edgecolor=colour,
            marker=marker,
            label=label,
        )

    def plot_pdf(
        self,
        ax,
        parameter,
        colour="black",
        fill_between=False,
        alpha=1,
        return_samples=False,
        linelabel="",
        norm_height=False,
    ):
        """Plot posterior PDF for a parameter.

        Args:
            ax: Matplotlib axis
            parameter: Parameter name to plot
            colour: Line color
            fill_between: Fill between line and zero
            alpha: Line opacity
            return_samples: Return raw samples instead of plotting
            linelabel: Legend label
            norm_height: Normalize peak height to 1
        """
        # Load parameter samples
        samples = self._load_item_from_h5(parameter)[:]

        # Special handling for some parameters
        if parameter == "sfr" or parameter == "formed_mass":
            samples = np.log10(samples)
            label = rf"$\log_{{10}}({parameter})$"
        else:
            label = parameter

        if return_samples:
            return samples

        # Calculate smoothed histogram
        from scipy.stats import gaussian_kde

        kde = gaussian_kde(samples[~np.isnan(samples)])
        x_plot = np.linspace(np.min(samples), np.max(samples), 100)
        y = kde(x_plot)

        if norm_height:
            y = y / np.max(y)

        # Plot PDF
        ax.plot(x_plot, y, color=colour, alpha=alpha, label=linelabel)

        if fill_between:
            ax.fill_between(x_plot, y, color=colour, alpha=0.3)

        ax.set_xlabel(label, fontsize="small", fontweight="bold")

        # Remove y ticks since PDF height is arbitrary
        ax.set_yticks([])

        # Auto x ticks
        from matplotlib.ticker import MaxNLocator

        ax.xaxis.set_major_locator(MaxNLocator(3))
        ax.tick_params(axis="both", which="major", labelsize="medium")


class PipesFit:
    def __init__(
        self,
        galaxy_id,
        field,
        h5_path,
        pipes_path,
        catalog,
        overall_field=None,
        load_spectrum=False,
        filter_path=bagpipes_filter_dir,
        get_advanced_quantities=True,
        ID_col="NUMBER",
        field_col="field",
        catalogue_flux_unit=u.MJy / u.sr,
        bands=None,
        data_func=None,
        attempt_to_load_direct_h5=False,
    ):
        time.time()
        # Manually loading the .h5 file with deepdish
        if overall_field is None:
            self.overall_field = field
        else:
            self.overall_field = overall_field
        self.galaxy_id = galaxy_id
        self.field = field
        self.h5_path = h5_path
        self.pipes_path = pipes_path
        self.catalog = catalog
        self.catalogue_flux_unit = catalogue_flux_unit

        self.bands = bands
        self.data_func = data_func
        self.has_advanced_quantities = get_advanced_quantities

        if self.catalog is not None:
            if len(self.catalog) > 1:
                raise Exception(
                    f"{len(self.catalog)} matches for {galaxy_id} in column {ID_col}"
                )
            try:
                if (
                    f"{catalog[ID_col][0]}_{catalog[field_col][0]}"
                    != galaxy_id
                ):
                    print(f"{catalog[ID_col][0]}_{catalog[field_col][0]}")
                    print(galaxy_id)
                    raise Exception("Catalogue doesn't match galaxy ID")
            except KeyError:
                pass
        # else:
        # print("No catalog provided. Output will be limited.")

        path = Path(h5_path)
        pipes_path = Path(pipes_path)

        os.chdir(pipes_path.parent)
        # print(f"Changed directory to {pipes_path.parent}")
        # data = dd.io.load(h5_path)
        data = h5py.File(h5_path, "r")
        # fit_instructions is attribute of data
        fit_instructions = data.attrs["fit_instructions"]
        try:
            self.config_used = eval(data.attrs["config"])
            if self.config_used["type"] == "BPASS":
                os.environ["use_bpass"] = str(int(True))
            elif self.config_used["type"] == "BC03":
                os.environ["use_bpass"] = str(int(False))
        except KeyError:
            self.config_used = None
            pass

        data.close()

        # Only import after we check the config

        if not attempt_to_load_direct_h5:
            # Reload the module if it's already been imported
            if "bagpipes" in sys.modules:
                import importlib

                importlib.reload(sys.modules["bagpipes"])
                from bagpipes import config, fit, galaxy

            else:
                # import run_bagpipes
                from bagpipes import config, fit, galaxy

            if self.config_used is not None:
                assert (
                    config.stellar_file == self.config_used["stellar_file"]
                ), f'{config.stellar_file} != {self.config_used["stellar_file"]}'
                assert (
                    config.neb_line_file == self.config_used["neb_line_file"]
                ), f'{config.neb_line_file} != {self.config_used["neb_line_file"]}'
                assert (
                    config.neb_cont_file == self.config_used["neb_cont_file"]
                ), f'{config.neb_cont_file} != {self.config_used["neb_cont_file"]}'

            # fit_instructions = data['fit_instructions']
            if type(fit_instructions) in [str, np.str_]:
                fit_instructions = ast.literal_eval(fit_instructions)

            """samples = data['samples2d']
            basic_quantities = data['basic_quantities']
            lnz = data['lnz']
            median = data['median']
            """

            # out_subdir = path.relative_to(f'{pipes_path}/posterior/')

            # Copy .h5 and rename
            if not os.path.exists(f"{pipes_path}/posterior/plot_temp/"):
                os.makedirs(f"{pipes_path}/posterior/plot_temp/")
            if not os.path.exists(f"{pipes_path}/plots/plot_temp/"):
                os.makedirs(f"{pipes_path}/plots/plot_temp/")

            temp_file = Path(
                f"{pipes_path}/posterior/plot_temp/{galaxy_id}.h5"
            )

            print(f"temp, {pipes_path}/posterior/plot_temp/{galaxy_id}.h5")

            shutil.copy(path, temp_file)
            out_subdir = temp_file.relative_to(
                f"{pipes_path}/posterior/"
            ).parent

            if "excluded" in str(h5_path):
                self.excluded_bands = h5_path.split("excluded")[-1].split("/")[
                    0
                ]
            else:
                self.excluded_bands = ""

            os.environ["excluded_bands"] = self.excluded_bands
            # Recreating the galaxy object

            if (
                "noise" in fit_instructions.keys()
                or "veldisp" in fit_instructions.keys()
            ):
                self.fitted_type = "spec"
            else:
                self.fitted_type = "phot"

            if self.fitted_type == "phot":
                if self.data_func is None:
                    self.data_func = run_bagpipes.load_fits
                os.environ["input_unit"] = self.catalogue_flux_unit.to_string()
                if self.bands is None:
                    self.bands = run_bagpipes.load_fits(
                        galaxy_id, return_bands=True, verbose=False
                    )
                # Get filter paths
                self.filts = [
                    f"{filter_path}/{filt}_LePhare.txt" for filt in self.bands
                ]
                spectrum_exists = False
                photometry_exists = True

            if self.fitted_type == "spec":
                if self.data_func is None:
                    self.data_func = run_bagpipes.load_spectra
                if self.bands is None:
                    self.bands = None
                spectrum_exists = True
                photometry_exists = False
                self.filts = None

            try:
                self.galaxy = galaxy(
                    galaxy_id,
                    self.data_func,
                    filt_list=self.filts,
                    spectrum_exists=spectrum_exists,
                    photometry_exists=photometry_exists,
                )

            except Exception as e:
                print(f"Error in {galaxy_id}")
                raise e

            # Recreating the posterior object
            # print(self.h5_path)
            try:
                self.fit = fit(
                    self.galaxy, fit_instructions, run=str(out_subdir)
                )
                self.fit.fit(verbose=True)
            except KeyError:
                raise Exception(f"Couldn't recreate {self.h5_path}. Skipping")

            # Get quantities
            self.fit.posterior.get_basic_quantities()

            if get_advanced_quantities:
                self.fit.posterior.get_advanced_quantities()

            dust = fit_instructions["dust"]["type"]
            try:
                dust_prior = fit_instructions["dust"][
                    "Av_prior"
                ]  # Some may not have this
            except KeyError:
                dust_prior = "uniform"
            # redshift = fit_instructions.get('redshift', False) # This is range of redshift allowed to fit
            (
                self.stellar_mass_16,
                self.stellar_mass_50,
                self.stellar_mass_84,
            ) = np.percentile(
                self.fit.posterior.samples["stellar_mass"], (16, 50, 84)
            )

            if "redshift" in self.fit.fitted_model.params:
                if fit_instructions.get("redshift_prior_sigma", False):
                    self.zphot = np.median(
                        self.fit.posterior.samples["redshift"]
                    )
                    self.zphot_16 = f'-{self.zphot - np.percentile(self.fit.posterior.samples["redshift"], 16):.1f}'
                    self.zphot_84 = f'+{np.percentile(self.fit.posterior.samples["redshift"], 84)-self.zphot:.1f}'

            else:
                # print(self.fit.fitted_model.model_components.keys())
                self.zphot = self.fit.fitted_model.model_components["redshift"]
                self.zphot_16 = ""
                self.zphot_84 = ""
            # print('zphot', self.zphot)

            sfh = [
                x.split(":")[0]
                for x in self.fit.fitted_model.params
                if x.split(":")[-1] == "massformed"
            ][0]
        else:
            # need to set zphot, sfh name
            # need to set self.fit, self.galaxy and load quantities.
            raise NotImplementedError

        if "bin_edges" in fit_instructions[sfh].keys():
            age_prior = ""  #    continuity prior
        else:
            try:
                age_prior = fit_instructions[sfh]["age_prior"]
            except KeyError:
                try:
                    age_prior = fit_instructions[sfh]["tstart_prior"]
                except KeyError:
                    age_prior = ""

        metallicity_prior = fit_instructions[sfh].get(
            "metallicity_prior", "uniform"
        )

        if dust_prior == "log_10":
            dust_prior = r"$\mathrm{log_{10}}$"
        if metallicity_prior == "log_10":
            metallicity_prior = r"$\mathrm{log_{10}}$"
        if age_prior == "log_10":
            age_prior = r"$\mathrm{log_{10}}$"
        if age_prior == "uniform":
            age_prior = "flat"
        if "bursty" in self.h5_path:
            sfh = "bursty"
        if "bursty_blue" in self.h5_path:
            sfh = "bursty (blue)"
        sps = "BC03"
        if "bpass" in self.h5_path:
            sps = "BPASS"
        fesc = 0.0
        if "fesc" in self.h5_path:
            try:
                fesc = [
                    float(i[3:])
                    for i in self.h5_path.split("_")
                    if "fesc" in i
                ][0]
            except:
                fesc = "free"

        self.chi2 = self.calc_low_chi2()
        fiducial = {
            "sfh": "lognormal",
            "dust_type": "Calzetti",
            "dust_prior": r"$\mathrm{log_{10}}$",
            "metallicity_prior": r"$\mathrm{log_{10}}$",
            "age_prior": r"$\mathrm{log_{10}}$",
            "redshift": "zgauss",
            "fesc": 0.0,
            "sps": "BC03",
        }
        show = []

        if sfh != fiducial["sfh"]:
            show.append("sfh")
        if dust != fiducial["dust_type"]:
            show.append("dust")
        if dust_prior != fiducial["dust_prior"]:
            show.append("dust_prior")
        if metallicity_prior != fiducial["metallicity_prior"]:
            show.append("metallicity_prior")
        if age_prior != fiducial["age_prior"]:
            if age_prior != "":
                show.append("age_prior")
        if fesc != fiducial["fesc"]:
            show.append("fesc")
        if sps != fiducial["sps"]:
            show.append("sps")

        if len(show) == 0:
            self.name = "BP Fiducial"
        else:
            self.name = ["BP"]
            for key in show:
                if key == "sfh":
                    self.name.append(f"{sfh} SFH")
                elif key == "dust":
                    self.name.append(f"Dust: {dust}")
                elif key == "dust_prior":
                    if "dust" in show:
                        self.name.append(f" {dust_prior}")
                    else:
                        self.name.append(f"Dust:{dust_prior}")
                elif key == "metallicity_prior":
                    self.name.append(f"Z: {metallicity_prior}")
                elif key == "age_prior":
                    self.name.append(f"Age: {age_prior}")
                if key == "fesc":
                    self.name.append(f"f$_{{esc}}$: {fesc}")
                if key == "sps":
                    self.name.append(f"{sps} SPS")

            self.name = ", ".join(self.name)
        # _{{\\rm{{phot}}}}
        end = f"$z:{self.zphot:.1f}^{{{self.zphot_84}}}_{{{self.zphot_16}}} \\ \\log{{M_{{*}}}}:{self.stellar_mass_50:.1f}^{{+{self.stellar_mass_84-self.stellar_mass_50:.1f}}}_{{-{self.stellar_mass_50-self.stellar_mass_16:.1f}}} \\ \\chi^2:{self.chi2:<4.1f}$"
        self.name = f"{end:<30} ({self.name})"
        # self.name = f'SFH: {sfh[:3]} Age: {age_prior:<7} Dust: {dust[:3]}, {dust_prior:<10} Z: {metallicity_prior:<10} z$_{{phot}}$: {self.zphot:.2f}$^{{{self.zphot_84}}}_{{{self.zphot_16}}}$ $\chi^2$: {self.chi2:.2f}'

        if self.excluded_bands != "":
            print("Adding excluded bands to name.")
            self.name += f' Exclude: {self.excluded_bands.replace("_", " ")}'

        # print(self.name)
        try:
            shutil.rmtree(temp_file.parent)
        except:
            pass

    def add_advanced_quantities(self):
        self.fit.posterior.get_advanced_quantities()
        self.has_advanced_quantities = True

    def plot_csfh(
        self,
        ax,
        colour="black",
        modify_ax=True,
        add_zaxis=True,
        timescale="Gyr",
        plottype="lookback",
        logify=False,
        cosmo=None,
        **kwargs,
    ):
        add_csfh_posterior(
            self.fit,
            ax,
            color=colour,
            use_color=True,
            zvals=[6.5, 8, 10, 13, 16, 20],
            z_axis=add_zaxis,
            alpha=0.3,
            plottype=plottype,
            timescale=timescale,
        )
        if modify_ax:
            if plottype == "lookback":
                ax.set_xlabel(
                    f"$\\mathbf{{\\mathrm{{Lookback\\ Time \\ ({timescale})}}}}$",
                    fontsize="small",
                    path_effects=[
                        pe.withStroke(linewidth=2, foreground="white")
                    ],
                )

            elif plottype == "absolute":
                ax.set_xlabel(
                    f"$\\mathbf{{\\mathrm{{Age\\ of\\ Universe \\ ({timescale})}}}}$",
                    fontsize="small",
                    patheffects=[
                        pe.withStroke(linewidth=2, foreground="white")
                    ],
                )

            ax.set_ylabel(
                "$\\mathbf{\\mathrm{SFR\\ (M_\\odot)}$", fontsize="medium"
            )

            # ax.set_ylabel('SFR (M$_\odot$ yr$^{-1}$)', fontsize='medium')
            ax.tick_params(axis="both", which="major", labelsize="medium")
            ax.tick_params(axis="both", which="minor", labelsize="medium")

        if logify:
            ax.set_xscale("log")
            ax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())

    def plot_corner_plot(
        self,
        show=False,
        save=True,
        bins=25,
        type="fit_params",
        fig=None,
        color="black",
        facecolor="white",
    ):
        from bagpipes import plot_corner

        fig = plot_corner(
            self.fit,
            show=show,
            save=save,
            bins=bins,
            type=type,
            fig=fig,
            color=color,
            facecolor=facecolor,
        )
        return fig

    def plot_sfh(
        self,
        ax,
        colour="black",
        modify_ax=True,
        add_zaxis=True,
        timescale="Myr",
        plottype="lookback",
        logify=False,
        cosmo=None,
        return_sfh=False,
        **kwargs,
    ):
        from bagpipes import add_sfh_posterior

        sfh = add_sfh_posterior(
            self.fit,
            ax,
            color=colour,
            use_color=True,
            zvals=[6.5, 8, 10, 13, 16, 20],
            z_axis=add_zaxis,
            alpha=0.3,
            plottype=plottype,
            timescale=timescale,
            save=False,
            return_sfh=return_sfh,
            **kwargs,
        )
        if modify_ax:
            if plottype == "lookback":
                ax.set_xlabel(
                    f"$\\mathbf{{\\mathrm{{Lookback\\ Time \\ ({timescale})}}}}$",
                    fontsize="small",
                    path_effects=[
                        pe.withStroke(linewidth=2, foreground="white")
                    ],
                )

            elif plottype == "absolute":
                ax.set_xlabel(
                    f"$\\mathbf{{\\mathrm{{Age\\ of\\ Universe \\ ({timescale})}}}}$",
                    fontsize="small",
                    path_effects=[
                        pe.withStroke(linewidth=2, foreground="white")
                    ],
                )

            ax.set_ylabel(
                "$\\mathrm{SFR\\ (M_\\odot\\ \\mathrm{yr}^{-1}})$",
                fontsize="medium",
            )
            # Move label to the other side
            # ax.yaxis.set_label_position("right")

            # ax.set_ylabel('SFR (M$_\odot$ yr$^{-1}$)', fontsize='medium')
            ax.tick_params(axis="both", which="major", labelsize="medium")
            ax.tick_params(axis="both", which="minor", labelsize="medium")
        if logify:
            ax.set_xscale("log")
            ax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
            # ax.yaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
        if return_sfh:
            return sfh

    def plot_best_photometry(
        self,
        ax,
        colour="black",
        wav_units=u.um,
        flux_units=u.ABmag,
        zorder=4,
        y_scale=None,
        skip_no_obs=False,
        background_spectrum=False,
        **kwargs,
    ):
        """Plots best-fitting photometry from fitting

        Args:
            ax (_type_): matplotlib axis object to plot onto.
            colour (str, optional): marker color. Defaults to 'black'.
            wav_units (astropy unit, optional): wavelength units. Defaults to u.um.
            flux_units (astropy unit, optional): flux unit. Defaults to u.ABmag.
            zorder (int, optional): zorder to plot markers on. Defaults to 4.
            y_scale (_type_, optional): _description_. Defaults to None.
            skip_no_obs (bool, optional): _description_. Defaults to False.
            background_spectrum (bool, optional): _description_. Defaults to False.
        """
        if self.fitted_type in ["phot", "both"]:
            fit = self.fit
            mask = fit.galaxy.photometry[:, 1] > 0.0
            upper_lims = (
                fit.galaxy.photometry[:, 1] + fit.galaxy.photometry[:, 2]
            )
            ymax = 1.05 * np.max(upper_lims[mask])

            if not y_scale:
                y_scale = int(np.log10(ymax)) - 1

            # Calculate posterior median redshift.
            if "redshift" in fit.fitted_model.params:
                redshift = np.median(fit.posterior.samples["redshift"])

            else:
                redshift = fit.fitted_model.model_components["redshift"]

            # Plot the posterior photometry and full spectrum.
            wavs = (
                fit.posterior.model_galaxy.wavelengths
                * (1.0 + redshift)
                * u.AA
            )
            eff_wavs = fit.galaxy.filter_set.eff_wavs * u.AA
            # Convert to desired units.
            wavs_plot = wavs.to(wav_units).value
            eff_wavs.to(wav_units).value

            if background_spectrum:
                spec_post = (
                    np.percentile(
                        fit.posterior.samples["spectrum_full"],
                        (16, 84),
                        axis=0,
                    ).T
                    * u.erg
                    / (u.cm**2 * u.s * u.AA)
                )

                spec_post = spec_post.astype(
                    float
                )  # fixes weird isfinite error

                wavs_micron_3 = np.vstack([wavs, wavs]).T
                flux_nu = spec_post * wavs_micron_3**2 / c.c
                flux_nu = flux_nu.to(flux_units).value

                ax.plot(
                    wavs_plot, flux_nu[:, 0], color="black", zorder=zorder - 1
                )

                ax.plot(
                    wavs_plot, flux_nu[:, 1], color="black", zorder=zorder - 1
                )

                ax.fill_between(
                    wavs_plot,
                    flux_nu[:, 0],
                    flux_nu[:, 1],
                    zorder=zorder - 1,
                    color="navajowhite",
                    linewidth=0,
                )

            phot_post = (
                np.percentile(
                    fit.posterior.samples["photometry"], (16, 84), axis=0
                ).T
                * u.erg
                / (u.cm**2 * u.s * u.AA)
            )

            for j in range(fit.galaxy.photometry.shape[0]):
                if skip_no_obs and fit.galaxy.photometry[j, 1] == 0.0:
                    continue

                phot_band = (
                    fit.posterior.samples["photometry"][:, j]
                    * u.erg
                    / (u.cm**2 * u.s * u.AA)
                )
                mask = (phot_band > phot_post[j, 0]) & (
                    phot_band < phot_post[j, 1]
                )
                phot_1sig = phot_band[mask]
                wav_array = np.zeros(phot_1sig.shape[0]) + eff_wavs[j]
                phot_1sig = phot_1sig * wav_array**2 / c.c
                phot_1sig = phot_1sig.to(flux_units).value
                wav_array = wav_array.to(wav_units).value

                if len(phot_1sig) > 0:
                    ax.scatter(
                        wav_array,
                        phot_1sig,
                        color=colour,
                        zorder=zorder,
                        alpha=0.05,
                        s=40,
                        rasterized=True,
                    )

    def calc_low_chi2(self):
        if "chisq_phot" in self.fit.posterior.samples.keys():
            chi2 = np.min(self.fit.posterior.samples["chisq_phot"])
            return chi2
        else:
            return -99.99

    def plot_best_fit(
        self,
        ax,
        colour="black",
        wav_units=u.um,
        flux_units=u.ABmag,
        lw=1,
        fill_uncertainty=False,
        zorder=5,
        label=None,
        return_flux=False,
        **kwargs,
    ):
        if "redshift" in self.fit.fitted_model.params:
            redshift = np.median(self.fit.posterior.samples["redshift"])
        else:
            redshift = self.fit.fitted_model.model_components["redshift"]
        wavs_aa = (
            self.fit.posterior.model_galaxy.wavelengths
            * (1.0 + redshift)
            * u.AA
        )

        spec_post = np.percentile(
            self.fit.posterior.samples["spectrum_full"], (16, 50, 84), axis=0
        ).T

        spec_post = spec_post.astype(float)  # fixes weird isfinite error

        flux_lambda = spec_post * u.erg / (u.s * u.cm**2 * u.AA)

        wavs_micron_3 = np.vstack([wavs_aa, wavs_aa, wavs_aa]).T

        flux = flux_lambda.to(
            flux_units, equivalencies=u.spectral_density(wavs_micron_3)
        ).value

        # flux_nu = flux_lambda * wavs_micron_3**2/c.c

        # flux_nu = flux_nu.to(flux_units).value

        if label is None:
            label = self.name

        wavs = wavs_aa.to(wav_units).value

        if return_flux:
            # Return the median fluxes
            return wavs, flux[:, 1]

        ax.plot(
            wavs,
            flux[:, 1],
            lw=lw,
            color=colour,
            alpha=0.7,
            zorder=zorder,
            label=label,
            **kwargs,
        )
        if fill_uncertainty:
            ax.fill_between(
                wavs,
                flux[:, 0],
                flux[:, 2],
                alpha=0.5,
                color=colour,
                lw=lw,
                zorder=zorder,
            )

    def plot_sed(
        self,
        ax,
        colour="black",
        wav_units=u.um,
        flux_units=u.ABmag,
        x_ticks=None,
        zorder=4,
        ptsize=40,
        y_scale=None,
        lw=1.0,
        skip_no_obs=False,
        fcolour="blue",
        label=None,
        marker="o",
        rerun_fluxes=False,
        **kwargs,
    ):
        if self.fitted_type in ["phot", "both"]:
            photometry = np.copy(self.galaxy.photometry)
            mask = photometry[:, 1] > 0.0
            ymax = 1.05 * np.max((photometry[:, 1] + photometry[:, 2])[mask])

            if not y_scale:
                y_scale = int(np.log10(ymax)) - 1

            wav = photometry[:, 0] * u.AA
            flambda = photometry[:, 1:] * u.erg / (u.s * u.cm**2 * u.AA)
            wavs_2 = np.vstack([wav, wav]).T
            fnu = flambda * wavs_2**2 / c.c
            fnu_jy = fnu[:, 0].to(u.Jy)
            fnu_jy_err = fnu[:, 1].to(u.Jy)
            plot_fnu = fnu[:, 0].to(flux_units).value
            if rerun_fluxes:
                fluxes, bands = run_bagpipes.load_fits(
                    self.galaxy_id,
                    return_both=True,
                    verbose=False,
                    input_unit=self.catalogue_flux_unit,
                )
                plot_fnu = fluxes * u.uJy
                plot_fnu = plot_fnu[:, 0].to(flux_units).value
                fnu_jy = fluxes[:, 0] * u.uJy
                fnu_jy_err = fluxes[:, 1] * u.uJy

            non_detect_mask = np.ones(shape=len(self.bands), dtype=bool)
            three_sig_depths = []
            try:
                for pos, band in enumerate(self.bands):
                    if self.catalog is not None:
                        sigma = self.catalog[f"sigma_{band}"]
                    else:
                        sigma = [[100]]
                    if sigma[0][0] < 2:
                        non_detect_mask[pos] = False
                        loc_depth = self.catalog[f"loc_depth_{band}"][0][0]
                        three_sig_depth = (
                            useful_funcs.five_sig_depth_to_n_sig_depth(
                                loc_depth, 3
                            )
                            * u.ABmag
                        )
                        three_sig_depth = three_sig_depth.to(flux_units).value
                        wav_pos = wav[pos].to(wav_units).value
                        p1 = patches.FancyArrowPatch(
                            (wav_pos, three_sig_depth),
                            (wav_pos, three_sig_depth + 0.5),
                            arrowstyle="-|>",
                            mutation_scale=10,
                            alpha=1,
                            color=colour,
                        )
                        ax.add_patch(p1)
                        three_sig_depths.append(three_sig_depth)
            except KeyError:
                pass
            # If it fails calculate smaller error
            if flux_units == u.ABmag:
                inner = fnu_jy / (fnu_jy - fnu_jy_err)
                err_up = 2.5 * np.log10(inner)
                err_low = 2.5 * np.log10(1 + (fnu_jy_err / fnu_jy))
            else:
                err_up = fnu_jy_err.to(flux_units).value
                err_low = fnu_jy_err.to(flux_units).value

            wav = wav.to(wav_units).value
            # Errors wrong - need assymetric
            # Plot the data
            # Apply masks

            wav_lim = wav[non_detect_mask]
            plot_fnu = plot_fnu[non_detect_mask]
            err_up = err_up[non_detect_mask]
            err_low = err_low[non_detect_mask]
            mask = ~np.isfinite(plot_fnu)
            err_low[mask] = 0
            err_up[mask] = 0

            ax.errorbar(
                wav_lim,
                plot_fnu,
                yerr=[err_low, err_up],
                lw=lw,
                linestyle=" ",
                capsize=3,
                capthick=lw,
                zorder=zorder - 1,
                color=colour,
            )

            ax.scatter(
                wav_lim,
                plot_fnu,
                color=colour,
                s=ptsize,
                zorder=zorder,
                linewidth=lw,
                facecolor=fcolour,
                edgecolor=colour,
                marker=marker,
                label=label,
            )

            if flux_units == u.ABmag:
                plot_fnu_lim = plot_fnu[(plot_fnu > 15) & (plot_fnu < 32)]
                plot_fnu_lim = np.append(plot_fnu_lim, three_sig_depths)
                plot_fnu_up = np.max(plot_fnu_lim) + 1
                plot_fnu_low = np.min(plot_fnu_lim) - 2
                ax.set_ylim(plot_fnu_up, plot_fnu_low)
            ax.set_xlim(wav[0] - 0.2, wav[-1] + 0.65)

        if self.fitted_type in ["spec", "both"]:
            spectrum = np.copy(self.galaxy.spectrum)

            wav = spectrum[:, 0] * u.AA
            flambda = spectrum[:, 1:] * u.erg / (u.s * u.cm**2 * u.AA)
            wavs_2 = np.vstack([wav, wav]).T
            fnu = flambda * wavs_2**2 / c.c
            fnu_jy = fnu[:, 0].to(u.Jy)
            fnu_jy_err = fnu[:, 1].to(u.Jy)
            wav = wav.to(wav_units).value
            fnu_plot = fnu_jy.to(flux_units).value
            fnu_jy_err.to(flux_units).value

            ax.plot(
                wav,
                fnu_plot,
                color=color,
                zorder=zorder,
                lw=lw,
                label=label,
                alpha=0.7,
            )

            """
            Make this work - need assymetric uncertanties
            if spectrum.shape[1] == 2:

                ax.plot(wav, fnu_plot,
                        color=color, zorder=zorder, lw=lw, label=label, alpha=0.7)

            elif spectrum.shape[1] == 3:

                ax.plot(wav, fnu_plot, label = label,
                        color=color, zorder=zorder, lw=lw, alpha=0.7)

                lower = (fnu_out - fn)
                upper = (spectrum[:, 1] + spectrum[:, 2])

                ax.fill_between(spectrum[:, 0], lower, upper, color=color,
                                zorder=zorder-1, alpha=0.75, linewidth=0)
            """
            # Sort out x tick locations
            ax.set_xlim(0.6, 5)
            ax.set_ylim(31, 26)
            """if x_ticks is None:
                auto_x_ticks(ax)

            else:
                ax.set_xticks(x_ticks)"""

            # Sort out axis labels.
            # auto_axis_label(ax, y_scale, z_non_zero=z_non_zero)

    def plot_pdf(
        self,
        ax,
        parameter,
        colour="black",
        fill_between=False,
        alpha=1,
        return_samples=False,
        linelabel="",
        norm_height=False,
        **kwargs,
    ):
        # Generate list of parameters to plot
        fit = self.fit
        sfh_names = [
            "stellar_mass",
            "sfr",
            "ssfr",
            "tform",
            "nsfr",
            "formed_mass",
            "UV_colour",
            "VJ_colour",
            "mass_weighted_age",
            "dust:Av",
            "tquench",
            "sfr_10myr",
            "sfr_100myr",
        ]
        names = sfh_names + fit.fitted_model.params
        if parameter not in names:
            # This deals with name changes between SFH components
            val = [
                x.split(":")[0]
                for x in fit.fitted_model.params
                if x.split(":")[-1] == "massformed"
            ][0]
            parameter = f"{val}:{parameter}"
            # if parameter not in names:
            #     raise Exception(f'{parameter} was not fit.')
        # print(names, parameter)
        if parameter not in names:
            raise Exception(f"{parameter} was not fit. Not in {names}")

        np.argwhere(np.array(names) == parameter)[0][0]

        # Get axis labels
        name = parameter
        from bagpipes.plotting import fix_param_names

        label = fix_param_names(name)

        samples = np.copy(fit.posterior.samples[name])

        # Log parameter samples and labels for parameters with log priors
        if (name == "sfr" or name == "formed_mass") and not return_samples:
            samples = np.log10(samples)
            label = "$\\mathrm{log_{10}}(" + label[1:-1] + ")$"

        # Replace any r params for Dirichlet distributions with t_x vals
        if "dirichlet" in name:
            comp = name.split(":")[0]
            samples = fit.posterior.samples[comp + ":tx"][:, j]
            n_x = fit.fitted_model.model_components[comp]["bins"]
            t_percentile = int(np.round(100 * (j + 1) / n_x))
            j += 1
            label = "$t_{" + str(t_percentile) + r"}\ /\ \mathrm{Gyr}$"

        if return_samples:
            return samples

        try:
            from bagpipes.plotting import hist1d

            hist1d(
                samples[np.invert(np.isnan(samples))],
                ax,
                smooth=True,
                color=colour,
                percentiles=False,
                lw=1,
                alpha=alpha,
                fill_between=fill_between,
                label=linelabel,
                norm_height=norm_height,
            )

        except ValueError:
            print("I shit myself.")
            pass

        ax.set_xlabel(label, fontsize="small", fontweight="bold")
        # ax.set_title(parameter, fontsize='small', fontweight='bold')
        from bagpipes.plotting import auto_x_ticks

        auto_x_ticks(ax, nticks=3)
        ax.tick_params(axis="both", which="major", labelsize="medium")
        ax.tick_params(axis="both", which="minor", labelsize="medium")


# Need to give Plotspector
# plot_pdf(ax, parameter, color, etc)
# plot_best_fit(ax, color, etc)
# plot_best_photometry(ax, color, etc)


class PlotPipes:
    def __init__(
        self,
        galaxy_id,
        field,
        pipes_path=f"{bagpipes_dir}/pipes/",
        match_field=True,
        match_version=False,
        catalog_path="",
        spectra_path="",
        catalog_version=None,
        exclude_folders=[],
        ID_col="NUMBER",
        field_col="field",
        ra_col="ALPHA_J2000",
        dec_col="DELTA_J2000",
        redshift_col="zbest",
        robust_col=None,
        mixed_cat=False,
        overall_field=None,
        radius_col="FLUX_RADIUS_F277W+F356W+F444W",
        eazy_template="fsps_larson",
        add_prospector=False,
        compact_plot=True,
        prospector_run_name="continuity_flex_dynesty",
        catalogue_flux_unit=u.MJy / u.sr,
        simulated_cat=False,
    ):
        path = Path(pipes_path)
        self.pipes_path = path
        self.galaxy_id = galaxy_id
        self.num_id = self.galaxy_id.split("_")[0]
        self.field = field
        self.field_col = field_col
        self.match_field = match_field
        self.match_version = match_version
        self.catalog_path = catalog_path
        self.spectra_path = spectra_path
        self.catalog_version = catalog_version
        self.redshift_col = redshift_col
        self.id_col = ID_col
        self.robust_col = robust_col
        self.eazy_template = eazy_template
        self.radius_col = radius_col
        self.catalogue_flux_unit = catalogue_flux_unit
        self.simulated_cat = simulated_cat
        self.compact_plot = compact_plot
        if overall_field is None:
            self.overall_field = field
        else:
            self.overall_field = overall_field
        self.ra_col = ra_col
        self.dec_col = dec_col
        # Set some environment variables for bagpipes
        os.environ["catalog_path"] = str(catalog_path)
        os.environ["spectra_path"] = str(spectra_path)
        os.environ["mixed_cat"] = str(mixed_cat)

        self.h5_paths = []
        self.fits = []
        # Raise errors if path is not a directory or does not exist
        if not path.exists():
            raise NotADirectoryError(f"Path does not exist\n{pipes_path}")
        if not path.is_dir():
            raise NotADirectoryError(f"Path is not a directory\n{pipes_path}")

        # Recursively search for h5 files in the posterior folder
        files = glob.glob(
            f"{str(path)}/posterior/**/{galaxy_id}.h5", recursive=True
        )

        # Filter out files that do not match the field or catalog version if requested
        exclude_folders = [
            f"{self.overall_field}/{i}" for i in exclude_folders
        ]
        exclude_folders.extend(
            [f"{self.overall_field}/plot_temp", f"{self.overall_field}/temp"]
        )

        for file in files:
            add = True
            if match_field:
                if field not in file:
                    add = False
            if match_version:
                if catalog_version not in file:
                    add = False
            # Ignore excluded folders (if you want to not plot a certain run)
            for folder in exclude_folders:
                folder_name = Path(f"{str(path)}/posterior/{folder}/")
                if Path(folder_name) in Path(file).parents:
                    print(f"Ignoring {file} as it is in {folder}")
                    add = False
            if "2023" in Path(file).parents:
                add = False

            if add:
                self.h5_paths.append(file)
        # Check we found some h5 files
        if len(self.h5_paths) == 0:
            raise ValueError(
                f"No h5 files found for {galaxy_id} in {pipes_path}"
            )
        print(
            f"Found {len(self.h5_paths)} h5 files for {galaxy_id} in {pipes_path}"
        )
        print(self.h5_paths)

        # Load the catalog

        try:
            self.catalog = Table.read(self.catalog_path, format="fits")

            try:
                mask = self.catalog[self.field_col] == self.field

            except KeyError:
                print("No field column found in catalog. Skipping")
                mask = True
            # Needs to be the right type to evalulate elementwise!
            num_id = int(self.galaxy_id.split("_")[0])

            self.row = Table(
                self.catalog[(self.catalog[self.id_col] == num_id) & mask]
            )

            if len(self.row) == 0:
                raise Exception(
                    f"No row found for {self.galaxy_id} in {self.catalog_path}"
                )
            elif len(self.row) > 1:
                raise Exception(
                    f"Multiple matches found for {self.galaxy_id} in {self.catalog_path}"
                )
        except FileNotFoundError:
            print(f"Catalog {catalog_path} not found.")
        except ValueError:
            print("No catalog path set. Skipping catalog load.")
            self.row = None

        # load the .h5 files
        for h5_file in self.h5_paths:
            # print('Doing ', h5_file)
            try:
                fit_obj = PipesFit(
                    self.galaxy_id,
                    self.field,
                    h5_file,
                    self.pipes_path,
                    self.row,
                    load_spectrum=False,
                    overall_field=self.overall_field,
                    field_col=self.field_col,
                    ID_col=self.id_col,
                    catalogue_flux_unit=self.catalogue_flux_unit,
                )
                self.fits.append(fit_obj)
            except ZeroDivisionError as e:
                print(e)
                pass

        if add_prospector:
            try:
                if type(prospector_run_name) is list:
                    for run_name in prospector_run_name:
                        self.add_prospector(run_name=run_name)
                else:
                    self.add_prospector(run_name=prospector_run_name)
            except FileNotFoundError as e:
                print(e)
                print(
                    f"Prospector file not found for {self.galaxy_id}. Skipping."
                )
        if self.compact_plot:
            self.pretty_plot_condensed()
        else:
            self.pretty_plot()

    def annotate_nearby(self, ax, size, pixel_scale):
        length = len(self.catalog)
        if length < 2000:
            print(
                "This catalogue seems short. If only robust galaxies are included, the nearby annotations won't work."
            )
        center = SkyCoord(
            self.row[self.ra_col][0] * u.degree,
            self.row[self.dec_col][0] * u.degree,
            unit="deg",
        )
        catalog_sky = SkyCoord(
            ra=self.catalog[self.ra_col] * u.degree,
            dec=self.catalog[self.dec_col] * u.degree,
        )
        for i in range(2, 10):
            idx, d2d, d3d = center.match_to_catalog_sky(
                catalog_sky, nthneighbor=i
            )

            # print(idx, d2d, d3d)
            if d3d != 0.0:
                max_sep = size * pixel_scale * u.arcsec
                if d2d[0] < max_sep:
                    match = self.catalog[idx]
                    wcs = self.cutout_wcs
                    coords = wcs.world_to_pixel(
                        SkyCoord(
                            match[self.ra_col] * u.degree,
                            match[self.dec_col] * u.degree,
                        )
                    )
                    color = (
                        "green"
                        if match[f"{self.robust_col}_{self.eazy_template}"]
                        else "white"
                    )
                    ax.text(
                        coords[0],
                        coords[1] - 10,
                        f"{match[self.id_col]}:z~{match[self.redshift_col+'_'+self.eazy_template]:.1f}",
                        color=color,
                        ha="center",
                        va="center",
                        fontsize="small",
                    )
                    radius = match[self.radius_col]
                    region = patches.Circle(
                        (coords[0], coords[1]),
                        radius,
                        fill=False,
                        linestyle="--",
                        lw=1,
                        color=color,
                        zorder=20,
                    )
                    ax.add_patch(region)

    def jaguar_cat(
        self,
        ax_photo,
        ax_z_pdf,
        ax_sfr,
        ax_metallicity,
        ax_mass_pdf,
        ax_ssfr,
        ax_dust,
    ):
        true_redshift = self.row["redshift"][0]
        true_mass = self.row["mStar"][0]
        true_sfr = self.row["SFR_10"][0]
        true_Z = self.row["metallicity"][0]
        true_dust = self.row["tauV_eff"][0]
        ax_z_pdf.set_ylim(ax_z_pdf.get_ylim()[0], ax_z_pdf.get_ylim()[1])
        ax_z_pdf.vlines(
            true_redshift,
            ax_z_pdf.get_ylim()[0],
            ax_z_pdf.get_ylim()[1],
            color="black",
            linestyle="--",
            lw=1,
            zorder=10,
        )

        ax_sfr.set_ylim(ax_sfr.get_ylim()[0], ax_sfr.get_ylim()[1])
        ax_sfr.vlines(
            true_sfr,
            ax_sfr.get_ylim()[0],
            ax_sfr.get_ylim()[1],
            color="black",
            linestyle="--",
            lw=1,
            zorder=10,
        )

        ax_metallicity.set_ylim(
            ax_metallicity.get_ylim()[0], ax_metallicity.get_ylim()[1]
        )
        ax_metallicity.vlines(
            true_Z,
            ax_metallicity.get_ylim()[0],
            ax_metallicity.get_ylim()[1],
            color="black",
            linestyle="--",
            lw=1,
            zorder=10,
        )

        ax_mass_pdf.set_ylim(
            ax_mass_pdf.get_ylim()[0], ax_mass_pdf.get_ylim()[1]
        )
        ax_mass_pdf.vlines(
            true_mass,
            ax_mass_pdf.get_ylim()[0],
            ax_mass_pdf.get_ylim()[1],
            color="black",
            linestyle="--",
            lw=1,
            zorder=10,
        )

        # ax_ssfr.set_ylim(ax_ssfr.get_ylim()[0], ax_ssfr.get_ylim()[1])
        # ax_ssfr.vlines(true_sfr/true_mass, ax_ssfr.get_ylim()[0], ax_ssfr.get_ylim()[1], color='black', linestyle='--', lw=1, zorder=10)

        ax_dust.set_ylim(ax_dust.get_ylim()[0], ax_dust.get_ylim()[1])
        ax_dust.vlines(
            true_dust,
            ax_dust.get_ylim()[0],
            ax_dust.get_ylim()[1],
            color="black",
            linestyle="--",
            lw=1,
            zorder=10,
        )

    def pretty_plot_condensed(
        self,
        cmap=flexoki_colors,
        wav_units=u.um,
        add_cutouts=True,
        add_sfh=True,
        flux_units=u.ABmag,
        max_bands=8,
    ):
        if type(cmap) is str:
            cmap = plt.get_cmap(cmap)
            colours = cmap(np.linspace(0, 1, len(self.fits)))
            if len(self.fits) == 1:
                colours = ["purple"]
        elif type(cmap) is list and len(cmap) == len(self.fits):
            colours = cmap
        elif type(cmap) is list and len(cmap) != len(self.fits):
            step = len(cmap) / len(self.fits)
            colours = []
            for i in range(len(self.fits)):
                index = int(i * step)
                colours.append(cmap[index])

        fig = plt.figure(layout="constrained", figsize=(8, 6))

        # Maybe generalize in future?
        found = False
        for fit in self.fits:
            try:
                bands = fit.bands
                if bands is not None:
                    bands = bands
                    found = True
            except:
                pass

        if not found:
            bands = None
            print(self.galaxy_id, self.field)
            print(self.h5_paths)
            raise Exception("No bands found. Cannot plot cutouts.")

        width = len(bands)
        height_bands = 1
        height_ratios = [2.5, 1.0]

        if bands is not None:
            if len(bands) > max_bands:
                width = max_bands
                height_bands = int(np.ceil(len(bands) / max_bands))
                height_ratios = [2.5 - height_bands / 4, 1.0]

        subfigs = fig.subfigures(
            2, 1, wspace=0.07, hspace=0.07, height_ratios=height_ratios
        )
        top_fig = subfigs[0]
        cutout_fig = subfigs[1]
        gs = top_fig.add_gridspec(2, 4)
        cutout_gs = cutout_fig.add_gridspec(height_bands, width)

        ax_photo = top_fig.add_subplot(gs[:, :3])
        ax_photo.tick_params(axis="both", which="major", direction="out")
        ax_z_pdf = top_fig.add_subplot(gs[0, 3:])
        ax_mass_pdf = top_fig.add_subplot(gs[1, 3:])
        ax_sfh = inset_axes(
            ax_photo,
            width="30%",
            height="30%",
            loc="upper left",
            bbox_to_anchor=(-0.013, 0.020, 1, 1),
            bbox_transform=ax_photo.transAxes,
        )  # bbox_to_anchor=(0.10,-0.005,1,1)

        """t = ax_photo.text(0.5, 0.5, 'Text')

        fonts = ['xx-small', 'x-small', 'small', 'medium', 'large',
                'x-large', 'xx-large', 'larger', 'smaller']

        for font in fonts:
            t.set_fontsize(font)
            print (font, round(t.get_fontsize(), 2))"""

        countdown = 0
        count = 0
        if add_cutouts:
            if bands is not None:
                for pos, band in tqdm(enumerate(bands), desc="Adding cutouts"):
                    if height_bands > 1:
                        cutout_ax = cutout_fig.add_subplot(
                            cutout_gs[pos // width, pos % width]
                        )
                    else:
                        cutout_ax = cutout_fig.add_subplot(cutout_gs[0, pos])

                    if self.plot_cutout(
                        cutout_ax,
                        band,
                        show_scale=0.5 if band == "F444W" else False,
                    ):
                        countdown += 1
                        if countdown == 0 or countdown % 5 == 0:
                            count = 0
                        else:
                            count += 2

                        color = "black"

                        # cutout_ax.set_xlabel(band, fontsize='medium', color=color, fontweight='bold')
                        # cutout_ax.set_aspect('equal', adjustable='box', anchor='N')
                        cutout_ax.set_ylabel(
                            band,
                            fontsize="medium",
                            color=color,
                            fontweight="bold",
                        )
                        cutout_ax.set_xticks([])
                        cutout_ax.set_yticks([])
                        try:
                            path_effects = [
                                pe.withStroke(linewidth=4, foreground="black")
                            ]
                            cutout_ax.annotate(
                                rf"{self.row[f'sigma_{band}'][0][0]:.1f}$\sigma$",
                                xy=(0.94, 0.82),
                                ha="right",
                                xycoords="axes fraction",
                                color="white",
                                path_effects=path_effects,
                            )
                        except:
                            pass
                    else:
                        cutout_ax.remove()

        # This searches for nearby galaxies in the catalog - FOR THIS TO MAKE SENSE IT NEEDS TO BE RUN ON THE UMERGED FULL CATALOG

        ax_photo.set_xlabel(
            f"Wavelength ({wav_units})", fontsize="medium", fontweight="bold"
        )
        ax_photo.set_ylabel(flux_units, fontweight="bold")
        # ax_photo.set_title(f'{self.field}:{self.galaxy_id.split("_")[0]}', fontsize='large', fontweight='bold')
        top_fig.suptitle(
            f'{self.field}:{self.galaxy_id.split("_")[0]}',
            fontsize="large",
            fontweight="bold",
        )
        fill_uncertainty = True if len(colours) else False
        fill_uncertainty = False
        for pos, fit in enumerate(self.fits):
            # Plot best fitting SED
            fit.plot_best_fit(
                ax_photo,
                colour=colours[pos],
                zorder=3,
                wav_units=wav_units,
                flux_units=flux_units,
                fill_uncertainty=fill_uncertainty,
            )
            # Plot best photometry
            fit.plot_best_photometry(
                ax_photo,
                colour=colours[pos],
                zorder=6,
                wav_units=wav_units,
                flux_units=flux_units,
            )

            if True:
                fit.plot_sfh(
                    ax_sfh,
                    colour=colours[pos],
                    add_zaxis=True if pos == 0 else False,
                    modify_ax=True if pos == 0 else False,
                    logify=True,
                    timescale="Myr",
                    plottype="lookback",
                    cosmo=FlatLambdaCDM(H0=70, Om0=0.3, Tcmb0=2.725),
                )

                # print('Adding posterior PDFs')
                try:
                    fit.plot_pdf(
                        ax_z_pdf,
                        "redshift",
                        colour=colours[pos],
                        fill_between=fill_uncertainty,
                    )
                except:  # Exclude the zfix ones
                    pass
                fit.plot_pdf(
                    ax_mass_pdf,
                    "stellar_mass",
                    colour=colours[pos],
                    fill_between=fill_uncertainty,
                )

        ax_mass_pdf.set_title(
            "Stellar Mass", fontsize="medium", fontweight="bold"
        )
        # if nothing on ax_z_pdf
        if len(ax_z_pdf.lines) == 0:
            ax_z_pdf.remove()
        else:
            ax_z_pdf.set_title(
                "Redshift", fontsize="medium", fontweight="bold"
            )
            ax_z_pdf.set_xlabel(
                r"Redshift", fontsize="medium", fontweight="bold"
            )
            ax_z_pdf.set_title("")

        ax_mass_pdf.set_xlabel(
            r"$ \mathrm{\bf Stellar \ Mass } \ (\log_{10}\mathrm{M}_{*}/\mathrm{M}_{\odot})$",
            fontsize="medium",
            fontweight="bold",
        )

        ax_mass_pdf.set_title("")
        # Plot observed photometry
        # logscale SFH
        ax_sfh.set_yscale("log")

        ax_sfh.yaxis.set_label_position("right")
        ax_sfh.yaxis.tick_right()

        ax_sfh.set_xlim(2, ax_sfh.get_xlim()[1])
        if ax_sfh.get_ylim()[1] <= 1e2:
            ax_sfh.set_ylim(1e-2, 3e2)
        ax_sfh.set_yticks([1e-2, 1e0, 1e2])
        ax_sfh.set_yticklabels(
            [0.01, 1, 100],
            path_effects=[pe.withStroke(linewidth=2, foreground="white")],
        )
        ax_sfh.yaxis.set_minor_locator(AutoMinorLocator(10))
        ax_sfh.yaxis.set_label_position("right")

        # ax_sfh.yaxis.set_major_formatter(ScalarFormatter())
        # ax_sfh.tick_params(axis='both', which='minor')
        # ax_sfh.yaxis.set_ticks([1e-1, 1e0, 1e1, 1e2])
        # ax_sfh.set_ylim(1e-2, 3e2)

        self.fits[0].plot_sed(
            ax_photo, colour="black", fcolour="black", zorder=10
        )

        # Add legend
        leg = ax_photo.legend(fontsize=7.8, frameon=False, loc="lower right")
        leg.set_zorder(15)

        fig.get_layout_engine().set(hspace=-3)

        if not Path(
            f"{bagpipes_dir}/pipes/plots/{self.overall_field}/"
        ).is_dir:
            os.mkdir(f"{bagpipes_dir}/pipes/plots/{self.overall_field}/")
        if not Path(f"{bagpipes_dir}/pipes/plots/merged/").is_dir:
            os.mkdir(f"{bagpipes_dir}/pipes/plots/merged/")
        try:
            fig.savefig(
                f"{bagpipes_dir}/pipes/plots/{self.overall_field}/{self.galaxy_id}_compact.png",
                dpi=200,
                bbox_inches="tight",
            )
            fig.savefig(
                f"{bagpipes_dir}/pipes/plots/merged/{self.galaxy_id}_compact.png",
                dpi=200,
                bbox_inches="tight",
            )
        except Exception as e:
            # Sometimes redshift axis breaks, this attempts to remove and resave
            ax_sfh.cla()
            print("Redshift axis broke. Trying again.")
            print(e)
            for pos, fit in enumerate(self.fits):
                fit.plot_sfh(ax_sfh, colour=colours[pos], add_zaxis=False)

            fig.savefig(
                f"{bagpipes_dir}/pipes/plots/{self.overall_field}/{self.galaxy_id}_compact.png",
                dpi=200,
                bbox_inches="tight",
            )
            fig.savefig(
                f"{bagpipes_dir}/pipes/plots/merged/{self.galaxy_id}_compact.png",
                dpi=200,
                bbox_inches="tight",
            )

    def pretty_plot(
        self,
        cmap=flexoki_colors,
        wav_units=u.um,
        add_rgb=True,
        add_posterior=True,
        add_cutouts=True,
        add_sfh=True,
        flux_units=u.ABmag,
        size=150,
        pixel_scale=0.03,
        blue=["F090W", "F115W", "F150W"],
        green=["F200W", "F277W"],
        red=["F356W", "F410M", "F444W"],
    ):
        # cmap='cmr.ember'
        # Get cmap to get colors for each fit
        if type(cmap) is str:
            cmap = plt.get_cmap(cmap)
            colours = cmap(np.linspace(0, 1, len(self.fits)))
            if len(self.fits) == 1:
                colours = ["purple"]
        elif type(cmap) is list and len(cmap) == len(self.fits):
            colours = cmap
        elif type(cmap) is list and len(cmap) != len(self.fits):
            step = len(cmap) / len(self.fits)
            colours = []
            for i in range(len(self.fits)):
                index = int(i * step)
                colours.append(cmap[index])

        fig = plt.figure()

        # Maybe generalize in future?
        found = False
        for fit in self.fits:
            try:
                bands = fit.bands
                if bands is not None:
                    bands = bands
                    found = True
            except:
                pass

        if not found:
            bands = None

        width = 14
        height = 5
        if not add_posterior:
            height -= 1
        if not add_cutouts:
            height -= 1
        gs = fig.add_gridspec(height, width)
        ax_photo = fig.add_subplot(gs[0:2, :12])
        ax_z_pdf = fig.add_subplot(gs[0, 12:])
        if add_posterior:
            ax_mass_pdf = fig.add_subplot(gs[1, 12:])

            ax_sfh = fig.add_subplot(gs[2, :4])
            ax_sfr = fig.add_subplot(gs[2, 4:6])
            ax_ssfr = fig.add_subplot(gs[2, 6:8])
            ax_metallicity = fig.add_subplot(gs[2, 8:10])
            ax_dust = fig.add_subplot(gs[2, 10:12])
            ax_nebular = fig.add_subplot(gs[2, 12:14])
        if add_rgb:
            ax_color = fig.add_subplot(gs[3:, 10:14])

        countdown = 0
        count = 0
        if add_cutouts:
            if bands is not None:
                for pos, band in tqdm(enumerate(bands), desc="Adding cutouts"):
                    cutout_ax = fig.add_subplot(
                        gs[
                            int(np.floor(3 + 0.2 * countdown)),
                            count : count + 2,
                        ]
                    )

                    if self.plot_cutout(cutout_ax, band):
                        countdown += 1
                        if countdown == 0 or countdown % 5 == 0:
                            count = 0
                        else:
                            count += 2

                        color = (
                            "blue"
                            if band in blue
                            else "green"
                            if band in green
                            else "red"
                            if band in red
                            else "black"
                        )

                        cutout_ax.set_xlabel(
                            band,
                            fontsize="medium",
                            color=color,
                            fontweight="bold",
                        )
                        cutout_ax.set_aspect("equal", adjustable="box")
                        cutout_ax.set_xticks([])
                        cutout_ax.set_yticks([])
                        try:
                            path_effects = [
                                pe.withStroke(linewidth=4, foreground="black")
                            ]
                            cutout_ax.annotate(
                                rf"{self.row[f'sigma_{band}'][0][0]:.1f}$\sigma$",
                                xy=(0.94, 0.85),
                                ha="right",
                                xycoords="axes fraction",
                                color="white",
                                path_effects=path_effects,
                            )
                        except:
                            pass
                    else:
                        cutout_ax.remove()

        try:
            if add_rgb:
                ax_color.set_aspect("equal", adjustable="box")
                ax_color.set_xticks([])
                ax_color.set_yticks([])
                added = self.plot_colour(
                    ax_color,
                    size=size,
                    pixel_scale=pixel_scale,
                    blue=blue,
                    green=green,
                    red=red,
                )
                if added:
                    print("Adding RGB")
                    ax_color.set_title("RGB", fontsize="medium")
                    self.annotate_nearby(ax_color, size, pixel_scale)
                else:
                    ax_color.remove()

        except Exception as e:
            print("Error adding RGB")
            print(e)
            ax_color.remove()

        # This searches for nearby galaxies in the catalog - FOR THIS TO MAKE SENSE IT NEEDS TO BE RUN ON THE UMERGED FULL CATALOG

        ax_photo.set_xlabel(
            f"Wavelength ({wav_units})", fontsize="medium", fontweight="bold"
        )
        ax_photo.set_ylabel(flux_units)
        ax_photo.set_title(
            f'{self.overall_field}:{self.galaxy_id.split("_")[0]}',
            fontsize="large",
        )

        fill_uncertainty = True if len(colours) else False
        for pos, fit in enumerate(self.fits):
            # Plot best fitting SED
            fit.plot_best_fit(
                ax_photo,
                colour=colours[pos],
                zorder=3,
                wav_units=wav_units,
                flux_units=flux_units,
                fill_uncertainty=fill_uncertainty,
            )
            # Plot best photometry
            fit.plot_best_photometry(
                ax_photo,
                colour=colours[pos],
                zorder=6,
                wav_units=wav_units,
                flux_units=flux_units,
            )

            if True:
                fit.plot_sfh(
                    ax_sfh,
                    colour=colours[pos],
                    add_zaxis=True if pos == 0 else False,
                    modify_ax=True if pos == 0 else False,
                    logify=False,
                    timescale="Myr",
                    plottype="absolute",
                )
            if add_posterior:
                print("Adding posterior PDFs")
                try:
                    fit.plot_pdf(
                        ax_z_pdf,
                        "redshift",
                        colour=colours[pos],
                        fill_between=fill_uncertainty,
                    )
                except:  # Exclude the zfix ones
                    pass
                fit.plot_pdf(
                    ax_mass_pdf,
                    "stellar_mass",
                    colour=colours[pos],
                    fill_between=fill_uncertainty,
                )
                fit.plot_pdf(
                    ax_metallicity,
                    "metallicity",
                    colour=colours[pos],
                    fill_between=fill_uncertainty,
                )
                fit.plot_pdf(
                    ax_dust,
                    "dust:Av",
                    colour=colours[pos],
                    fill_between=fill_uncertainty,
                )
                fit.plot_pdf(
                    ax_sfr,
                    "sfr",
                    colour=colours[pos],
                    fill_between=fill_uncertainty,
                )
                fit.plot_pdf(
                    ax_ssfr,
                    "ssfr",
                    colour=colours[pos],
                    fill_between=fill_uncertainty,
                )
                # This is broken for some reason - think
                fit.plot_pdf(
                    ax_nebular,
                    "nebular:logU",
                    colour=colours[pos],
                    fill_between=fill_uncertainty,
                )
                # fit.plot_pdf(ax_nebular, 'mass_weighted_age', colour=colours[pos], fill_between=fill_uncertainty)

            # Plot observed photometry
        # logscale SFH

        ax_sfh.set_yscale("log")

        self.fits[0].plot_sed(
            ax_photo, colour="black", fcolour="black", zorder=10
        )

        if self.simulated_cat:
            self.jaguar_cat(
                ax_photo,
                ax_z_pdf,
                ax_sfr,
                ax_metallicity,
                ax_mass_pdf,
                ax_ssfr,
                ax_dust,
            )
        # Add legend
        ax_photo.legend(fontsize="small", frameon=False)

        if not Path(
            f"{bagpipes_dir}/pipes/plots/{self.overall_field}/"
        ).is_dir:
            os.mkdir(f"{bagpipes_dir}/pipes/plots/{self.overall_field}/")
        try:
            fig.savefig(
                f"{bagpipes_dir}/pipes/plots/{self.overall_field}/{self.galaxy_id}.png",
                dpi=200,
                bbox_inches="tight",
            )
        except:
            # Sometimes redshift axis breaks, this attempts to remove and resave
            ax_sfh.cla()
            self.fits[0].plot_sfh(ax_sfh, colour=colours[pos], add_zaxis=False)
            fig.savefig(
                f"{bagpipes_dir}/pipes/plots/{self.overall_field}/{self.galaxy_id}.png",
                dpi=200,
                bbox_inches="tight",
            )

    def plot_colour(
        self,
        ax,
        ra_col="ALPHA_J2000",
        dec_col="DELTA_J2000",
        pixel_scale=0.03,
        size=150,
        blue=["F115W", "F150W"],
        green=["F200W", "F277W"],
        red=["F356W", "F444W"],
    ):
        path = {
            band: self.get_image_path(band, self.field)
            for band in blue + green + red
        }
        red_data, _ = combine_bands(red, path)
        green_data, _ = combine_bands(green, path)
        blue_data, wcs = combine_bands(blue, path)
        if (
            red_data is not None
            and green_data is not None
            and blue_data is not None
        ):
            ra_coord = self.row[ra_col] * u.deg
            dec_coord = self.row[dec_col] * u.deg

            sky_pos = SkyCoord(ra_coord, dec_coord)
            try:
                cutout_r = Cutout2D(red_data, sky_pos, wcs=wcs, size=size)
                cutout_g = Cutout2D(green_data, sky_pos, wcs=wcs, size=size)
                cutout_b = Cutout2D(blue_data, sky_pos, wcs=wcs, size=size)
                self.cutout_wcs = cutout_r.wcs
                # data_cutout = np.dstack([cutout_r.data, cutout_g.data, cutout_b.data])

                skip = False
            except (NoOverlapError, ValueError) as e:
                # cutout = None
                skip = True
                print(e)

            if not skip:
                # norm = ImageNormalize(data_cutout, interval=ManualInterval(1e-10, 1), stretch=LogStretch(a=0.1))
                rgb = make_lupton_rgb(
                    cutout_r.data,
                    cutout_g.data,
                    cutout_b.data,
                    Q=1,
                    stretch=0.05,
                )
                # with np.printoptions(threshold=np.inf):
                #    print(rgb)
                ax.imshow(rgb, origin="lower", interpolation="none")
                scalebar = AnchoredSizeBar(
                    ax.transData,
                    1 / pixel_scale,
                    '1"',
                    "lower right",
                    pad=0.3,
                    color="white",
                    frameon=False,
                    size_vertical=2,
                )
                ax.add_artist(scalebar)
                xpos = np.mean(ax.get_xlim())
                ypos = np.mean(ax.get_ylim())
                region = patches.Circle(
                    (xpos, ypos),
                    0.32 / pixel_scale,
                    fill=False,
                    linestyle="--",
                    lw=1,
                    color="white",
                )
                ax.add_patch(region)
                return True
        else:
            print("Data is None")
            print(red_data, green_data, blue_data)
        return False

    def get_image_path(self, band, field):
        # field_im_dirs = {field: f"{field}/mosaic_1084_wisptemp2"}
        # field_im_dirs = {key: f"/raid/scratch/data/jwst/{value}" for (key, value) in field_im_dirs.items()}
        field_dir = f"/raid/scratch/data/jwst/{field}"
        options = glob.glob(f"{field_dir}/*")

        if len(options) == 0:
            raise Exception(f"No directories found for {field}")
        for option in options:
            if "NIRCam" in option:
                options = glob.glob(f"{field_dir}/NIRCam/*")

                break
        priority = 5
        field_dir = None

        for option in options:
            if "mosaic_1084_wispnathan" in option:
                new_priority = 1
                if new_priority < priority:
                    priority = new_priority
                    field_dir = option

            elif "mosaic_1084_wisptemp2_whtfix" in option:
                new_priority = 2
                if new_priority < priority:
                    priority = new_priority
                    field_dir = option

            elif "mosaic_1084_wisptemp2" in option:
                new_priority = 3
                if new_priority < priority:
                    priority = new_priority
                    field_dir = option
                field_dir = option
        if field_dir is None:
            raise Exception(f"No image directory found for {field}")
        # print(field_dir)
        try:
            im_path = glob.glob(f"{field_dir}/*{band.lower()}*_i2d*.fits")
            if len(im_path) > 1:
                raise Exception(f"Multiple images found for {band}, {im_path}")
            im_path = im_path[0]
            im_ext = 1
        except:
            im_path = f"/raid/scratch/data/hst/{field}/ACS_WFC/30mas/ACS_WFC_{band}_{field}_drz.fits"
            im_ext = 0
        return im_path, im_ext

    def plot_cutout(
        self,
        ax,
        band,
        size=30,  # pixels
        radius=0.16,  # Aperture radius
        pixel_scale=0.03,  # arcsec/pixel
        ra_col="ALPHA_J2000",
        dec_col="DELTA_J2000",
        cmap="magma",
        zorder=10,
        linecolor="white",
        use_orig=False,
        origin="lower",
        show_scale=False,
    ):
        row = self.row
        field = self.field
        try:
            n_sig_detect = self.row[f"sigma_{band}"][0][0]
        except KeyError:
            n_sig_detect = 5
            pass

        im_path, im_ext = self.get_image_path(band, field)

        try:
            # print(f'Looking for {im_path}')
            fits_image = fits.open(im_path)
            found_image = True

        except (FileNotFoundError, ValueError):
            print("Image file not found.")
            print(f"path was {im_path}, band was {band}")
            found_image = False

        if found_image:
            # fits.info(path)
            header = fits_image[im_ext].header
            data = np.array(fits_image[im_ext].data)
            wcs = WCS(header)

            # Get position of galaxy

            ra_coord = row[ra_col] * u.deg
            dec_coord = row[dec_col] * u.deg

            radius_pix = radius / pixel_scale
            sky_pos = SkyCoord(ra_coord, dec_coord)

            try:
                cutout = Cutout2D(data, sky_pos, wcs=wcs, size=size)
                skip = False
            except (NoOverlapError, ValueError):
                cutout = None
                skip = True
                #
            if not skip:
                wcs = cutout.wcs
                data_cutout = cutout.data
                # Set top value based on central 10x10 pixel region
                top = np.max(data_cutout[10:20, 10:20])

                bottom_val = top / 10**5
                stretch = LogStretch(a=0.1)

                if n_sig_detect <= 15:
                    bottom_val = top / 10**3
                if n_sig_detect < 8:
                    bottom_val = top / 100000
                    stretch = LinearStretch()

                norm = ImageNormalize(
                    data_cutout,
                    interval=ManualInterval(bottom_val, top),
                    clip=True,
                    stretch=stretch,
                )

                ax.imshow(
                    data_cutout,
                    norm=norm if not use_orig else None,
                    cmap=cmap,
                    origin=origin,
                )

                xpos = np.mean(ax.get_xlim())
                ypos = np.mean(ax.get_ylim())
                region = patches.Circle(
                    (xpos, ypos),
                    radius_pix,
                    fill=False,
                    linestyle="--",
                    lw=1,
                    color=linecolor,
                    zorder=zorder,
                )
                ax.add_patch(region)

                if show_scale:
                    scalebar = AnchoredSizeBar(
                        ax.transData,
                        0.5 / pixel_scale,
                        '0.5"',
                        "lower right",
                        pad=0.1,
                        color="white",
                        frameon=False,
                        size_vertical=1,
                        label_top=True,
                        # fontproperties={'size': 'small'}
                    )
                    ax.add_artist(scalebar)

                return True
            else:
                return False

    def add_prospector(
        self, run_name="continuity_flex_dynesty", prospect_dir=prospector_dir
    ):
        path = f"{prospect_dir}/{self.field}/{self.num_id}_{run_name}.h5"
        try:
            from Plotspector import Plotspector

            fit_obj = Plotspector(path)
            self.fits.append(fit_obj)
        except (OSError, FileNotFoundError) as e:
            print(e)
            print(f"Prospector file not found for {self.galaxy_id}. Skipping.")


def main(id, field):
    if field == "CEERS":
        catalog_path = f"/raid/scratch/work/austind/GALFIND_WORK/Catalogues/v9/ACS_WFC+NIRCam/Combined/{field}_MASTER_Sel-f277W+f356W+f444W_v9_loc_depth_masked_10pc_EAZY_matched_selection.fits"

    if field in [
        "JADES-Deep-GS",
        "NEP-1",
        "NEP-2",
        "NEP-3",
        "NEP-4",
        "NGDEEP",
        "MACS-0416",
    ] or field.startswith("CEERSP"):
        catalog_path = f"/raid/scratch/work/austind/GALFIND_WORK/Catalogues/v9/ACS_WFC+NIRCam/{field}/{field}_MASTER_Sel-f277W+f356W+f444W_v9_loc_depth_masked_10pc_EAZY_matched_selection.fits"
    # For CEERS
    if field in ["SMACS-0723", "El-Gordo", "CLIO", "GLASS"]:
        catalog_path = f"/raid/scratch/work/austind/GALFIND_WORK/Catalogues/v9/NIRCam/{field}/{field}_MASTER_Sel-f277W+f356W+f444W_v9_loc_depth_masked_10pc_EAZY_matched_selection.fits"

    PlotPipes(
        f"{id}_{field}",
        field,
        catalog_path=catalog_path,
        overall_field="CEERS" if field.startswith("CEERSP") else None,
        robust_col="final_sample_highz",
        eazy_template="fsps_larson",
        simulated_cat=False,
        add_prospector=False,
        prospector_run_name=[
            "continuity_bursty_dynesty_zgauss",
            "continuity_bursty_dynesty_hot_imf_zgauss",
            "delayed_dynesty_zgauss",
            "delayed_dynesty_hot_imf_zgauss",
        ],
        exclude_folders=[
            "sfh_continuity_bursty_dust_Cal_uniform_Z_log_10_age_log_10_bpass_fesc_zgauss",
            "sfh_continuity_bursty_dust_Cal_uniform_Z_log_10_age_log_10_bpass_fesc_zfix",
            "sfh_continuity_bursty_dust_Cal_uniform_Z_log_10_age_log_101.0_zfix",
            "sfh_continuity_bursty_dust_Cal_uniform_Z_log_10_age_log_10_bpass_fesc0.0_zfix",
            "sfh_continuity_dust_Cal_uniform_Z_log_10_age_log_10_zfix",
            "sfh_lognorm_dust_Cal_uniform_Z_log_10_age_log_10_zfix",
            "sfh_continuity_bursty_dust_Cal_uniform_Z_log_10_age_log_10_zgauss",
            "sfh_lognorm_dust_Cal_uniform_Z_log_10_age_uniform_zgauss",
            "sfh_lognorm_dust_CF0_uniform_Z_log_10_age_log_10_zgauss",
            "sfh_continuity_bursty_dust_Cal_uniform_Z_log_10_age_log_10_bpass_fesc_zfix"
            "sfh_lognorm_dust_Cal_uniform_Z_log_10_age_uniform_zgauss",
            "sfh_lognorm_dust_CF0_uniform_Z_log_10_age_log_10_zgauss",
            "sfh_continuity_bursty_blue_dust_Cal_log_10_Z_log_10_age_log_10_bpass_fesc1.0_zgauss",
            "temp_austind",
            "2023-12-23 14:14:55.671336",
            "2024-01-14 13:06:01.975786",
            "sfh_continuity_bursty_dust_Cal_uniform_Z_log_10_age_log_10_bpass_fesc_Lya_zfix",
            "sfh_continuity_bursty_blue_dust_Cal_log_10_Z_log_10_age_log_10_bpass_fesc0.0_zfix",
            "sfh_continuity_bursty_blue_dust_Cal_log_10_Z_log_10_age_log_10_bpass_fesc_zgauss",
            "sfh_lognorm_dust_Cal_log_10_Z_log_10_age_log_10_bpass_fesc_zgauss",
            "sfh_continuity_bursty_blue_dust_Cal_log_10_Z_log_10_age_log_10_bpass_fesc0.0_zgauss",
            "sfh_continuity_bursty_blue_dust_Cal_log_10_Z_log_10_age_log_10_bpass_fesc_zfix",
            "sfh_continuity_dust_Cal_uniform_Z_log_10_age_log_10_zfix",
            "sfh_lognorm_dust_Cal_uniform_Z_log_10_age_log_10_zfix",
            "sfh_continuity_bursty_blue_dust_Cal_log_10_Z_log_10_age_log_10_bpass_fesc0.0_zfix",
            "sfh_continuity_bursty_dust_Cal_uniform_Z_log_10_age_log_10_bpass_fesc0.0_zfix",
            "sfh_continuity_bursty_dust_Cal_uniform_Z_log_10_age_log_10_bpass_fesc1.0_zfix",
            "sfh_continuity_bursty_dust_Cal_uniform_Z_log_10_age_log_10_bpass_fesc_zgauss",
            "sfh_continuity_bursty_dust_Cal_uniform_Z_log_10_age_log_10_bpass?_fesc0.0_zfix",
            "2023-12-23 17:39:08.493408",
            "sfh_lognorm_dust_Cal_uniform_Z_log_10_age_log_10_zgauss_old",
            "sfh_continuity_bursty_dust_Cal_uniform_Z_log_10_age_log_10_bpass_fesc1.0_zfix",
            "sfh_continuity_dust_Cal_uniform_Z_log_10_age_log_10_bpass_fesc_zfix",
            "sfh_lognorm_dust_Cal_log_10_Z_log_10_age_log_10_bpass_fesc0.0_zgauss",  # just a BPASSS run? maybe useful.
            "temp_sfh_continuity_bursty_blue_dust_Cal_log_10_Z_log_10_age_log_10_bpass_fesc_zfix",
            "sfh_continuity_bursty_dust_Cal_uniform_Z_log_10_age_log_101.0_zfix",
        ],
        compact_plot=True,
    )


if __name__ == "__main__":
    "/nvme/scratch/work/tharvey/EXPANSE/pipes/posterior/photoz_cnst_zfix/JOF_psfmatched/005_z010p000_00_104_mock/TOTAL_BIN.h5"
