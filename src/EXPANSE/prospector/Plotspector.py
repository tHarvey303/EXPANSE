import sys

import astropy.units as u

# import deepdish as dd
import h5py as h5py
import matplotlib.pyplot as plt
from astropy.cosmology import FlatLambdaCDM, z_at_value
from astropy.cosmology.core import CosmologyError
from astropy.table import Table
from matplotlib.ticker import ScalarFormatter
from numpy.random import choice
from prospect.models.transforms import (
    logsfr_ratios_to_agebins,
    logsfr_ratios_to_masses,
    logsfr_ratios_to_masses_flex,
    logsfr_ratios_to_sfrs,
    tage_from_tuniv,
    zfrac_to_masses,
    zfrac_to_sfr,
)
from prospect.plotting import FigureMaker
from prospect.plotting.sed import to_nufnu
from prospect.plotting.sfh import (
    nonpar_mwa,
    nonpar_recent_sfr,
    parametric_mwa,
    parametric_sfr,
)
from scipy.special import gamma, gammainc
from sklearn.neighbors import KernelDensity
from tqdm import tqdm

try:
    plt.style.use("/nvme/scratch/work/tharvey/scripts/paper.mplstyle")
except (FileNotFoundError, OSError):
    pass
import os

import numpy as np
import tables
from .utils import weighted_quantile

maggies_to_mag = lambda maggies: -2.5 * np.log10(maggies * 3631) + 8.90

# TODO: Support plotting more SFH
# TODO: Support plotting dust attenuation law
# TODO: Plot priors on distributions and show calculated relative entropy - https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.entropy.html
# TODO: Support cumulative SFH
# TODO: Support plotting spectra/zoom in spectral region
# TODO: Support plot maximum likelihood models vs draws
# TODO: On spectra, show smoothing/resolution


class Plotspector(FigureMaker):
    def save_updated_h5(self, data_to_add, name_to_add):
        tables.file._open_files.close_all()
        if type(data_to_add) == list:
            data_to_add = np.array(data_to_add)
        with h5py.File(self.results_file, "r+") as f:
            try:
                grp = f.create_group("extra_data")
            except ValueError:
                grp = f["extra_data"]
            try:
                del grp[name_to_add]
            except KeyError:
                pass

            data = grp.create_dataset(name_to_add, data=data_to_add)
            print("Added data to .h5")

    def plot_sed_new(
        self,
        sedax=None,
        fig=None,
        residax=None,
        nufnu=False,
        microns=True,
        normalize=False,
        logify=False,
        plot_type="all",
        modify_ax=True,
        label_phot="auto",
        label_spec="auto",
        color="black",
    ):
        imf_type = self.model.init_config["imf_type"]["init"]
        if imf_type != 5 and self.result["run_params"]["use_hot_imf"]:
            print("You probably don't want to use me as I was actually run with a Kroupa IMF")
            return False

            # This is model config.
        # print(model.init_config)
        """A very basic plot of the observed photometry and the best fit
        photometry and spectrum.
        """
        # --- Data ---
        pmask = self.obs["phot_mask"]
        ophot, ounc = self.obs["maggies"][pmask], self.obs["maggies_unc"][pmask]
        owave = np.array([f.wave_effective for f in self.obs["filters"]])[pmask]
        phot_width = np.array([f.effective_width for f in self.obs["filters"]])[pmask]
        if nufnu:
            _, ophot = to_nufnu(owave, ophot, microns=microns)
            owave, ounc = to_nufnu(owave, ounc, microns=microns)
        if normalize:
            renorm = 1 / np.mean(ophot)
        else:
            renorm = 1.0

        # models

        if self.sps is None:
            self.build_sps()

        # --- best sample ---
        xbest = self.result["chain"][self.ind_best, :]
        blob = self.model.predict(xbest, obs=self.obs, sps=self.sps)
        self.spec_best, self.phot_best, self.mfrac_best = blob

        pwave, phot_best = self.obs["phot_wave"][pmask], self.phot_best[pmask]

        spec_best = self.spec_best
        swave = self.obs.get("wavelength", None)
        if swave is None:
            if "zred" in self.model.free_params:
                zred = self.chain["zred"][self.ind_best]
            else:
                zred = self.model.params["zred"]
            swave = self.sps.wavelengths * (1 + zred)
        if nufnu:
            swave, spec_best = to_nufnu(swave, spec_best, microns=microns)
            pwave, phot_best = to_nufnu(pwave, phot_best, microns=microns)
        if microns and not nufnu:
            swave = swave / 10**4
            pwave = pwave / 10**4
            owave = owave / 10**4

        # plot SED
        # Added
        self.styles()
        if plot_type == "all" or plot_type == "fitted_photometry":
            sedax.plot(
                pwave,
                maggies_to_mag(phot_best * renorm),
                marker="o",
                linestyle="",
                color=color,  # **self.pkwargs,
                label=r"Best-fit photometry" if label_phot == "auto" else label_phot,
                zorder=3,
                alpha=0.7,
            )
        if plot_type == "all" or plot_type == "fitted_spectrum":
            sedax.plot(
                swave,
                maggies_to_mag(spec_best * renorm),
                color=color,
                lw=1,
                linestyle="dashdot",  # **self.lkwargs
                label=r"Best-fit spectrum" if label_spec == "auto" else label_spec,
                zorder=6,
            )
        # sedax.plot(owave, ophot * renorm, **self.dkwargs)

        # plot residuals
        if residax is not None:
            chi_phot = (ophot - phot_best) / ounc
            residax.plot(owave, chi_phot, **self.dkwargs)

        self.chi2 = np.sum(((ophot - phot_best) ** 2) / ounc**2)
        if sedax == None:
            fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(10, 5), facecolor="w", edgecolor="k")
            [ax.tick_params(axis="both", which="major", direction="out") for ax in axs]
        else:
            ax = sedax

        mag_err_low = np.abs(2.5 * np.log10(ophot / (ophot - ounc)))
        # If it fails calculate smaller error
        mag_err_up = np.abs(2.5 * np.log10(1 + (ounc / ophot)))
        if plot_type == "all" or plot_type == "photometry":
            ax.errorbar(
                owave,
                maggies_to_mag(ophot),
                yerr=[mag_err_low, mag_err_up],
                label="Observed photometry",
                marker="o",
                markersize=10,
                alpha=0.8,
                ls="",
                lw=3,
                ecolor="tomato",
                markerfacecolor="none",
                markeredgecolor="tomato",
                markeredgewidth=3,
            )

        # marker='s', markersize=10, alpha=0.8, ls='', lw=3, markerfacecolor='none', markeredgecolor='slateblue', markeredgewidth=3
        """
        if output != None or result != None:
            chi2 = np.sum((pphot-obs['maggies'])**2/obs['maggies_unc']**2)/(len(pphot)-1)
            
            model_out = np.stack((wspec, pspec), axis=1)
        """
        # Prettify
        xmin, xmax = np.min(pwave) * 0.8, np.max(pwave) / 0.8

        omask = np.array([True if xmin < i < xmax else False for i in owave])

        ymax = np.max(ophot[omask]) * 1.5
        ymin = np.min(ophot[omask]) / 1.1

        if ymin < 1e-12:
            ymin = 1e-12

        if modify_ax:
            ax.set_xlabel(r"Wavelength ($\mu m$)")
            ax.set_ylabel("AB Mag")
            ax.set_ylim([maggies_to_mag(ymin), maggies_to_mag(ymax)])
            ax.legend(loc="best", fontsize=10)

        # mag_to_maggies = lambda mag: 3631 * 10**((mag - 8.90 )/-2.5)

        # ax.set_xlim([xmin, xmax])

        if logify:
            ax.set_yscale("log")

        # secax = ax.secondary_yaxis('left', functions=(maggies_to_mag, mag_to_maggies))
        # secax.set_ylabel('AB Mag')
        # fig.canvas.draw()
        # secax.yaxis.set_major_formatter(ScalarFormatter())

    def nonpar_recent_sfr_flex(
        self,
        logmass,
        logsfr_ratios,
        logsfr_ratios_old,
        logsfr_ratios_young,
        agebins,
        sfr_period=0.01,
    ):
        """vectorized"""
        masses = [
            logsfr_ratios_to_masses_flex(
                np.squeeze(logm),
                np.squeeze(sr),
                logsfr_ratio_young,
                logsfr_ratio_old,
                agebins=agebin,
            )
            for logm, sr, logsfr_ratio_young, logsfr_ratio_old, agebin in zip(
                logmass, logsfr_ratios, logsfr_ratios_young, logsfr_ratios_old, agebins
            )
        ]
        masses = np.array(masses)

        ages = 10 ** (agebins - 9)
        # fractional coverage of the bin by the sfr period
        ft = np.clip((sfr_period - ages[:, :, 0]) / (ages[:, :, 1] - ages[:, :, 0]), 0.0, 1)
        mformed = (ft * masses).sum(axis=-1)

        return mformed / (sfr_period * 1e9)

    def nonpar_mwa_flex(
        self, logmass, logsfr_ratios, logsfr_ratios_old, logsfr_ratios_young, agebins
    ):
        masses = [
            logsfr_ratios_to_masses_flex(
                np.squeeze(logm),
                np.squeeze(sr),
                logsfr_ratio_young,
                logsfr_ratio_old,
                agebins=agebin,
            )
            for logm, sr, logsfr_ratio_young, logsfr_ratio_old, agebin in zip(
                logmass, logsfr_ratios, logsfr_ratios_young, logsfr_ratios_old, agebins
            )
        ]
        masses = np.array(masses)
        dtsqs = []
        for agebin in agebins:
            dt = 10 ** agebin[:, 1] - 10 ** agebin[:, 0]
            sfrs = masses / dt
            ages = 10 ** (agebin)

            # print(np.shape(ages), np.shape(sfrs), np.shape(logmass))
            dtsq = (ages[:, 1] ** 2 - ages[:, 0] ** 2) / 2
            dtsqs.append(dtsq)

        mwa = [(dtsq_i * sfr).sum() / 10**logm for dtsq_i, sfr, logm in zip(dtsqs, sfrs, logmass)]

        return np.array(mwa) / 1e9

        dt = 10 ** agebins[:, :, 1] - 10 ** agebins[:, :, 0]

        sfrs = masses / dt

        ages = 10 ** (agebins)

        # print(np.shape(ages), np.shape(sfrs), np.shape(logmass))
        dtsq = (ages[:, :, 1] ** 2 - ages[:, :, 0] ** 2) / 2

        mwa = [(dtsq * sfr).sum() / 10**logm for sfr, logm in zip(sfrs, logmass)]

        return np.array(mwa) / 1e9

    def plot_sfh(
        self,
        ax=None,
        logify=True,
        timescale="Myr",
        plottype="lookback",
        colour="#1f77b4",
        modify_ax=True,
        add_zaxis=True,
        show_lines=False,
        cosmo=FlatLambdaCDM(H0=70, Om0=0.3, Tcmb0=2.725),
        **kwargs,
    ):
        params = self.result["run_params"]
        # for parametric  For the nonparametric sfhs you can use prospect.models.transforms.logsfr_ratios_to_sfrs and then just read off the SFR of the most recent bin
        self.id = params["OBJID"]
        # Need to change this when fitting z

        self.field = params["field"]
        self.sfh_model = params["sfh_model"]
        self.fit_type = params["fit_type"]
        # This says what the parameters are
        bestfit = self.result["bestfit"]
        labels = np.array(self.result["theta_labels"])
        theta_best = self.result["chain"][self.ind_best, :]

        imf_type = self.model.init_config["imf_type"]["init"]
        if imf_type != 5 and self.result["run_params"]["use_hot_imf"]:
            print("You probably don't want to use me as I was actually run with a Kroupa IMF")
            return False

        try:
            redshift = self.chain["zred"]

            self.z_16, self.redshift, self.z_84 = weighted_quantile(
                redshift, [0.16, 0.5, 0.84], self.weights
            )

        except:
            self.redshift = self.model.init_config["zred"]["init"]
            self.z_16, self.z_84 = 0, 0

        """if self.sfh_model in ['continuity_flex']:
            print(labels)"""

        if self.sfh_model in [
            "continuity",
            "continuity_bursty",
            "non_parametric",
            "continuity_flex",
            "dirichlet",
            "alpha",
        ]:
            if self.sfh_model in [
                "continuity",
                "continuity_bursty",
                "non_parametric",
                "continuity_flex",
            ]:
                flex = False
                if self.sfh_model == "continuity_flex":
                    flex = True
                # These are probably wrong
                # need to use psb_logsfr_ratios_to_agebins,logsfr_ratios_to_masses_flex,in the right places

                # sfr_names = [i for i in labels if i.startswith('logsfr_ratios')]
                # sfr_names_mask = [True if i.startswith('logsfr_ratios') else False for i in labels ]

                logmass = self.chain["logmass"]

                self.logmass_best_low, self.logmass_best, self.logmass_best_high = (
                    weighted_quantile(np.ndarray.flatten(logmass), [0.16, 0.50, 0.84], self.weights)
                )

                logsfr_ratios = self.chain["logsfr_ratios"]

                sfrs = []
                masses = []
                agebins_init = np.array(self.model.init_config["agebins"]["init"])
                # might be able to replace this now
                if flex:
                    agebins = []

                    # Need to iterate over all draws in chain??
                    logsfr_ratios_old = self.chain["logsfr_ratio_old"]
                    logsfr_ratios_young = self.chain["logsfr_ratio_young"]
                    for logsfr_ratio, lm, logsfr_ratio_old, logsfr_ratio_young in zip(
                        logsfr_ratios, logmass, logsfr_ratios_old, logsfr_ratios_young
                    ):
                        agebin = logsfr_ratios_to_agebins(
                            logsfr_ratios=logsfr_ratio, agebins=agebins_init
                        )
                        mass = logsfr_ratios_to_masses_flex(
                            logmass=lm,
                            logsfr_ratios=logsfr_ratio,
                            logsfr_ratio_old=logsfr_ratio_old,
                            logsfr_ratio_young=logsfr_ratio_young,
                            agebins=agebins_init,
                        )
                        dt = 10 ** agebin[:, 1] - 10 ** agebin[:, 0]
                        sfr = mass / dt
                        sfrs.append(sfr)
                        masses.append(mass)
                        agebins.append(agebin)

                    agebins = np.array(agebins)

                    # mass = logsfr_ratios_to_masses_flex(logmass = logmass, logsfr_ratios = logsfr_ratios, logsfr_ratio_old = logsfr_ratios_old, logsfr_ratio_young = logsfr_ratios_young, agebins=agebins_init)

                    # Unclear if this is correct
                    # logsfr_ratios =  np.concatenate([logsfr_ratios_young, logsfr_ratios, logsfr_ratios_old], axis=1)
                # agebins = logsfr_ratios_to_agebins(logsfr_ratios=logsfr_ratios, agebins=agebins_init)

                else:
                    agebins = np.array(self.model.init_config["agebins"]["init"])

                    for logsfr_ratio, lm in zip(logsfr_ratios, logmass):
                        sfrs.append(
                            logsfr_ratios_to_sfrs(
                                logsfr_ratios=np.squeeze(logsfr_ratio),
                                agebins=agebins,
                                logmass=np.squeeze(lm),
                            )
                        )
                        masses.append(
                            logsfr_ratios_to_masses(
                                logsfr_ratios=np.squeeze(logsfr_ratio),
                                agebins=np.squeeze(agebins),
                                logmass=np.squeeze(lm),
                            )
                        )

                sfrs = np.array(sfrs).T
                masses = np.array(masses).T

                low_sfrs, median_sfrs, high_sfrs = [], [], []
                for bin in sfrs:
                    sfr_16, sfr_50, sfr_84 = weighted_quantile(bin, [0.16, 0.5, 0.84], self.weights)
                    low_sfrs.append(sfr_16)
                    median_sfrs.append(sfr_50)
                    high_sfrs.append(sfr_84)

                # for bin in masses:
                #    mass_16, mass_50, mass_84 = weighted_quantile(bin, [0.16, 0.5, 0.84], self.weights)

                # WARNING nonpar_recent_sfr and mwa_recent_sfr are not suitable for continuity_flex since they use
                # logsfr_ratios_to_sfrs instead of logsfr_ratios_to_masses_flex!!
                # Can probably just write own like done below for alpha
                if not flex:
                    sfr_10myr = nonpar_recent_sfr(logmass, logsfr_ratios, agebins, sfr_period=0.01)
                    self.sfr_10myr = weighted_quantile(sfr_10myr, [0.16, 0.5, 0.84], self.weights)

                    self.sfr_10myr_dist = sfr_10myr
                    sfr_100myr = nonpar_recent_sfr(logmass, logsfr_ratios, agebins, sfr_period=0.1)
                    self.sfr_100myr_dist = sfr_100myr
                    self.sfr_100myr = weighted_quantile(sfr_100myr, [0.16, 0.5, 0.84], self.weights)

                    mwa = np.ndarray.flatten(nonpar_mwa(logmass, logsfr_ratios, agebins))
                    self.mwa_dist = mwa
                    # in myr now
                    self.mwa = weighted_quantile(mwa, [0.16, 0.5, 0.84], self.weights) * 10**3

                else:
                    self.sfr_10myr_dist = self.nonpar_recent_sfr_flex(
                        logmass,
                        logsfr_ratios,
                        logsfr_ratios_old,
                        logsfr_ratios_young,
                        agebins,
                        sfr_period=0.01,
                    )
                    self.sfr_10myr = weighted_quantile(
                        self.sfr_10myr_dist, [0.16, 0.5, 0.84], self.weights
                    )
                    # sfr_10myr_dist.append(sfr_10myr)

                    self.sfr_100myr_dist = self.nonpar_recent_sfr_flex(
                        logmass,
                        logsfr_ratios,
                        logsfr_ratios_old,
                        logsfr_ratios_young,
                        agebins,
                        sfr_period=0.1,
                    )
                    self.sfr_100myr = weighted_quantile(
                        self.sfr_100myr_dist, [0.16, 0.5, 0.84], self.weights
                    )
                    # sfr_100myr_dist.append(sfr_100myr)

                    self.mwa_dist = np.ndarray.flatten(
                        self.nonpar_mwa_flex(
                            logmass, logsfr_ratios, logsfr_ratios_old, logsfr_ratios_young, agebins
                        )
                    )

                    self.mwa = (
                        weighted_quantile(self.mwa_dist, [0.16, 0.5, 0.84], self.weights) * 10**3
                    )

                    arrays, larray, harray = [], [], []
                    for time_col in np.transpose(agebins, (1, 2, 0)):
                        first = weighted_quantile(
                            time_col[0], [0.16, 0.5, 0.84], sample_weight=self.weights
                        )
                        second = weighted_quantile(
                            time_col[1], [0.16, 0.5, 0.84], sample_weight=self.weights
                        )
                        median = [first[1], second[1]]
                        low = [first[0], second[0]]
                        high = [first[2], second[2]]

                        arrays.append(median)
                        larray.append(low)
                        harray.append(high)
                    # self.agebins_dist = weighted_quantile(agebins, [0.16, 0.5, 0.84], sample_weight=self.weights, axis=0)
                    agebins = np.array(arrays)
                    agebins_low = np.array(larray)
                    agebins_high = np.array(harray)

            elif self.sfh_model in ["dirichlet", "alpha"]:
                agebins = np.array(self.model.init_config["agebins"]["init"])

                """z_frac = theta_best[labels == 'z_frac']
                total_mass  = theta_best[labels == 'mass']
                print(z_frac, total_mass)
                """
                # names = self.chain.dtype.names
                z_fracs = self.chain["z_fraction"]

                logmass = np.log10(self.chain["total_mass"])
                #

                sfrs = []
                masses = []
                for z_frac, total_mass in zip(z_fracs, logmass):
                    # print(z_frac, total_mass, agebins)
                    sfrs.append(zfrac_to_sfr(10**total_mass, z_frac, agebins))
                    masses.append(zfrac_to_masses(10**total_mass, z_frac, agebins))

                sfrs = np.array(sfrs).T
                masses = np.array(masses).T

                self.logmass_best_low, self.logmass_best, self.logmass_best_high = (
                    weighted_quantile(np.ndarray.flatten(logmass), [0.16, 0.50, 0.84], self.weights)
                )

                low_sfrs, median_sfrs, high_sfrs = [], [], []
                for bin in sfrs:
                    sfr_16, sfr_50, sfr_84 = weighted_quantile(bin, [0.16, 0.5, 0.84], self.weights)
                    low_sfrs.append(sfr_16)
                    median_sfrs.append(sfr_50)
                    high_sfrs.append(sfr_84)

                # for bin in masses:
                #    mass_16, mass_50, mass_84 = weighted_quantile(bin, [0.16, 0.5, 0.84], self.weights)

                # Adapted from prospect.plotting.sfh

                ages = 10 ** (agebins - 9)
                mass = 10**logmass
                # fractional coverage of the bin by the sfr period
                ft = lambda sfr_timescale: np.clip(
                    (sfr_timescale - ages[:, 0]) / (ages[:, 1] - ages[:, 0]), 0.0, 1
                )

                sfr_period = 0.01
                self.sfr_10myr_dist = (ft(sfr_period) * masses.T).sum(axis=-1) / (sfr_period * 1e9)
                self.sfr_10myr = weighted_quantile(
                    self.sfr_10myr_dist, [0.16, 0.5, 0.84], self.weights
                )

                sfr_period = 0.1
                self.sfr_100myr_dist = (ft(sfr_period) * masses.T).sum(axis=-1) / (sfr_period * 1e9)
                self.sfr_100myr = weighted_quantile(
                    self.sfr_100myr_dist, [0.16, 0.5, 0.84], self.weights
                )

                # Adapted from prospect.plotting.sfh
                ages = 10 ** (agebins)

                dtsq = (ages[:, 1] ** 2 - ages[:, 0] ** 2) / 2
                mwa = [(dtsq * sfr).sum() / 10**logm for sfr, logm in zip(sfrs.T, logmass)]

                self.mwa_dist = np.array(mwa) / 1e6  # into Myr
                # print(self.mwa_dist, np.shape(self.mwa_dist))
                self.mwa = weighted_quantile(
                    np.ndarray.flatten(self.mwa_dist), [0.16, 0.5, 0.84], self.weights
                )

                # mid_bins = np.array([np.mean(10**np.array(i)) for i in agebins])/10**6

            if ax is None:
                fig, ax = plt.subplots(
                    nrows=2,
                    ncols=1,
                    sharey=False,
                    sharex=True,
                    figsize=(10, 5),
                    facecolor="w",
                    edgecolor="k",
                )
                [a.tick_params(axis="both", which="major", direction="out") for a in ax]
                ax = ax[0]
            else:
                fig = None

            cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Tcmb0=2.725)

            time_convert = lambda lookback_time: z_at_value(
                cosmo.lookback_time,
                cosmo.lookback_time(self.redshift) + lookback_time * 10 ** (-3) * u.Gyr,
            ).value

            z_convert = lambda z_plot: (
                cosmo.lookback_time(self.redshift + z_plot) - cosmo.lookback_time(self.redshift)
            ).value

            # Have to do this for entire chain to get posterior for survivng stellar mass
            # spec, phot, mfrac = self.model.predict(theta_best, obs=self.obs, sps=self.sps)
            # self.approx_surviving_mass = np.sum(self.model.params["mass"]) * mfrac

            # ax[0].set_title(f'{field}:{id}, z$_{{phot}}$ = {redshift:.2f}, Log M$_*$ = {float(logmass_best):.2f} M$_{{\odot}}$, Log M$_{{*, 0}}$={np.log10(surviving_mass):.2f}')
            # ax[0].set_xlabel('Lookback Time (Myr)')
            if modify_ax:
                if plottype == "lookback":
                    ax.set_xlabel(f"Lookback Time ({timescale})")
                elif plottype == "absolute":
                    ax.set_xlabel(f"Age of Universe ({timescale})")

                ax.set_ylabel(r"SFR ($M_{\odot}$yr $^{-1}$)")
                # ax[1].set_ylabel(r'sSFR (yr $^{-1}$)')
                ax.annotate(self.sfh_model, (0.65, 0.05), xycoords="axes fraction", alpha=0.8)

            # ymax = np.max(median_sfrs)
            # ymax_ssfr = np.max(sfrs/cum_mass)

            agebins = np.ndarray.flatten(agebins)
            sfr_bins = np.ndarray.flatten(
                np.array(
                    [
                        [median_sfrs[i], median_sfrs[i]]
                        if (i != 0 or i != len(median_sfrs))
                        else median_sfrs[i]
                        for i in range(len(median_sfrs))
                    ]
                )
            )
            sfr_bins_low = np.ndarray.flatten(
                np.array(
                    [
                        [low_sfrs[i], low_sfrs[i]]
                        if (i != 0 or i != len(low_sfrs))
                        else low_sfrs[i]
                        for i in range(len(low_sfrs))
                    ]
                )
            )
            sfr_bins_high = np.ndarray.flatten(
                np.array(
                    [
                        [high_sfrs[i], high_sfrs[i]]
                        if (i != 0 or i != len(high_sfrs))
                        else high_sfrs[i]
                        for i in range(len(high_sfrs))
                    ]
                )
            )

            if timescale == "Myr":
                plot_agebins = 10 ** (agebins - 6)
            elif timescale == "Gyr":
                plot_agebins = 10 ** (agebins - 9)
            # plot_agebins = agebins
            if plottype == "lookback":
                line = ax.plot(
                    plot_agebins, sfr_bins, color=colour, linewidth=1, linestyle="dashdot", zorder=4
                )
                ax.fill_between(
                    plot_agebins,
                    sfr_bins_high,
                    sfr_bins_low,
                    alpha=0.3,
                    color=line[0].get_color(),
                    zorder=5,
                )
                """try:
                    plot_agebins_percentiles_lower = self.agebins_dist """
            elif plottype == "absolute":
                plot_agebins_absolute = np.array([time_convert(i) for i in plot_agebins])
                # print('sfh testing')
                # print(plot_agebins_absolute)
                line = ax.plot(plot_agebins_absolute, sfr_bins, color=colour, zorder=4)
                ax.fill_between(
                    plot_agebins_absolute,
                    sfr_bins_high,
                    sfr_bins_low,
                    alpha=0.5,
                    color=line[0].get_color(),
                    zorder=5,
                )

            ylim = ax.get_ylim()
            xlim = ax.get_xlim()
            if modify_ax:
                if ylim[0] < 1e3:
                    ylim = (1e-1, ylim[1])

                ax.set_ylim(ylim)

                ax.set_xlim(xlim)

            if show_lines:
                ax.vlines(
                    self.mwa[1],
                    ylim[0],
                    ylim[1],
                    color="black",
                    linestyle="--",
                    label="MWA",
                    lw=1.5,
                )
                ax.hlines(
                    self.sfr_10myr[1],
                    plot_agebins[0],
                    plot_agebins[-1],
                    color="black",
                    linestyle="-.",
                    label="SFR$_{10Myr}$",
                    lw=1.5,
                )
                ax.hlines(
                    self.sfr_100myr[1],
                    plot_agebins[0],
                    plot_agebins[-1],
                    color="black",
                    linestyle=":",
                    label="SFR$_{100Myr}$",
                    lw=1.5,
                )

            if flex:
                fact = 6 if timescale == "Myr" else 9
                ax.vlines(
                    10 ** (np.ndarray.flatten(agebins_init) - fact),
                    ylim[0],
                    ylim[1],
                    color="darkred",
                    linestyle="--",
                    label="Initial Agebins",
                    alpha=0.6,
                    lw=1.5,
                )
                """ax.vlines(10**(np.ndarray.flatten(agebins_low)-fact), ylim[0], ylim[1], color='darkred', linestyle='-.', alpha=0.6, lw=1)
                ax.vlines(10**(np.ndarray.flatten(agebins_high)-fact), ylim[0], ylim[1], color='darkred', linestyle=':', alpha=0.6, lw=1)"""

            if modify_ax:
                ax.legend(loc="best", fontsize=10, frameon=False)
                if logify:
                    ax.set_yscale("log")
                # ax[0].set_xscale('log')
                ax.xaxis.set_major_formatter(ScalarFormatter())
                # ax.yaxis.set_major_formatter(ScalarFormatter())

            if modify_ax and add_zaxis:
                if plottype != "lookback":
                    try:
                        test = time_convert(ax.get_xlim()[1])

                        secax = ax.secondary_xaxis("top", functions=(time_convert, z_convert))
                        secax.set_xlabel("Redshift")
                    except CosmologyError:
                        print(
                            "Cosmology error, probably because one of more agebins (or something else plotted) has lookback time > age of the Universe"
                        )
                        print("No redshift axis will be shown.")
            # np.random.c
            # hoice with weights for dynesty to select samples from chain

            return fig

        elif self.sfh_model in ["continuity_psb", "dirichlet"]:
            print("SFH plot not implemented for this SFH.")

        elif self.sfh_model in ["exp", "delayed", "rising"]:
            try:
                tage = np.ndarray.flatten(self.chain["tage"])
            except ValueError:
                tage_from_z = np.ndarray.flatten(self.chain["tage_tuniv"])
                z = np.ndarray.flatten(self.chain["zred"])
                tage = tage_from_tuniv(z, tage_from_z)  # in gyr
            tau = np.ndarray.flatten(self.chain["tau"])
            mass = np.ndarray.flatten(self.chain["mass"])
            tau_16, tau_50, tau_84 = weighted_quantile(tau, [0.16, 0.5, 0.84], self.weights)

            self.logmass_best_low, self.logmass_best, self.logmass_best_high = weighted_quantile(
                np.ndarray.flatten(np.log10(mass)), [0.16, 0.50, 0.84], self.weights
            )
            # tage = theta_best[labels=='tage']
            # tau  = theta_best[labels=='tau']
            # mass = theta_best[labels=='mass']
            # These need to be actual fitted values
            # for delay tau this function gives the (unnormalized) SFR
            if self.sfh_model in ["delayed", "rising"]:
                sfr = lambda t, tau: (t / tau) * np.exp(-t / tau)
                power = 1
            elif self.sfh_model == "exp":
                sfr = lambda t, tau: np.exp(-t / tau)
                power = 0

            self.mwa_dist = np.array(
                [parametric_mwa(tau_i, tage_i, power=power) for tau_i, tage_i in zip(tau, tage)]
            )
            self.mwa = weighted_quantile(self.mwa_dist, [0.16, 0.5, 0.84], self.weights)

            self.sfr_10myr_dist = np.array(
                [
                    parametric_sfr(tau_i, tage_i, power=power, sfr_period=0.01)
                    for tau_i, tage_i in zip(tau, tage)
                ]
            )

            self.sfr_10myr_dist = np.ndarray.flatten(self.sfr_10myr_dist)

            self.sfr_10myr = weighted_quantile(self.sfr_10myr_dist, [0.16, 0.5, 0.84], self.weights)

            self.sfr_100myr_dist = np.array(
                [
                    parametric_sfr(tau_i, tage_i, power=power, sfr_period=0.1)
                    for tau_i, tage_i in zip(tau, tage)
                ]
            )
            self.sfr_100myr_dist = np.ndarray.flatten(self.sfr_100myr_dist)
            self.sfr_100myr = weighted_quantile(
                self.sfr_100myr_dist, [0.16, 0.5, 0.84], self.weights
            )

            # now we numerically integrate this SFH from 0 to tage to get the mass formed
            # 2d lisnapce
            # times = np.meshgrid(*[np.linspace(0, t, 1000) for t in tage])
            # A = np.trapz(sfr(times, tau), times)
            # But this could also be done using an incomplete gamma function (integral of xe^{-x})
            A = tau * gamma(2) * gammainc(2, tage / tau)
            # and now we renormalize the formed mass to the actual mass value
            # to get the the SFR in M_sun per Gyr
            psi = mass * sfr(tage, tau) / A
            # if we want SFR in Msun/year
            psi /= 1e9
            # Maube use - https://prospect.readthedocs.io/en/latest/api/plotting_api.html#prospect.plotting.sfh.sfh_quantiles
            # prospect.plotting.sfh.parametric_sfr(times=None, tavg=0.001, tage=1, **sfh)
            times = np.linspace(0, np.max(tage), 1000)
            # spec, phot, mfrac = self.model.predict(theta_best, obs=self.obs, sps=self.sps)
            if ax is None:
                fig, ax = plt.subplots(1, 1, sharex=True)
                ax.set_ylabel(r"SFR ($M_{\odot}$yr $^{-1}$)")
                # ax[1].set_ylabel(r'sSFR (yr $^{-1}$)')
                # ax[0].set_title(f'{field}:{id}, z$_{{phot}}$ = {redshift:.2f}, Log M$_*$ = {float(mass):.2f} M$_{{\odot}}$, Log M$_{{*, 0}}$={np.log10(surviving_mass):.2f}, SFR: {psi:.2f}')
                ax.set_xlabel("Lookback Time (Gyr)")
                # A_t = []
                # for i in range(len(times)):
                #    time = times[:i]
                #    A_t.append(np.trapz(sfr(times, tau), times))
                # ax[1].plot(times, sfr(times, tau)/np.array(A_t))
                # ax = ax[0]
            else:
                fig = None
            # all_sfr = np.array([sfr(times, tau_i) for tau_i in tau])
            # sfh_16, sfh_50, sfh_84 = weighted_quantile(all_sfr, [0.16, 0.5, 0.84], self.weights)
            # ax.plot(times, sfh_50)
            # ax.fill_between(times, sfh_16, sfh_84, alpha=0.5)
            if plottype == "absolute":
                ax.plot(
                    times, sfr(times, tau_50), alpha=0.7, color=colour, linestyle="dashdot", lw=1
                )
                ax.fill_between(
                    times, sfr(times, tau_16), sfr(times, tau_84), alpha=0.5, color=colour
                )
            elif plottype == "lookback":
                lookback_times = cosmo.age(self.redshift).to(u.Gyr).value - times
                ax.plot(
                    lookback_times,
                    sfr(times, tau_50),
                    alpha=1,
                    color=colour,
                    linestyle="dashdot",
                    lw=1,
                )
                ax.fill_between(lookback_times, sfr(times, tau_16), alpha=0.3, color=colour)
            return fig

    def plot_table(self, ax, approx_mass=False, display_type={"total_mass": "log_10"}):
        if getattr(self, "sfr_10myr", None) is None:
            # This gets SFRs, MWA, survivng mass etc
            _ = self.plot_sfh()

        if not getattr(self, "redshift", False):
            try:
                redshift = self.chain["zred"]

                self.z_16, self.redshift, self.z_84 = weighted_quantile(
                    redshift, [0.16, 0.5, 0.84], self.weights
                )

            except:
                self.redshift = self.model.init_config["zred"]["init"]
                self.z_16, self.z_84 = 0, 0

        if not getattr(self, "chi2", False):
            self.get_chi2()

        if not getattr(self, "surviving_mass", False):
            test = self.recalc_surviving_mass(only_quick=approx_mass)

            if type(test) == bool:
                theta_best = self.result["chain"][self.ind_best, :]
                spec, phot, mfrac = self.model.predict(theta_best, obs=self.obs, sps=self.sps)
                self.approx_surviving_mass = np.log10(np.sum(self.model.params["mass"]) * mfrac)
                use_approx = True
            else:
                use_approx = False
        col_labels = None
        row_labels = [
            "ID",
            "Field",
            "Photo-z",
            "IMF",
            r"$ \rm{Surv \ Stellar \ Mass} \ (\log M_*/M_\odot)$",
            r"$\chi^2_{r}$",
        ]
        # ax[0].set_title(f'{field}:{id}, z$_{{phot}}$ = {redshift:.2f}, Log M$_*$ = {float(mass):.2f} M$_{{\odot}}$, Log M$_{{*, 0}}$={np.log10(surviving_mass):.2f}, SFR: {psi:.2f}')
        imf = {
            0: "Salpeter+55",
            1: "Chabrier+03",
            2: "Kroupa+01",
            3: "van Dokkum+08",
            4: "Dave+08",
            5: "custom",
        }
        imf_type = imf[self.model.init_config["imf_type"]["init"]]
        if imf_type == "custom" and self.result["run_params"]["use_hot_imf"]:
            imf_type = "Steinhardt+22"
        table_vals = [
            [str(self.id)],
            [self.field],
            [f"{self.redshift:.1f}"]
            if self.z_16 + self.z_84 == 0
            else [
                f"${self.redshift:.1f}^{{+{self.redshift-self.z_16:.1f}}}_{{-{self.z_84-self.redshift:.1f}}}$"
            ],
            [imf_type],
            [
                f"${self.approx_surviving_mass[1]:.2f}^{{+{self.approx_surviving_mass[1]-self.approx_surviving_mass[0]:.1f}}}_{{-{self.approx_surviving_mass[2]-self.approx_surviving_mass[1]:.1f}}}$"
            ]
            if use_approx
            else [
                f"${self.surviving_mass[1]:.2f}^{{+{self.surviving_mass[1]-self.surviving_mass[0]:.1f}}}_{{-{self.surviving_mass[2]-self.surviving_mass[1]:.1f}}}$"
            ],
            [f"{self.chi2:.1f}"],
        ]

        for param in self.show:
            print("Shown params:", param)

            if display_type.get(param, False):
                dtype = display_type[param]
                if dtype == "log_10":
                    values = np.log10(self.chain[param])
            else:
                values = self.chain[param]

            low, mid, high = weighted_quantile(np.squeeze(values), [0.16, 0.5, 0.84], self.weights)
            table_vals.append([f"{mid:.2f}$^{{+{mid-low:.1f}}}_{{-{high-low:.1f}}}$"])
            row_labels.append(self.param_name.get(param, param))

        try:
            table_vals.append(
                [
                    f"{self.sfr_10myr[1]:.2f}$^{{+{self.sfr_10myr[2]-self.sfr_10myr[1]:.1f}}}_{{-{self.sfr_10myr[1]-self.sfr_10myr[0]:.1f}}}$"
                ]
            )
            row_labels.append(r"$ \rm{\ SFR_{10 Myr}} (M_\odot \ \rm{yr}^{-1})$")
            table_vals.append(
                [
                    f"{self.sfr_100myr[1]:.2f}$^{{+{self.sfr_100myr[2]-self.sfr_100myr[1]:.1f}}}_{{-{self.sfr_100myr[1]-self.sfr_100myr[0]:.1f}}}$"
                ]
            )
            row_labels.append(r"$ \rm{\ SFR_{100 Myr}} (M_\odot \ \rm{yr}^{-1})$")
            table_vals.append(
                [
                    f"{self.mwa[1]:.1f}$^{{+{self.mwa[2]-self.mwa[1]:.1f}}}_{{-{self.mwa[1]-self.mwa[0]:.1f}}}$"
                ]
            )
            row_labels.append(r"$\rm{MW \ Age \ (Myr)}$")

        except UnboundLocalError:
            pass
            # ax[1].plot([bin_low, bin_high], sfrs[i]/cum_mass)

        table = ax.table(
            cellText=table_vals,
            rowLabels=row_labels,
            colLabels=col_labels,
            edges="vertical",
            fontsize=30,
            bbox=[0.606, 0, 1 - 0.66, 1],
        )

    def add_to_cat(
        self,
        cat_path,
        exclude=["logsfr_ratios", "logsfr_ratio_old", "logsfr_ratio_young", "z_fraction"],
        approx_mass=True,
        replace_rows=True,
    ):
        run_params = self.result["run_params"]
        id = run_params["OBJID"]
        field = run_params["field"]
        sfh_model = run_params["sfh_model"]
        fit_type = run_params["fit_type"]
        hot_imf = run_params["use_hot_imf"]
        imf_type = self.model.init_config["imf_type"]["init"]
        if imf_type != 5 and self.result["run_params"]["use_hot_imf"]:
            print("You probably don't want to use me as I was actually run with a Kroupa IMF")
            return False
        dust_prior = run_params["dust_prior"]
        age_prior = run_params["age_prior"]
        vary_redshift = run_params["vary_redshift"]
        vary_u = run_params["vary_u"]
        try:
            redshift_sigma = run_params["redshift_sigma"]
        except:
            redshift_sigma = np.NaN

        # Dynesty settings

        nlive_init = run_params["nlive_init"]
        nested_sample = run_params["nested_sample"]
        nested_target_n_effective = run_params[
            "nested_target_n_effective"
        ]  #  A value of 10,000 for this keyword specifies high-quality posteriors, whereas a value of 3,000 will produce reasonable but approximate posteriors.
        nlive_batch = run_params["nlive_batch"]
        nested_dlogz_init = run_params["nested_dlogz_init"]
        nested_maxcal = run_params["nested_maxcall"]
        try:
            redshift = self.chain["zred"]

            redshift = weighted_quantile(redshift, [0.16, 0.5, 0.84], self.weights)

        except:
            redshift = [0, self.model.init_config["zred"]["init"], 0]
        # Recalculate chi2 if it doesn't exist
        if getattr(self, "chi2", None) is None:
            self.get_chi2()
        # This gets SFRs, MWA, survivng mass etc
        _ = self.plot_sfh()

        use_approx = False
        if not getattr(self, "surviving_mass", False):
            test = self.recalc_surviving_mass(only_quick=approx_mass)

            if test == False:
                theta_best = self.result["chain"][self.ind_best, :]
                spec, phot, mfrac = self.model.predict(theta_best, obs=self.obs, sps=self.sps)
                self.approx_surviving_mass = np.log10(np.sum(self.model.params["mass"]) * mfrac)
                use_approx = True

        # MUST match parnames_quant list!
        table = {
            "id": id,
            "field": field,
            "redshift": redshift,
            "sfh_model": sfh_model,
            "chi2_red": self.chi2,
            "fit_type": fit_type,
            "nlive_init": nlive_init,
            "nested_sample": nested_sample,
            "nested_target_n_effective": nested_target_n_effective,
            "nlive_batch": nlive_batch,
            "nested_dlogz_init": nested_dlogz_init,
            "nested_maxcal": nested_maxcal,
            "dust_prior": dust_prior,
            "surviving_mass": self.approx_surviving_mass if use_approx else self.surviving_mass[1],
            "surviving_mass_16": 0.00 if use_approx else self.surviving_mass[0],
            "surviving_mass_84": 0.00 if use_approx else self.surviving_mass[2],
            "sfr_10myr": self.sfr_10myr,
            "sfr_100myr": self.sfr_100myr,
            "mwa": self.mwa,
            "age_prior": age_prior,
            "vary_redshift": vary_redshift,
            "redshift_sigma": redshift_sigma,
            "UNIQUE_ID": f"{id}_{field}",
        }

        self.show = [i for i in self.chain.dtype.names if i not in exclude]
        for param in self.show:
            vals = weighted_quantile(np.squeeze(self.chain[param]), [0.16, 0.5, 0.84], self.weights)
            table[param] = vals

        data = table.values()
        data = [[i] for i in data]

        if os.path.exists(cat_path):
            catalog = Table.read(cat_path)
        else:
            names = table.keys()

            # if len(data) != len(names):
            #    print('Catalog writing broken!')
            catalog = Table(data=data, names=names)
        if len(catalog) == 0:
            catalog = Table(data=data, names=names)
        skip = False
        for pos, row in enumerate(catalog):
            if id == row["id"] and field == row["field"]:
                if replace_rows:
                    print("Replacing row")
                    catalog.remove_row(pos)
                else:
                    print("Skipping row")
                    skip = True

        if not skip:
            for key in table.keys():
                if key not in catalog.colnames:
                    print("Adding missing column(s)")
                    print(f"Adding {key}")
                    if type(table[key]) in [np.ndarray, list]:
                        catalog[key] = np.ones(shape=(len(catalog), len(table[key]))) * -99

                    row_len = np.shape(table[key])

            # print(table.keys(), catalog.colnames)
            try:
                catalog.add_row(table)
            except ValueError as e:
                print("mismatch")
                print(id, field)
                print(len(table), len(catalog.colnames))
                print(table.keys(), catalog.colnames)
                print(e)
        path = "/".join(cat_path.split("/")[:-1])
        # print(path)
        if not os.path.exists(path):
            os.makedirs(path)
        catalog["UNIQUE_ID"] = [
            f"{id}_{field}" for id, field in zip(catalog["id"], catalog["field"])
        ]
        catalog.write(cat_path, overwrite=True)

    def get_chi2(self):
        pmask = self.obs["phot_mask"]
        pmask = pmask & (self.obs["maggies"][pmask] > 0)
        ophot, ounc = self.obs["maggies"][pmask], self.obs["maggies_unc"][pmask]
        if getattr(self, "phot_best", None) is None:
            xbest = self.result["chain"][self.ind_best, :]
            if self.sps is None:
                self.build_sps()
            blob = self.model.predict(xbest, obs=self.obs, sps=self.sps)

            self.spec_best, self.phot_best, self.mfrac_best = blob

        phot_best = self.phot_best[pmask]
        print("chi2 debug")
        print("Note not using negative flux bands ")
        print(phot_best / ophot, ounc / ophot * 100, ((phot_best - ophot) ** 2) / ounc**2)
        self.chi2 = np.sum(((ophot - phot_best) ** 2) / ounc**2)

    def recalc_surviving_mass(self, indices=None, draws=None, only_quick=True, num_draws=400):
        if getattr(self, "surviving_mass_dist", None) is None:
            # This is really slow!
            if self.sps is None:
                self.build_sps()

            try:
                with h5py.File(self.results_file, "r") as f:
                    mass = np.array(f["extra_data/surviving_mass"][:])
                    indices = np.array(f["extra_data/mass_indices"][:])
                    if draws is not None:
                        if len(mass) != len(draws):
                            mfrac = []
                            print("Found surviving mass in h5 but not the right length.")
                            self.cause_exception

                    draws = mass
                    print("Loaded surviving mass from h5.")

            except Exception as e:
                print(e)

                if not only_quick:
                    mfrac = []
                    if indices is None and draws is None:
                        try:
                            values = np.ndarray.flatten(self.chain["logmass"])
                        except ValueError:
                            values = np.log10(np.ndarray.flatten(self.chain["mass"]))

                        if np.sum(np.ndarray.flatten(self.weights)) != 1:
                            print("Weights not normalized, normalizing.")
                            self.weights = self.weights / np.sum(self.weights)
                        indices = choice(
                            range(len(values)), num_draws, p=np.ndarray.flatten(self.weights)
                        )
                        draws = values[indices]
                    for result in tqdm(self.result["chain"][indices, :]):
                        try:
                            blob = self.model.predict(result, obs=self.obs, sps=self.sps)
                            _, _, mfrac_chain = blob
                            # print('Not failed.')
                            mfrac.append(mfrac_chain)
                        except ValueError:
                            mfrac.append(0)
                            print("Failed.")

                    mfrac = np.array(mfrac)
                    draws = np.log10((10**draws) * mfrac)
                    self.save_updated_h5(draws, "surviving_mass")
                    self.save_updated_h5(mfrac, "mfrac")
                    self.save_updated_h5(indices, "mass_indices")
                else:
                    return False

            self.surviving_mass_dist = draws
            # self.mfrac = mfrac

            low, mid, high = weighted_quantile(
                np.squeeze(draws),
                [0.16, 0.5, 0.84],
                self.weights[indices] / np.sum(self.weights[indices]),
            )
            self.surviving_mass = [low, mid, high]

            return draws

        else:
            return self.surviving_mass_dist

    def save_pdf(self, parameter, pdf_path, num_draws=400):
        params = self.result["run_params"]
        self.id = params["OBJID"]
        self.field = params["field"]

        if parameter == "surviving_mass":
            draws = self.recalc_surviving_mass()
        else:
            values = np.ndarray.flatten(self.chain[parameter])
            indexes = choice(range(len(values)), num_draws, p=np.ndarray.flatten(self.weights))
            draws = values[indexes]

        header = str(self.field) + ",ID=" + str(self.id) + "," + parameter + "_PDF"
        try:
            samples = self.chain[parameter]
            units = samples[0].unit
            samples = samples.value
            header = header + ": UNITS=" + str(units)
        except:
            header = header + ": UNITS=unitless"

        path = f"{pdf_path}/{self.id}_{self.field}.txt"  # ,{param_name}
        os.makedirs(pdf_path, exist_ok=True)

        print("Saving PDF: " + path)
        np.savetxt(path, draws, header=header)

    def save_sed(self, sed_path, microns=True, nufnu=False, save=False):
        # This is model config.
        # print(model.init_config)
        """A very basic plot of the observed photometry and the best fit
        photometry and spectrum.
        """

        if self.sps is None:
            self.build_sps()

        # --- best sample ---

        pmask = self.obs["phot_mask"]
        xbest = self.result["chain"][self.ind_best, :]
        if getattr(self, "spec_best", None) is None:
            blob = self.model.predict(xbest, obs=self.obs, sps=self.sps)
            self.spec_best, self.phot_best, self.mfrac_best = blob

        pwave, phot_best = self.obs["phot_wave"][pmask], self.phot_best[pmask]

        filtname = np.array([f.name for f in self.obs["filters"]])[pmask]

        spec_best = self.spec_best
        swave = self.obs.get("wavelength", None)
        if swave is None:
            if "zred" in self.model.free_params:
                zred = self.chain["zred"][self.ind_best]
            else:
                zred = self.model.params["zred"]
            swave = self.sps.wavelengths * (1 + zred)
        if nufnu:
            # nufnu is cgs, erg cm-2 s-1
            swave, spec_best = to_nufnu(swave, spec_best, microns=microns)
            pwave, phot_best = to_nufnu(pwave, phot_best, microns=microns)

        if microns and not nufnu:
            swave = swave / 10**4
            pwave = pwave / 10**4

        if not nufnu:
            phot_best = maggies_to_mag(phot_best)
            spec_best = maggies_to_mag(spec_best)

        table = Table(
            [swave, spec_best],
            names=["Wavelength", "Best Observed Frame SED"],
            units=[
                u.micron if microns else u.angstrom,
                u.ABmag if not nufnu else u.erg / (u.cm**2 * u.s),
            ],
        )

        run_params = self.result["run_params"]

        id = run_params["OBJID"]
        field = run_params["field"]
        sfh_model = run_params["sfh_model"]
        vary_redshift = run_params["vary_redshift"]

        table.meta["ID"] = id
        table.meta["FIELD"] = field
        table.meta["redshift"] = float(zred)
        table.meta["vary_redshift"] = vary_redshift
        table.meta["sfh_model"] = sfh_model

        if not getattr(self, "chi2", False):
            self.get_chi2()

        table.meta["chi2"] = float(self.chi2)
        run_params = self.result["run_params"]

        for filter, wave, phot in zip(filtname, pwave, phot_best):
            table.meta[str(filter)] = [float(wave), float(phot)]
        pwave, phot_best  # Best-fit photometry

        if save:
            if sed_path is not None:
                table.write(f"{sed_path}_{id}_{field}.ecsv", format="ascii.ecsv", overwrite=True)
            else:
                print("No table name given.")
                return False
        return table

    def pretty_plot(
        self,
        xlim=None,
        params_hist=None,
        exclude=["logsfr_ratios", "logsfr_ratio_old", "logsfr_ratio_young", "z_fraction"],
        param_name={
            "logzsol": r"Stellar $Z_* (\log_{10} \frac{Z_*}{Z_\odot})$",
            "gas_logz": r"Gas $Z (\log_{10} \frac{Z_g}{Z_\odot})$",
            "dust2": "Dust Extinction (mag)",
            "logmass": r"Formed Stellar Mass ( $ \log_{10}\frac{M_*}{M_\odot}$)",
            "gas_logu": "log$_{10}$ U",
            "total_mass": r"Formed Stellar Mass ( $ \log_{10}\frac{M_*}{M_\odot}$)",
            "fagn": "AGN Fraction",
            "dust_ratio": "Dust Ratio",
            "agn_tau": "AGN Optical Depth",
            "dust_index": "Dust Index",
            "zred": "Redshift",
        },
        display_type={"total_mass": "log_10"},
    ):
        imf_type = self.model.init_config["imf_type"]["init"]
        if imf_type != 5 and self.result["run_params"]["use_hot_imf"]:
            print("You probably don't want to use me as I was actually run with a Kroupa IMF")
            return False

        fig = plt.figure(facecolor="w", edgecolor="k", constrained_layout=True)
        gs = fig.add_gridspec(4, 5)
        self.param_name = param_name
        ax_photo = fig.add_subplot(gs[0:2, 0:3])
        table_plot = fig.add_subplot(gs[0:2, 3:])
        sfr_plot = fig.add_subplot(gs[2:4, 0:2])
        cutout_ax = []
        for pos in range(2, 5):
            cutout_ax.append(fig.add_subplot(gs[3, pos : pos + 1]))
            cutout_ax.append(fig.add_subplot(gs[2, pos : pos + 1]))

        # [ax.tick_params(axis='both', which='major', direction='out') for ax in [ax_photo, table_plot, sfr_plot, cutout_ax]]

        self.plot_sed_new(ax_photo, fig=fig, microns=True)
        pmask = self.obs["phot_mask"]

        phot_wav = self.obs["phot_wave"][pmask]

        if xlim is not None:
            ax_photo.set_xlim(xlim[0], xlim[1])
        else:
            # print('Setting lim to ', 0.9*phot_wav[0]*10**(-4), phot_wav[-1]*10**(-4)*1.1)
            ax_photo.set_xlim(
                0.9 * np.min(phot_wav) * 10 ** (-4), 1.1 * np.max(phot_wav) * 10 ** (-4)
            )

        self.plot_sfh(ax=sfr_plot)
        table_plot.set_axis_off()

        if params_hist is None:
            loop = [i for i in self.chain.dtype.names if i not in exclude]

        else:
            loop = params_hist

        self.show = loop

        self.plot_table(table_plot, display_type=display_type)

        lim = []
        num = 0
        for pos, param in enumerate(loop):
            if pos < len(cutout_ax):
                ax = cutout_ax[pos]
                values = self.chain[param]
                if display_type.get(param, False) == "log_10":
                    values = np.log10(values)
                ax.set_xlabel(param_name.get(param, param), fontsize=10)

                if param in ["logmass", "total_mass"]:
                    mass = getattr(self, "surviving_mass_dist", None)
                    if mass is not None:
                        values = mass
                        ax.set_xlabel(
                            r"Surv Stellar Mass ( $ \log_{10}\frac{M_*}{M_\odot}$)", fontsize=10
                        )

                try:
                    ax.set_yticks([])
                    # ax.hist(values, weights=self.weights)
                    plot_range_init = np.linspace(np.min(values), np.max(values), 100)
                    plot_range = plot_range_init.reshape(-1, 1)
                    values[~np.isfinite(values)] = np.median(values[np.isfinite(values)])
                    values[np.isnan(values)] = np.median(values[~np.isnan(values)])
                    values = np.array(values).reshape(-1, 1)
                    kde = KernelDensity(kernel="tophat", bandwidth=0.2).fit(values)

                    data = np.exp(kde.score_samples(plot_range))
                    line = ax.plot(plot_range_init, data, alpha=0.8)
                    ax.fill_between(
                        plot_range_init,
                        y2=data,
                        y1=np.zeros(len(data)),
                        alpha=0.4,
                        color=line[0].get_color(),
                    )

                    lim.append(ax.get_xlim())
                    num += 1
                except:
                    print("Failed to plot histogram for ", param)
                    pass

            else:
                print(f"Too many parameters to plot. Not showing {param}.")
                # print(self.show, param)
                # self.show.remove(param)
        try:
            self.show_priors(cutout_ax[: num - 1], spans=lim, linestyle="dashed", alpha=0.6)
        except IndexError:  # This errors because it trys to plot priors for all fitted parameters, but we dont make axis for all of them
            print(len(self.show), len(cutout_ax), num)
            pass
        [ax.set_axis_off() for ax in cutout_ax[len(loop) :]]

        # gs.tight_layout(fig)

        # except Exception as e:
        #    print(e)
        #    print('Failed to plot table. Tom please fix.')

        # plt.subplots_adjust(hspace=0.6, wspace=0.4)
        return fig

    def convert_param_name(self, parameter):
        param_dict = {
            "stellar_mass": "surviving_mass",
            "metallicity": "logzsol",
            "dust:Av": "dust2",
            "nebular:logU": "gas_logu",
            "redshift": "zred",
        }
        return param_dict.get(parameter, parameter)

    def plot_pdf(
        self,
        ax,
        parameter,
        num_draws=400,
        colour="black",
        fill_between=False,
        alpha=1,
        modify_ax=False,
        **kwargs,
    ):
        imf_type = self.model.init_config["imf_type"]["init"]
        if imf_type != 5 and self.result["run_params"]["use_hot_imf"]:
            print("You probably don't want to use me as I was actually run with a Kroupa IMF")
            return False

        if parameter not in ["ssfr", "sfr"]:
            parameter = self.convert_param_name(parameter)
            if parameter == "surviving_mass":
                draws = self.recalc_surviving_mass(only_quick=False)
                self.surviving_mass_dist = draws
            else:
                if parameter == "logmass":
                    try:
                        values = self.chain["logmass"]
                    except ValueError:
                        values = np.log10(self.chain["mass"])
                else:
                    values = np.ndarray.flatten(self.chain[parameter])

        elif parameter == "sfr":
            values = np.log10(self.sfr_10myr_dist)
        elif parameter == "ssfr":
            return False
        if parameter != "surviving_mass":
            indexes = choice(range(len(values)), num_draws, p=np.ndarray.flatten(self.weights))
            draws = values[indexes]

        try:
            sys.path.insert(3, "/nvme/scratch/work/tharvey/pipes/")
            from general import hist1d

            hist1d(
                draws[np.invert(np.isnan(draws))],
                ax,
                smooth=True,
                color=colour,
                percentiles=False,
                lw=1,
                alpha=alpha,
                linestyle="dashdot",
                fill_between=fill_between,
            )

        except ValueError:
            print("I shit myself.")
            pass

        if modify_ax:
            label = parameter
            # ax.set_xlabel(label,  fontsize='small')
            ax.set_title(parameter, fontsize="small")
            ax.tick_params(axis="both", which="major", labelsize="medium")
            ax.tick_params(axis="both", which="minor", labelsize="medium")

    # Need to give Plotspector

    def plot_best_fit(
        self,
        ax,
        colour="black",
        wav_units=u.um,
        alpha=0.7,
        flux_units=u.ABmag,
        lw=1,
        fill_uncertainty=False,
        zorder=5,
        **kwargs,
    ):
        microns = True if wav_units == u.um else False

        imf_type = self.model.init_config["imf_type"]["init"]
        if imf_type != 5 and self.result["run_params"]["use_hot_imf"]:
            print("You probably don't want to use me as I was actually run with a Kroupa IMF")
            return False

        if not getattr(self, "surviving_mass", False):
            test = self.recalc_surviving_mass(only_quick=False)

        if not getattr(self, "chi2", False):
            self.get_chi2()
        try:
            self.lkwargs
        except AttributeError:
            self.lkwargs = {}
        self.lkwargs["lw"] = lw
        self.lkwargs["zorder"] = zorder
        # self.lkwargs['color'] = colour
        self.lkwargs["alpha"] = alpha
        run_params = self.result["run_params"]
        sfh_model = run_params["sfh_model"]
        dust_prior = run_params["dust_prior"]
        dust_type = self.model.init_config["dust_type"]["init"]

        if int(dust_type) in [2, 2.0]:
            dust_type == "Cal"

        age_prior = run_params["age_prior"]
        vary_redshift = run_params["vary_redshift"]
        vary_u = run_params["vary_u"]
        imf = {
            0: "Salpeter+55",
            1: "Chabrier+03",
            2: "Kroupa+01",
            3: "van Dokkum+08",
            4: "Dave+08",
            5: "custom",
        }
        imf_type = imf[self.model.init_config["imf_type"]["init"]]
        if imf_type == "custom" and self.result["run_params"]["use_hot_imf"]:
            imf_type = "Steinhardt+22"
        try:
            redshift = self.chain["zred"]

            self.z_16, self.redshift, self.z_84 = weighted_quantile(
                np.squeeze(redshift), [0.16, 0.5, 0.84], self.weights
            )

        except Exception as e:
            print(e)
            self.redshift = self.model.init_config["zred"]["init"]
            self.z_16, self.z_84 = 0, 0

        redshift = (
            f"{self.redshift:.1f}"
            if self.z_16 == self.z_84 == 0
            else f"{self.redshift:.1f}^{{+{self.redshift-self.z_16:.1f}}}_{{-{self.z_84-self.redshift:.1f}}}"
        )

        try:
            redshift_sigma = run_params["redshift_sigma"]
            redshift_type = "zgauss"
        except:
            redshift_type = "zfix"
            redshift_sigma = np.NaN

        if getattr(self, "chi2", None) is None:
            self.get_chi2()

        if dust_prior == "uniform":
            dust_prior = "flat"
        if age_prior == "uniform":
            age_prior = "flat"
        if dust_prior == "log_10":
            dust_prior = r"$\log_{10}$"
        if age_prior == "log_10":
            age_prior = r"$\log_{10}$"

        if sfh_model == "continuity_bursty":
            sfh_model = "bursty"

        use_approx = False
        ## Dust: {dust_type}, {dust_prior}
        mass = (
            f"{self.approx_surviving_mass[1]:.1f}^{{+{self.approx_surviving_mass[1]-self.approx_surviving_mass[0]:.1f}}}_{{-{self.approx_surviving_mass[2]-self.approx_surviving_mass[1]:.1f}}}"
            if use_approx
            else f"{self.surviving_mass[1]:.1f}^{{+{self.surviving_mass[1]-self.surviving_mass[0]:.1f}}}_{{-{self.surviving_mass[2]-self.surviving_mass[1]:.1f}}}"
        )
        start = f"PR, {sfh_model} SFH"
        if imf_type == "Steinhardt+22":
            start += ", HOT IMF"
        end = f"$z:{redshift} \\ \\log M_{{*}}:{mass} \\ \\chi^{{2}}:{self.chi2:<4.1f}$"

        label = f"{end:<30} ({start})"
        self.plot_sed_new(
            ax,
            plot_type="fitted_spectrum",
            modify_ax=False,
            microns=microns,
            label_spec=label,
            color=colour,
        )

    def plot_best_photometry(
        self,
        ax,
        colour="black",
        wav_units=u.um,
        flux_units=u.ABmag,
        zorder=4,
        y_scale=None,
        lw=4,
        skip_no_obs=False,
        background_spectrum=False,
        **kwargs,
    ):
        imf_type = self.model.init_config["imf_type"]["init"]
        if imf_type != 5 and self.result["run_params"]["use_hot_imf"]:
            print("You probably don't want to use me as I was actually run with a Kroupa IMF")
            return False

        microns = True if wav_units == u.um else False
        try:
            self.pkwargs
        except AttributeError:
            self.pkwargs = {}
        self.pkwargs["lw"] = lw
        self.pkwargs["zorder"] = zorder
        # self.pkwargs['color'] = colour
        label = ""
        self.plot_sed_new(
            ax,
            plot_type="fitted_photometry",
            modify_ax=False,
            microns=microns,
            label_phot=label,
            color=colour,
        )
