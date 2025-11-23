#!/usr/bin/env python3
# Author: Simeon Reusch (simeon.reusch@desy.de)
# License: BSD-3-Clause

import argparse
import logging
import os
import time

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy import constants as const
from astropy import units as u
from astropy.cosmology import Planck15 as cosmo
from astropy.table import Table
from astropy.time import Time

import seaborn as sns
from modelSED import utilities

flabel_sel = "filterlabel_with_wl"

matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.serif"] = "Palatino"

MJD_INTERVALS = [[58700, 58720], [59006, 59130], [59220, 59271]]


def nu_to_ev(nu):
    """ """
    energy = const.h.value * const.c.value / (utilities.nu_to_lambda(nu) * 1e-10)
    ev = energy / 1.602e-19
    return ev


def ev_to_nu(ev):
    """ """
    lam = const.h.value * const.c.value / (ev * 1e-10)
    nu = utilities.lambda_to_nu(lam)
    return nu


def convert_mJy_to_abmag(df):
    """ """
    fnu = df["fnu_mJy"] / 1000 * 1e-23
    fnu_err = df["fnu_mJy_err"] / 1000 * 1e-23
    df["mag"] = utilities.flux_to_abmag(fnu)
    df["mag_err"] = utilities.flux_err_to_abmag_err(fnu, fnu_err)
    df.drop(columns=["fnu_mJy", "fnu_mJy_err"], inplace=True)
    return df


def plot_lightcurve(df, fluxplot=False):
    """ """
    fig = plt.figure(dpi=DPI, figsize=(FIG_WIDTH, FIG_WIDTH / GOLDEN_RATIO))

    config = {
        "P48+ZTF_g": {"s": 3, "fmt": "o", "a": 0.5, "c": "g", "mfc": None},
        "P48+ZTF_r": {"s": 3, "fmt": "^", "a": 0.5, "c": "r", "mfc": "none"},
        "P48+ZTF_i": {"s": 3, "fmt": "s", "a": 0.5, "c": "orange", "mfc": "none"},
        "Swift+U": {"s": 3, "fmt": "D", "a": 1, "c": "purple", "mfc": None},
        "WISE+W1": {"s": 6, "fmt": "p", "a": 1, "c": "tab:blue", "mfc": None},
        "WISE+W2": {"s": 6, "fmt": "h", "a": 1, "c": "tab:red", "mfc": None},
        "P200+J": {"s": 5, "fmt": "d", "a": 1, "c": "black", "mfc": "none"},
        "P200+H": {"s": 5, "fmt": "d", "a": 1, "c": "gray", "mfc": None},
        "P200+Ks": {"s": 5, "fmt": "s", "a": 1, "c": "blue", "mfc": None},
        "Swift+UVW1": {"s": 7, "fmt": "$\u2665$", "a": 1, "c": "orchid", "mfc": "none"},
        "Swift+UVW2": {"s": 3, "fmt": "D", "a": 1, "c": "m", "mfc": None},
    }

    cmap_rgba = {}

    for entry in cmap:
        rgba = {entry: mcolors.to_rgba(cmap[entry])}
        cmap_rgba.update(rgba)

    for entry in config:
        if config[entry]["mfc"] == "none":
            temp = list(cmap_rgba[entry])
            temp[-1] = 0.4
            cmap_rgba.update({entry: tuple(temp)})

    plt.subplots_adjust(bottom=0.12, top=0.85, left=0.11, right=0.9)

    filter_wl = utilities.load_info_json("filter_wl")

    ax1 = fig.add_subplot(14, 1, (2, 14))

    ax1.set_xlim([58580, 59480])

    if fluxplot:
        plt.yscale("log")
        ax1.set_ylim([1.7e-14, 1.7e-12])

    lcs_per_band = []

    for instrband in cmap:
        telescope, band = instrband.split("+")
        if telescope == "P200_sextractor":
            fmt = "*"
        else:
            fmt = "."

        if instrband not in BANDS_TO_EXCLUDE:
            lc = df.query(f"telescope == '{telescope}' and band == '{band}'")
            if not fluxplot:
                y = lc.mag
                yerr = lc.mag_err
            else:
                d = cosmo.luminosity_distance(REDSHIFT)
                d = d.to(u.cm).value
                lumi = lambda flux: flux * 4 * np.pi * d**2
                flux = lambda lumi: lumi / (4 * np.pi * d**2)
                ax2 = ax1.secondary_yaxis("right", functions=(lumi, flux))
                ax2.tick_params(axis="y", which="major", labelsize=BIG_FONTSIZE)

                flux_density = utilities.abmag_to_flux(lc.mag)
                flux_density_err = utilities.abmag_err_to_flux_err(lc.mag, lc.mag_err)
                flux, flux_err = utilities.flux_density_to_flux(
                    filter_wl[instrband], flux_density, flux_density_err
                )
                y = flux
                yerr = flux_err

                lc["flux"] = y
                lc["flux_err"] = yerr

                lc.drop(columns=["mag", "mag_err", "alert"], inplace=True)

            if instrband in [
                "P48+ZTF_g",
                "P48+ZTF_r",
                "P48+ZTF_i",
                "WISE+W1",
                "WISE+W2",
                "P200+H",
                "P200+Ks",
                "P200+J",
                "Swift+UVW1",
                # "Swift+UVW2",
            ]:
                ax1.errorbar(
                    x=lc.obsmjd,
                    y=y,
                    yerr=yerr,
                    color=cmap[instrband],
                    mec=cmap[instrband],
                    mfc=cmap_rgba[instrband],
                    marker=config[instrband]["fmt"],
                    ms=config[instrband]["s"],
                    # alpha=config[instrband]["a"],
                    # mfc=config[instrband]["mfc"],
                    elinewidth=0.7,
                    mew=0.7,
                    linestyle=" ",
                    label=filterlabel[instrband],
                )
                lcs_per_band.append(lc)

    if fluxplot:
        if flabel_sel == "filterlabel_with_wl":
            label = "XRT (0.3–10 keV)"
        else:
            label = "Swift XRT"

        y = df_swift_xrt["flux_unabsorbed"]
        yerr = df_swift_xrt["flux_unabsorbed"] / 10
        ax1.errorbar(
            x=df_swift_xrt.obsmjd,
            xerr=df_swift_xrt.range,
            y=y,
            yerr=yerr,
            uplims=True,
            marker="x",
            color="darkviolet",
            label=label,
            elinewidth=0.8,
        )

        if flabel_sel == "filterlabel_with_wl":
            label = "eROSITA (0.3–2 keV)"
        else:
            label = "SRG eROSITA"

        erosita_detection = pd.DataFrame()
        erosita_detection["band"] = np.asarray(["0.2-2 keV"])
        erosita_detection["telescope"] = np.asarray(["eROSITA"])
        erosita_detection["obsmjd"] = np.asarray([59283.685482])
        erosita_detection["flux"] = np.asarray([6.2e-14])
        erosita_detection["flux_err"] = np.asarray([2.7e-14])

        lcs_per_band.append(erosita_detection)

        ax1.errorbar(
            x=59283.685482,
            y=6.2e-14,
            yerr=[[2.7e-14], [2.1e-14]],
            fmt="D",
            ms=6,
            color="darkcyan",
            label=label,
            elinewidth=0.8,
        )

        df_erosita_ulims = pd.read_csv(os.path.join(LC_DIR, "erosita_ulims.csv"))
        y = df_erosita_ulims.flux
        yerr = y / 10

        ax1.errorbar(
            x=df_erosita_ulims.obsmjd,
            xerr=df_erosita_ulims.obsmjd - df_erosita_ulims.obsmjd_start,
            y=df_erosita_ulims.flux,
            yerr=yerr,
            uplims=True,
            fmt="D",
            ms=3,
            color="darkcyan",
            elinewidth=0.8,
        )

        df_fermi_cut = df_fermi.query("obsmjd == 58799.5")

        y = df_fermi_cut["flux"]
        yerr = df_fermi_cut["flux"] / 10

        if flabel_sel == "filterlabel":
            label = "Fermi LAT"
        else:
            label = "LAT (0.1–800 GeV)"

        ax1.errorbar(
            x=df_fermi_cut.obsmjd,
            xerr=df_fermi_cut.obsmjd - df_fermi_cut.obsmjd_start,
            y=y,
            yerr=yerr,
            uplims=True,
            fmt=" ",
            color="turquoise",
            label=label,
            elinewidth=0.8,
        )

    if not fluxplot:
        ax1.invert_yaxis()
        ax1.set_ylabel(r"Apparent Magnitude [AB]", fontsize=BIG_FONTSIZE)
    else:
        ax1.set_ylabel(
            r"$\nu$ F$_\nu$ (erg s$^{-1}$ cm$^{-2}$)", fontsize=BIG_FONTSIZE + 1
        )
        ax2.set_ylabel(r"$\nu$ L$_\nu$ (erg s$^{-1}$)", fontsize=BIG_FONTSIZE + 1)

    ax1.tick_params(axis="both", which="major", labelsize=BIG_FONTSIZE)
    ax1.set_xlabel("Date (MJD)", fontsize=BIG_FONTSIZE + 1)
    ax1.grid(which="both", visible=True, axis="both", alpha=0.2)
    if not fluxplot:
        ax1.set_ylim([22, 16])
        absmag = lambda mag: mag - cosmo.distmod(REDSHIFT).value
        mag = lambda absmag: absmag + cosmo.distmod(REDSHIFT).value
        ax2 = ax1.secondary_yaxis("right", functions=(absmag, mag))
        ax2.tick_params(axis="both", which="major", labelsize=BIG_FONTSIZE)
        ax2.set_ylabel(f"Absolute Magnitude (z={REDSHIFT:.3f})", fontsize=BIG_FONTSIZE)

    t_neutrino = Time("2020-05-30T07:54:29.43", format="isot", scale="utc")

    ax1.axvline(t_neutrino.mjd, linestyle=":", label="IC200530A", color="tab:red")

    loc = (0.73, 0.75)

    if flabel_sel == "filterlabel":
        bbox = [1.105, 1.26]
        fontsize = 10
    else:
        # bbox = [1.12, 1.26]
        bbox = [1.12, 1.26]
        fontsize = 9.8

    if flabel_sel == "filterlabel":
        ncol = 6
    else:
        ncol = 5

    ax1.legend(
        ncol=ncol,
        bbox_to_anchor=bbox,
        fancybox=True,
        shadow=False,
        fontsize=fontsize + 0.2,
        edgecolor="gray",
    )

    ax1.text(
        t_neutrino.mjd - 30,
        5e-14,
        "Neutrino",
        fontsize=BIG_FONTSIZE,
        color="tab:red",
        rotation=90,
    )

    sns.despine(top=False, right=False)

    if not fluxplot:
        outfile_pdf = "at2019fdr_light_curve_mag.pdf"
    else:
        outfile_pdf = "at2019fdr_light_curve_flux.pdf"

    outfile_pdf = os.path.join(PLOT_DIR, outfile_pdf)
    plt.savefig(outfile_pdf)

    percent_forced = (
        100
        / (counts_alert_photometry + counts_forced_photometry)
        * counts_forced_photometry
    )
    print(f"{percent_forced:.2f}% of ZTF datapoints are from forced photometry")
    plt.close()

    detection_df = pd.concat(lcs_per_band, ignore_index=True)
    detection_df.to_csv("/Users/simeon/detections.csv")


def plot_sed(mjd_bounds, title="sed_peak", log=False):
    plt.figure(figsize=(0.8 * FIG_WIDTH, 0.8 * 1 / 1.414 * FIG_WIDTH), dpi=DPI)

    ax1 = plt.subplot(111)

    filter_wl = utilities.load_info_json("filter_wl")

    plt.yscale("log")
    plt.xscale("log")

    ax1.set_ylim([1e-14, 2e-12])
    ax1.set_xlim([1e-1, 1e2])

    d = cosmo.luminosity_distance(REDSHIFT)
    d = d.to(u.cm).value
    lumi = lambda flux: flux * 4 * np.pi * d**2
    flux = lambda lumi: lumi / (4 * np.pi * d**2)
    ax2 = ax1.secondary_yaxis("right", functions=(lumi, flux))
    ax3 = ax1.secondary_xaxis("top", functions=(nu_to_ev, ev_to_nu))

    ax1.set_xlabel("Energy [eV]", fontsize=SMALL_FONTSIZE)
    ax1.set_ylabel(
        "$\nu$ F$_\nu$ [erg s$^{-1}$ cm$^{-2}$]", fontsize=SMALL_FONTSIZE + 2
    )
    ax2.set_ylabel("$\nu$ L$_\nu$ [erg s$^{-1}$]", fontsize=SMALL_FONTSIZE + 2)
    ax3.set_xlabel("Frequency [Hz]", fontsize=SMALL_FONTSIZE + 2)

    for instrband in cmap:
        telescope, band = instrband.split("+")
        if instrband not in BANDS_TO_EXCLUDE:
            lc = df.query(
                f"telescope == '{telescope}' and band == '{band}' and obsmjd >= {mjd_bounds[0]} and obsmjd <= {mjd_bounds[1]}"
            )
            mag_median = np.median(lc.mag)
            mag_err_median = np.median(lc.mag_err)

            flux_density = utilities.abmag_to_flux(mag_median)
            flux_density_err = utilities.abmag_err_to_flux_err(
                np.median(lc.mag), np.median(mag_err_median)
            )

            flux, flux_err = utilities.flux_density_to_flux(
                filter_wl[instrband], flux_density, flux_density_err
            )

            y = flux
            yerr = flux_err
            wl_angstrom = filter_wl[instrband]
            nu = utilities.lambda_to_nu(wl_angstrom)
            ev = nu_to_ev(nu)
            if telescope == "P200_sextractor":
                fmt = "*"
            else:
                fmt = "."
            ax1.errorbar(
                x=ev,
                y=y,
                yerr=yerr,
                color=cmap[instrband],
                marker=fmt,
                markersize=10,
                linestyle=" ",
                label=filterlabel[instrband],
            )

    lc = df_swift_xrt.query(f"obsmjd >= {mjd_bounds[0]} and obsmjd <= {mjd_bounds[1]}")
    flux = np.median(lc["flux_unabsorbed"].values)
    x = nu_to_ev(utilities.lambda_to_nu(filter_wl["Swift+XRT"]))
    y = flux

    xerr = [np.asarray([x - 0.1e3]), np.asarray([x + 10e3])]

    if flabel_sel == "filterlabel_with_wl":
        label = "XRT (0.3–10 keV)"
    else:
        label = "Swift XRT"

    ax1.errorbar(
        x=x,
        xerr=xerr,
        y=flux,
        yerr=flux / 3,
        uplims=True,
        fmt=" ",
        color="darkviolet",
        label=label,
    )

    lc = df_fermi.query(
        f"obsmjd_start <= {mjd_bounds[0]} and obsmjd_end >= {mjd_bounds[1]}"
    )
    flux = np.median(lc["flux"].values)
    x = nu_to_ev(utilities.lambda_to_nu(filter_wl["Fermi+LAT"]))
    y = flux

    xerr = [np.asarray([x - 0.1e9]), np.asarray([x + 700e9])]

    ax1.errorbar(
        x=df_fermi["obsmjd"].values[0],
        xerr=[
            np.asarray([df_fermi["obsmjd"].values[0] - 199.5]),
            np.asarray([df_fermi["obsmjd"].values[0] + 199.5]),
        ],
        y=y,
        yerr=flux / 3,
        uplims=True,
        fmt=" ",
        color="turquoise",
        label="Fermi LAT",
    )

    ax1.tick_params(axis="both", which="major", labelsize=SMALL_FONTSIZE)
    ax2.tick_params(axis="both", which="major", labelsize=SMALL_FONTSIZE)
    ax3.tick_params(axis="both", which="major", labelsize=SMALL_FONTSIZE)
    ax1.legend(fontsize=SMALL_FONTSIZE, ncol=1, framealpha=1, loc="lower right")
    plt.grid(which="major", alpha=0.15)
    plt.tight_layout()
    outpath = f"{title}.png"
    plt.savefig(os.path.join(PLOT_DIR, outpath))
    plt.close()


if __name__ == "__main__":
    FLUXPLOT = True

    REDSHIFT = 0.267
    FIG_WIDTH = 8
    BIG_FONTSIZE = 14
    SMALL_FONTSIZE = 8
    GOLDEN_RATIO = 1.618
    DPI = 400

    CURRENT_FILE_DIR = os.path.dirname(__file__)
    DATA_DIR = os.path.abspath(os.path.join(CURRENT_FILE_DIR, "data"))
    PLOT_DIR = os.path.abspath(CURRENT_FILE_DIR)
    LC_DIR = os.path.join(DATA_DIR, "lightcurves")

    paths = [DATA_DIR, PLOT_DIR, LC_DIR]
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)

    BANDS_TO_EXCLUDE = [
        "P200_sextractor+J",
        "P200_sextractor+H",
        "P200_sextractor+Ks",
        "Swift+V",
        "Swift+B",
    ]

    infile_swift = os.path.join(LC_DIR, "swift_subtracted_synthetic.csv")
    infile_p200 = os.path.join(LC_DIR, "p200_subtracted_synthetic.csv")
    infile_ztf_forced = os.path.join(LC_DIR, "ZTF19aatubsj_SNT_5.0.csv")
    infile_ztf_alert = os.path.join(LC_DIR, "ZTF19aatubsj_alert.csv")
    infile_swift_xrt = os.path.join(LC_DIR, "swift_xrt_ulims_binned.csv")
    infile_fermi = os.path.join(LC_DIR, "fermi_ulims.csv")
    infile_vla = os.path.join(LC_DIR, "vla.csv")
    infile_wise = os.path.join(LC_DIR, "wise_subtracted_baseline.csv")

    outfile_lightcurve = os.path.join(LC_DIR, "full_lightcurve.csv")

    df_swift = pd.read_csv(infile_swift)
    df_p200 = pd.read_csv(infile_p200)
    df_ztf_forced = pd.read_csv(infile_ztf_forced)
    df_ztf_alert = pd.read_csv(infile_ztf_alert)
    df_swift_xrt = pd.read_csv(infile_swift_xrt)
    df_fermi = pd.read_csv(infile_fermi)
    df_vla = pd.read_csv(infile_vla)
    df_wise = pd.read_csv(infile_wise)

    df_ztf_forced = df_ztf_forced[["obsmjd", "mag", "mag_err", "filter"]]
    df_ztf_forced.rename(columns={"filter": "band"}, inplace=True)
    df_ztf_forced["telescope"] = "P48"
    df_ztf_forced["alert"] = False
    df_ztf_forced.query("mag < 99", inplace=True)
    counts_forced_photometry = len(df_ztf_forced)

    df_ztf_alert = df_ztf_alert[["obsmjd", "filter_id", "mag", "mag_err"]]
    df_ztf_alert["telescope"] = "P48"
    df_ztf_alert["alert"] = True
    df_ztf_alert.replace(
        {"filter_id": {1: "ZTF_g", 2: "ZTF_r", 3: "ZTF_i"}}, inplace=True
    )
    df_ztf_alert.rename(columns={"filter_id": "band"}, inplace=True)
    counts_alert_photometry = len(df_ztf_alert)

    df = pd.concat(
        [
            df_p200,
            df_swift,
            df_ztf_forced,
            df_ztf_alert,
            df_wise,
        ],
        ignore_index=True,
    )

    df.reset_index(inplace=True, drop=True)
    df.drop(columns=["Unnamed: 0"], inplace=True)

    cmap = utilities.load_info_json("cmap")
    filterlabel = utilities.load_info_json(flabel_sel)

    plot_lightcurve(df=df, fluxplot=True)

    # Save lightcurve for further use

    # df.to_csv(outfile_lightcurve)
