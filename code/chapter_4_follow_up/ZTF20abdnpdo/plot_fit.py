import json
import logging
import os
import pickle
import warnings
from collections import OrderedDict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from astropy.cosmology import Planck15 as cosmo
from astropy.table import Table
from matplotlib import rc
from reduce_mosfit_result_file import make_reduced_output

rc("font", **{"family": "serif", "serif": ["Palatino"]})

this_dir = os.path.dirname(os.path.realpath(__file__))
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def bandcolors(band):
    color_dict = {"ztfg": "g", "ztfr": "r", "ztfi": "orange", "desi": "orange"}

    if not band:
        return "grey"
    else:
        if band in color_dict.keys():
            return color_dict[band]
        else:
            for key in color_dict.keys():
                if band in key:
                    return color_dict[key]

    raise Exception(f"No color specified for band {band}, type(band)={type(band)}!")


def mosfit_plot(file, ax=None, fig=None, ylim=[20, 20], reduce_data=False):
    if not ax and not fig:
        logger.debug("neither axes nor figure given")
        fig, ax = plt.subplots()
        plt.gca().invert_yaxis()
    elif fig and not ax:
        logger.debug("only figure given")
        ax = fig.add_subplot()

    # sns.reset_orig()

    logger.debug(f"type of file is {type(file)}")
    if type(file) is str:
        logger.debug(f"file is {file}.")
        if not os.path.isfile(file):
            logger.warning("Not a file path!")

    if type(file) is str and os.path.isfile(file):
        with open(file, "r") as f:
            data = json.loads(f.read())
            if "name" not in data:
                data = data[list(data.keys())[0]]
    else:
        data = file

    if reduce_data:
        data = make_reduced_output(data)

    photo = data["photometry"]
    model = data["models"][0]
    real_data = (
        len(
            [
                x
                for x in photo
                if "band" in x
                and "magnitude" in x
                and ("realization" not in x or "simulated" in x)
            ]
        )
        > 0
    )
    ci_data = [x for x in photo if "band" in x and "confidence_level" in x]
    band_attr = ["band", "instrument", "telescope", "system", "bandset"]
    band_list = list(
        set(
            [
                tuple(x.get(y, "") for y in band_attr)
                for x in photo
                if "band" in x and "magnitude" in x
            ]
        )
    )
    real_band_list = list(
        set(
            [
                tuple(x.get(y, "") for y in band_attr)
                for x in photo
                if "band" in x
                and "magnitude" in x
                and ("realization" not in x or "simulated" in x)
            ]
        )
    )

    confidence_intervals = {}
    for x in ci_data:
        if x["band"] not in confidence_intervals.keys():
            confidence_intervals[x["band"]] = [[], [], []]
        confidence_intervals[x["band"]][0].append(float(x["time"]))
        confidence_intervals[x["band"]][1].append(float(x["confidence_interval_upper"]))
        confidence_intervals[x["band"]][2].append(float(x["confidence_interval_lower"]))

    used_bands = []
    for full_band in band_list:
        (band, inst, tele, syst, bset) = full_band

        logger.debug(f"plotting {band}")

        extra_nice = ", ".join(
            list(filter(None, OrderedDict.fromkeys((inst, syst, bset)).keys()))
        )
        nice_name = band + ((" [" + extra_nice + "]") if extra_nice else "")

        realizations = [[] for x in range(len(model["realizations"]))]
        for ph in photo:
            rn = ph.get("realization", None)
            ci = ph.get("confidence_interval", False)
            si = ph.get("simulated", False)
            if rn and not si and not ci:
                if tuple(ph.get(y, "") for y in band_attr) == full_band:
                    realizations[int(rn) - 1].append(
                        (
                            float(ph["time"]),
                            float(ph["magnitude"]),
                            [
                                float(
                                    ph.get(
                                        "e_lower_magnitude", ph.get("e_magnitude", 0.0)
                                    )
                                ),
                                float(
                                    ph.get(
                                        "e_upper_magnitude", ph.get("e_magnitude", 0.0)
                                    )
                                ),
                            ],
                            ph.get("upperlimit"),
                        )
                    )
        numrz = np.sum([1 for x in realizations if len(x)])

        if band in confidence_intervals.keys():
            logger.debug("plotting confidence intervals")
            ci = confidence_intervals[band]
            label = (
                ""
                if full_band in used_bands or full_band in real_band_list
                else nice_name
            )
            ax.fill_between(
                ci[0],
                ci[1],
                ci[2],
                color=bandcolors(band),
                edgecolor=bandcolors(band),
                alpha=0.2,
                label=label,
            )
            if label:
                used_bands = list(set(used_bands + [full_band]))

        rz_mask = [False if not len(rz) else True for rz in realizations]

        if np.any(rz_mask):
            logger.debug("plotting individual realizations")
            for rz in np.array(realizations)[rz_mask]:
                xs, ys, vs, us = zip(*rz)
                label = (
                    ""
                    if full_band in used_bands or full_band in real_band_list
                    else nice_name
                )
                if max(vs) == 0.0:
                    ax.plot(
                        xs,
                        ys,
                        color=bandcolors(band),
                        label=label,
                        linewidth=0.5,
                        alpha=0.1,
                    )
                else:
                    xs = np.array(xs)
                    ymi = np.array(ys) - np.array(
                        [np.inf if u else v[0] for v, u in zip(vs, us)]
                    )
                    yma = np.array(ys) + np.array([v[1] for v in vs])
                    ax.fill_between(
                        xs,
                        ymi,
                        yma,
                        color=bandcolors(band),
                        edgecolor=None,
                        label=label,
                        alpha=1.0 / numrz,
                        linewidth=0.0,
                    )
                    ax.plot(
                        xs,
                        ys,
                        color=bandcolors(band),
                        label=label,
                        alpha=0.1,
                        linewidth=0.5,
                    )
                if label:
                    used_bands = list(set(used_bands + [full_band]))

    marshal_file = os.path.join(lc_dir, "marshal.txt")
    fp_file = os.path.join(lc_dir, "fp.csv")

    dirs = [data_dir, plot_dir, spectra_dir, lc_dir]
    for directory in dirs:
        if not os.path.exists(directory):
            os.makedirs(directory)

    marshal_lc = pd.read_table(
        marshal_file,
        names=[
            "date",
            "jdobs",
            "filter",
            "absmag",
            "magpsf",
            "sigmamagpsf",
            "limmag",
            "instrument",
            "programid",
            "reducedby",
            "refsys",
            "issub",
            "isdiffpos",
        ],
        sep=",",
        comment="#",
    )

    marshal_lc["mjd"] = marshal_lc["jdobs"] - 2400000.5
    marshal_lc = marshal_lc.query(f"mjd > {MJD_CUT}")
    marshal_lc = marshal_lc.drop(
        columns=[
            "date",
            "jdobs",
            "absmag",
            "programid",
            "reducedby",
            "refsys",
            "issub",
            "isdiffpos",
        ]
    )
    marshal_lc["filter"] = marshal_lc["filter"] + marshal_lc["instrument"]

    fp_lc = pd.read_csv(fp_file)
    fp_lc["mjd"] = fp_lc["obsmjd"]
    fp_lc["magpsf"] = fp_lc["mag"]
    fp_lc["sigmamagpsf"] = fp_lc["mag_err"]
    fp_lc["filter"].replace(
        {"ZTF_g": "gP48+ZTF", "ZTF_r": "rP48+ZTF", "ZTF_i": "iP48+ZTF"}, inplace=True
    )

    ax.set_ylim(ylim)
    df = marshal_lc

    cmap = {
        "gP48+ZTF": "g",
        "rP48+ZTF": "r",
        "rP60+SEDM": "r",
        "iP48+ZTF": "orange",
    }

    p60_list = ["gP60+SEDM", "rP60+SEDM", "iP60+SEDM"]

    filterlabel = {
        "gP48+ZTF": "ZTF g",
        "gP60+SEDM": "P60 g",
        "rP48+ZTF": "ZTF r",
        "rP60+SEDM": "P60 r",
        "iP48+ZTF": "ZTF i",
        "iP60+SEDM": "P60 i",
        "V": "$\it{Swift}$ V",
        "UVW2": "$\it{Swift}$ UVW2",
        "UVW1": "$\it{Swift}$ UVW1",
        "UVM2": "$\it{Swift}$ UVM2",
        "u": "$\it{Swift}$ U",
        "B": "$\it{Swift}$ B",
    }
    uplim_df = df.query("magpsf == 99")
    df = df.query("magpsf < 99")
    fp_lc = fp_lc.query("magpsf < 99")

    for f in cmap:
        if f not in cmap.keys():
            continue
        lc = df[df["filter"] == f]
        lc_forced = fp_lc[fp_lc["filter"] == f]
        # print(lc["mjd"])
        # print(lc_forced["mjd"])
        lc["mjd_rounded"] = np.around(lc["mjd"].values, decimals=3)
        lc_forced["mjd_rounded"] = np.around(lc_forced["mjd"].values, decimals=3)

        lc = lc[~lc["mjd_rounded"].isin(lc_forced["mjd_rounded"])]

        uplim = uplim_df[uplim_df["filter"] == f]

        if f == "rP60+SEDM":
            ax.errorbar(
                lc["mjd"],
                lc["magpsf"],
                yerr=lc["sigmamagpsf"],
                color=cmap[f],
                marker="s",
                linestyle=" ",
                label=filterlabel[f],
                mec="black",
                mew=1,
                markersize=4,
            )
        else:
            ax.errorbar(
                lc["mjd"],
                lc["magpsf"],
                yerr=lc["sigmamagpsf"],
                color=cmap[f],
                marker=".",
                linestyle=" ",
                label=filterlabel[f],
                mec="black",
                mew=1,
                markersize=8,
            )
        ax.errorbar(
            lc_forced["mjd"],
            lc_forced["magpsf"],
            yerr=lc_forced["sigmamagpsf"],
            color=cmap[f],
            marker=".",
            linestyle=" ",
            mec="black",
            mew=1,
            markersize=8,
        )
        ax.scatter(
            uplim["mjd"], uplim["limmag"], color=cmap[f], marker="v", s=4.5, alpha=1
        )

    ax.invert_yaxis()
    ax.set_ylabel(r"Apparent Magnitude (AB)", fontsize=BIG_FONTSIZE + 2)
    ax.tick_params(axis="both", which="major", labelsize=BIG_FONTSIZE)
    ax.set_xlabel("Date (MJD)", fontsize=BIG_FONTSIZE + 2)
    ax.grid(axis="y")
    ax.set_ylim([22, 18.5])
    ax.set_xlim(left=MJD_CUT, right=59018)
    absmag = lambda mag: mag - cosmo.distmod(REDSHIFT).value
    mag = lambda absmag: absmag + cosmo.distmod(REDSHIFT).value
    ax2 = ax.secondary_yaxis("right", functions=(absmag, mag))
    ax2.tick_params(axis="both", which="major", labelsize=BIG_FONTSIZE)
    ax2.set_ylabel(rf"Absolute Magnitude (z={REDSHIFT:.3f})", fontsize=BIG_FONTSIZE + 2)

    ax.axvline(t_neutrino, linestyle=":", label="IC200530A")

    return fig, ax


def plot_fit(fig, ax, filename):
    logger.debug(f"file is {filename}")
    pickle_file = filename[:-4] + "pkl"

    with open(pickle_file, "rb") as f:
        collected_data = pickle.load(f)

    t_exp_fit = collected_data[0]["t_exp_fit"]
    t_exp_ic = collected_data[0]["t_exp_dif_0.9"]

    fig, ax = mosfit_plot(filename, ax, fig, reduce_data=True)
    ax.axvline(t_exp_fit, color="black", label="fitted $t_{{0}}:$" + f"{t_exp_fit:.2f}")
    ax.fill_between(
        t_exp_ic,
        y1=30,
        color="black",
        alpha=0.1,
        label=r"$t_{{0,90\%}}:$"
        + f"[-{t_exp_fit-t_exp_ic[0]:.2f},+{-t_exp_fit+t_exp_ic[1]:.2f}]",
    )
    ax.legend(fontsize=BIG_FONTSIZE, ncol=1, framealpha=0.8)
    return fig, ax, t_exp_fit


if __name__ == "__main__":
    FIG_WIDTH = 6
    BIG_FONTSIZE = 10
    REDSHIFT = 0.04106
    MJD_CUT = 58985
    t_neutrino = 58999.329507291666

    import glob, os

    base_dir = os.getcwd()
    mosfit_file = os.path.join(base_dir, "fit_results", "mosfit.json")
    data_dir = os.path.join(base_dir, "data")
    spectra_dir = os.path.join(data_dir, "spectra")
    lc_dir = os.path.join(data_dir, "lightcurves")
    plot_dir = base_dir

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_WIDTH / 1.61), dpi=300)
    fig, ax, t_exp_fit = plot_fit(fig, ax, filename=mosfit_file)
    plt.tight_layout()
    fig.savefig(os.path.join(plot_dir, f"ZTF20abdnpdo_mosfit.pdf"))
    plt.close()
