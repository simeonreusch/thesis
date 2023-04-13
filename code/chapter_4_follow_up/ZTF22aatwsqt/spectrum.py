#!/usr/bin/env python3
# Author: Simeon Reusch (simeon.reusch@desy.de)
# License: BSD-3-Clause
import os
from datetime import date

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table
from astropy.time import Time
from matplotlib import rc
from scipy.signal import savgol_filter

rc("font", **{"family": "serif", "serif": ["Palatino"]})
matplotlib.rcParams["mathtext.fontset"] = "custom"
matplotlib.rcParams["mathtext.rm"] = "Palatino"
matplotlib.rcParams["mathtext.it"] = "Palatino: italic"
matplotlib.rcParams["mathtext.bf"] = "Palatino:bold"
data_folder = "data"
plot_folder = "plots"
spectra_folder = os.path.join(data_folder, "spectra")
lc_folder = os.path.join(data_folder, "lightcurves")

path_obs = os.path.join(spectra_folder, "spec_for_snid.txt")
path_snid = os.path.join(spectra_folder, "snid_comp.txt")

redshift = 1 + 0.135
snid_redshift = 1 + 0.1321  # + 0.002

spectrum_obs = pd.read_table(path_obs, names=["wl", "flux"], sep="\s+", comment="#")
spectrum_snid = pd.read_table(path_snid, sep=r"\s+", comment="#", header=None)
spectrum_snid.rename(columns={0: "wl", 1: "flux"}, inplace=True)

mask = spectrum_obs["flux"] > 0.0

spectrum_obs["flux"][~mask] = 0.00
spectrum_snid["flux"][~mask] = 0.00
spectrum_snid["wl"] = spectrum_snid["wl"]


fig_width = 5.8
golden = 1.62
big_fontsize = 12
annotation_fontsize = 9

plt.figure(figsize=(fig_width, fig_width / golden), dpi=300)
ax1 = plt.subplot(111)
cols = ["C1", "C7", "k", "k"]


offset = 1.5


spectrum_snid["flux"] = spectrum_snid["flux"] * ((spectrum_snid["wl"]))
spectrum_snid["flux"] = spectrum_snid["flux"] / (np.mean(spectrum_snid["flux"] * 1.4))


spectrum_obs["flux"] = spectrum_obs["flux"] * ((spectrum_obs["wl"]))
spectrum_obs["flux"] = spectrum_obs["flux"] / np.mean(spectrum_obs["flux"])

spectrum_snid["wl"] = spectrum_snid["wl"] / snid_redshift
spectrum_obs["wl"] = spectrum_obs["wl"] / redshift

spectrum_obs.query("3050 < wl < 8950", inplace=True)
spectrum_snid.query("3050 < wl < 8950", inplace=True)

# now we smooth
obs_smoothed = savgol_filter(spectrum_obs["flux"], 51, 3)
snid_smoothed = savgol_filter(spectrum_snid["flux"], 51, 11)


plt.plot(
    spectrum_snid["wl"],
    snid_smoothed + offset,
    # spectrum _snid["flux"] + offset,
    color="red",
    alpha=1,
    label="SN2005cf @47 days)",
)
plt.plot(
    spectrum_obs["wl"],
    spectrum_obs["flux"],
    linewidth=0.5,
    color="C0",
    alpha=0.3,
)
plt.plot(
    spectrum_obs["wl"],
    obs_smoothed,
    linewidth=1,
    color="C0",
    alpha=1,
    label="AT2022oyn (smoothed)",
)

# balmer_lines = {r"$H_\alpha$": 6563, r"$H_\beta$": 4861, r"$H_\gamma$": 4340}

# for linename, value in balmer_lines.items():
#     plt.axvline(value, color="black", ls="dotted")
#     ax1.text(value - 180, 0.4, linename, fontsize=10, rotation=90, color="black")

bbox = dict(boxstyle="circle", fc="white", ec="k")

plt.ylabel(r"$F_{\lambda}$ (a.u.)", fontsize=big_fontsize)
ax1.set_xlim([3000, 9000])
ax1.set_ylim([0, 4.5])

ax1b = ax1.twiny()
rslim = ax1.get_xlim()
ax1b.set_xlim((rslim[0] * redshift, rslim[1] * redshift))
ax1.set_xlabel(r"Rest wavelength ($\rm \AA$)", fontsize=big_fontsize)
ax1b.set_xlabel(rf"Observed Wavelength (z={redshift-1.:.3f})", fontsize=big_fontsize)
ax1.tick_params(axis="both", which="major", labelsize=big_fontsize)
ax1b.tick_params(axis="both", which="major", labelsize=big_fontsize)
ax1.legend()

filename = "ZTF22aatwsqt_spectrum.pdf"

plt.tight_layout()

plt.savefig(filename)
