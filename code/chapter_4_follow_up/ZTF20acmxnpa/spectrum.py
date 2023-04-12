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

path_not = os.path.join(spectra_folder, "combined.dat")

redshift = 1 + 0.0865

spectrum_obs = pd.read_table(path_not, names=["wl", "flux"], sep="\s+", comment="#")

mask = spectrum_obs["flux"] > 0.0
spectrum_obs["flux"][~mask] = 0.00

smooth = 4
f = np.array(list(spectrum_obs["flux"]))
sf = np.zeros(len(f) - smooth)
swl = np.zeros(len(f) - smooth)

for i in range(smooth):
    sf += np.array(list(f)[i : -smooth + i])
    swl += np.array(list(spectrum_obs["wl"])[i : -smooth + i])

sf /= float(smooth)
swl /= float(smooth)

fig_width = 5.8
golden = 1.62
big_fontsize = 12
annotation_fontsize = 9

plt.figure(figsize=(fig_width, fig_width / golden), dpi=300)
ax1 = plt.subplot(111)
cols = ["C1", "C7", "k", "k"]


offset = 1.1

spectrum_obs["flux"] = spectrum_obs["flux"] * ((spectrum_obs["wl"]))
spectrum_obs["flux"] = spectrum_obs["flux"] / np.mean(spectrum_obs["flux"])

spectrum_obs["wl"] = spectrum_obs["wl"] / redshift

spectrum_obs.query("8950>wl > 5000", inplace=True)
# now we smooth
not_smoothed = savgol_filter(spectrum_obs["flux"], 51, 3)


plt.plot(
    spectrum_obs["wl"],
    spectrum_obs["flux"],
    linewidth=0.5,
    color="C0",
    alpha=0.3,
)
plt.plot(
    spectrum_obs["wl"],
    not_smoothed,
    linewidth=1,
    color="C0",
    alpha=1,
    label="AT2020ybb\n(smoothed)",
)

bbox = dict(boxstyle="circle", fc="white", ec="k")

plt.ylabel(r"$F_{\lambda}$ (a.u.)", fontsize=big_fontsize)
ax1.set_xlim([4950, 9000])
ax1.set_ylim([0.5, 1.7])
ax1b = ax1.twiny()
rslim = ax1.get_xlim()
ax1b.set_xlim((rslim[0] * redshift, rslim[1] * redshift))
ax1.set_xlabel(r"Rest wavelength ($\rm \AA$)", fontsize=big_fontsize)
ax1b.set_xlabel(rf"Observed Wavelength (z={redshift-1.:.3f})", fontsize=big_fontsize)
ax1.tick_params(axis="both", which="major", labelsize=big_fontsize)
ax1b.tick_params(axis="both", which="major", labelsize=big_fontsize)
ax1.legend(fontsize=11)

for telluric in [[6860, 6890], [7600, 7630]]:
    ax1b.axvspan(telluric[0], telluric[1], color="gray", alpha=0.4, ec=None)

balmer_lines = {"$H_\alpha$": 6563, "hbeta": 4861, "hgamma": 4340}
sii_line = {"SII": 6715}

ax1.text(6200, 0.7, "Telluric", fontsize=10, rotation=90, color="gray")
ax1.text(6430, 0.7, r"$H_\alpha$", fontsize=10, rotation=90, color="black")
ax1.text(6600, 0.7, r"SII", fontsize=10, rotation=90, color="red")
ax1.text(6880, 0.7, "Telluric", fontsize=10, rotation=90, color="gray")

for linename, value in balmer_lines.items():
    ax1.axvline(value, color="black", ls="dotted")
for linename, value in sii_line.items():
    ax1.axvline(value, color="red", ls="dashdot", lw=0.8)

filename = "ZTF20acmxnpa_spectrum.pdf"

plt.tight_layout()

plt.savefig(filename)
