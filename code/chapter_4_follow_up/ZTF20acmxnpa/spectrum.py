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

redshift = 1 + 0.0861

spectrum_not = pd.read_table(path_not, names=["wl", "flux"], sep="\s+", comment="#")

mask = spectrum_not["flux"] > 0.0
spectrum_not["flux"][~mask] = 0.00

smooth = 4
f = np.array(list(spectrum_not["flux"]))
sf = np.zeros(len(f) - smooth)
swl = np.zeros(len(f) - smooth)

for i in range(smooth):
    sf += np.array(list(f)[i : -smooth + i])
    swl += np.array(list(spectrum_not["wl"])[i : -smooth + i])

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

spectrum_not["flux"] = spectrum_not["flux"] * ((spectrum_not["wl"]))
spectrum_not["flux"] = spectrum_not["flux"] / np.mean(spectrum_not["flux"])

spectrum_not["wl"] = spectrum_not["wl"] / redshift

spectrum_not.query("9000>wl > 5000", inplace=True)
# now we smooth
not_smoothed = savgol_filter(spectrum_not["flux"], 51, 3)


plt.plot(
    spectrum_not["wl"],
    spectrum_not["flux"],
    linewidth=0.5,
    color="C0",
    alpha=0.3,
)
plt.plot(
    spectrum_not["wl"],
    not_smoothed,
    linewidth=1,
    color="C0",
    alpha=1,
    label="AT2020ybb (smoothed)",
)

# balmer_lines = {"$H_\alpha$": 6563, "hbeta": 4861, "hgamma": 4340}

# for linename, value in balmer_lines.items():
# plt.axvline(value, color="black", ls="dotted")


bbox = dict(boxstyle="circle", fc="white", ec="k")

plt.ylabel(r"$F_{\lambda}$ (a.u.)", fontsize=big_fontsize)
ax1.set_xlim([4950, 9050])
ax1b = ax1.twiny()
rslim = ax1.get_xlim()
ax1b.set_xlim((rslim[0] * redshift, rslim[1] * redshift))
ax1.set_xlabel(r"Rest wavelength ($\rm \AA$)", fontsize=big_fontsize)
ax1b.set_xlabel(rf"Observed Wavelength (z={redshift-1.:.3f})", fontsize=big_fontsize)
ax1.tick_params(axis="both", which="major", labelsize=big_fontsize)
ax1b.tick_params(axis="both", which="major", labelsize=big_fontsize)
ax1.legend()

for telluric in [6875, 7615]:
    ax1b.axvline(telluric, color="black", ls="dashdot")


filename = "AT2020ybb_spectrum.pdf"

plt.tight_layout()

plt.savefig(filename)
