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

path_not = os.path.join(spectra_folder, "ZTF20abdnpdo_20200612_NOT_v1.dat")
path_snid = os.path.join(spectra_folder, "not_comp0002_snidflux.dat")
# path_tns = os.path.join(spectra_folder, "tns_2019fdr_2019-06-08.33_MDM-2.4_OSMOS.flm")

redshift = 1 + 0.04106

spectrum_not = pd.read_table(path_not, names=["wl", "flux"], sep="\s+", comment="#")
spectrum_snid = pd.read_table(path_snid, names=["wl", "flux"], sep="\s+", comment="#")

# spectrum_not.query("wl > 3900", inplace=True)
# print(spectrum_not)
# spectrum_snid.query("wl > 3900", inplace=True)
# spectrum_tns = pd.read_table(path_tns, names=["wl", "flux", "fluxerr"], sep="\s+", comment='#')
mask = spectrum_not["flux"] > 0.0
spectrum_not["flux"][~mask] = 0.00
spectrum_snid["flux"][~mask] = 0.00


smooth = 6
f = np.array(list(spectrum_not["flux"]))
sf = np.zeros(len(f) - smooth)
swl = np.zeros(len(f) - smooth)

# f_tns = np.array(list(spectrum_tns["flux"]))
# sf_tns = np.zeros(len(f_tns) - smooth)
# swl_tns = np.zeros(len(f_tns) - smooth)

for i in range(smooth):
    sf += np.array(list(f)[i : -smooth + i])
    swl += np.array(list(spectrum_not["wl"])[i : -smooth + i])
    # sf_tns += np.array(list(f_tns)[i:-smooth+i])
    # swl_tns += np.array(list(spectrum_tns["wl"])[i:-smooth+i])

sf /= float(smooth)
swl /= float(smooth)
# sf_tns /= float(smooth)
# swl_tns /= float(smooth)

fig_width = 5.8
golden = 1.62
big_fontsize = 12
annotation_fontsize = 9

plt.figure(figsize=(fig_width, fig_width / golden), dpi=300)
# plt.figure(figsize=())
ax1 = plt.subplot(111)
cols = ["C1", "C7", "k", "k"]
# cols = [":", "--", "-.", "-"]


# offset = 2.4
offset = 1.3

discovery_date = date(2020, 5, 31)
# mdm_date = date(2019, 6, 8)
not_date = date(2020, 6, 12)
# delta_mdm = mdm_date - discovery_date
delta_not = not_date - discovery_date
# days_mdm = delta_mdm.days
days_not = delta_not.days

spectrum_not["flux"] = spectrum_not["flux"] * ((spectrum_not["wl"]))
spectrum_not["flux"] = spectrum_not["flux"] / (np.mean(spectrum_not["flux"]))

spectrum_not.query("wl > 4050", inplace=True)
spectrum_snid.query("wl > 4050", inplace=True)

# now we smooth
not_smoothed = savgol_filter(spectrum_not["flux"], 51, 3)

plt.plot(
    spectrum_snid["wl"] / redshift,
    spectrum_snid["flux"] + offset,
    color="red",
    alpha=1,
    label="SN1994l (@14 days)",
)
plt.plot(
    spectrum_not["wl"] / redshift,
    spectrum_not["flux"],
    # not_smoothed,
    linewidth=0.5,
    color="C0",
    alpha=0.3,
    # label="SN2020lls",
)
plt.plot(
    spectrum_not["wl"] / redshift,
    # spectrum_not["flux"] + offset,
    not_smoothed,
    linewidth=1,
    color="C0",
    alpha=1,
    label="SN2020lls (smoothed)",
)


bbox = dict(boxstyle="circle", fc="white", ec="k")

plt.annotate(
    f"NOT (+{days_not} days)", (5450, 3.6), fontsize=big_fontsize, color="black"
)
# plt.annotate(f"MDM-2.4 (+{days_mdm} days)", (5400 , 1.2), fontsize=big_fontsize, color="dimgrey")

plt.ylabel(r"$F_{\lambda}$ (a.u.)", fontsize=big_fontsize)
ax1.set_xlim([3800, 9450])
ax1b = ax1.twiny()
rslim = ax1.get_xlim()
ax1b.set_xlim((rslim[0] * redshift, rslim[1] * redshift))
ax1.set_xlabel(r"Rest wavelength ($\rm \AA$)", fontsize=big_fontsize)
ax1b.set_xlabel(rf"Observed Wavelength (z={redshift-1.:.3f})", fontsize=big_fontsize)
ax1.tick_params(axis="both", which="major", labelsize=big_fontsize)
ax1b.tick_params(axis="both", which="major", labelsize=big_fontsize)
ax1.legend()

filename = "ZTF20abdnpdo_not_spectrum.pdf"

plt.tight_layout()

plt.savefig(filename)
