#!/usr/bin/env python3
# Author: Simeon Reusch (simeon.reusch@desy.de)
# License: BSD-3-Clause

import argparse
import json
import logging
import os
import time

import astropy.units as u
import lmfit
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy import constants as const
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM
from astropy.modeling.models import BlackBody
from dust_model import plot_results_brute
from extinction import apply, calzetti00, ccm89, remove
from lmfit import Minimizer, Model, Parameters, minimize, report_fit
from matplotlib import rc
from modelSED import fit, sncosmo_spectral_v13, utilities
from modelSED.utilities import (
    FNU,
    broken_powerlaw_spectrum,
    calculate_bolometric_luminosity,
    flux_nu_to_lambda,
)

rc("font", **{"family": "serif", "serif": ["Palatino"]})

WITH_WISE = True

t_vals = np.linspace(700, 3000, num=40)
r_vals = np.linspace(1e17, 8e17, num=40)
total_length = len(t_vals) * len(r_vals)


plt.figure(figsize=(4.5, 4))
ax1 = plt.subplot(111)
fontsize = 14

plt.xscale("log")
ax1.set_xlabel("Radius (cm)", fontsize=fontsize)
ax1.set_ylabel("Temperature (K)", fontsize=fontsize)

ax1.tick_params(axis="x", labelsize=fontsize - 3, which="both")
ax1.tick_params(axis="y", labelsize=fontsize - 3)

labels = ["epoch 1", "epoch 2", "epoch 3"]

delta_chisq = 4.605
clrs = ["deeppink", "red", "darkred"]


for epoch in [0, 1, 2]:
    if epoch == 1:
        if WITH_WISE:
            df = pd.read_csv(f"data/epoch1free_combvals.csv").drop(columns="Unnamed: 0")
        else:
            df = pd.read_csv(f"data/epoch1_nowise_free_combvals.csv").drop(
                columns="Unnamed: 0"
            )
    else:
        df = pd.read_csv(f"data/epoch{epoch}free_combvals.csv").drop(
            columns="Unnamed: 0"
        )

    minchisq = min(df.chisq.values)

    df = df.sort_values(by=["temp", "radius"])

    x = df.temp.unique()
    y = df.radius.unique()
    z = np.empty([len(x), len(y)])

    for i, t in enumerate(x):
        df_temp = df.query("temp == @t")
        ch = df_temp.chisq.values
        for j, c in enumerate(ch):
            z[i, j] = c

    levels = [minchisq, minchisq + delta_chisq]

    ax1.contour(
        y,
        x,
        z,
        norm=matplotlib.colors.LogNorm(),
        levels=levels,
        colors=("blue", clrs[epoch]),
    )
    ax1.plot([3e17], [2000], label=labels[epoch], color=clrs[epoch])

    ax1.axhline(1850, color="black", ls="dotted", alpha=0.2)

    ax1.text(
        4e17,
        1700,
        "1850 K",
        color="black",
        alpha=0.2,
        fontsize=fontsize,
    )
ax1.set_xlim((1e17, 6e17))

plt.legend(fontsize=fontsize)
plt.tight_layout()

if WITH_WISE:
    outfile = "at2019fdr_contour_all.pdf"
else:
    outfile = "at2019fdr_contour_all_no_WISE.pdf"

plt.savefig(outfile)
