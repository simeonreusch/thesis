#!/usr/bin/env python3
# Author: Simeon Reusch (simeon.reusch@desy.de)
# License: BSD-3-Clause

import argparse
import json
import logging
import os
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rc
from numpy.polynomial import polynomial as P

rc("font", **{"family": "serif", "serif": ["Palatino"]})
matplotlib.rcParams["mathtext.fontset"] = "custom"
matplotlib.rcParams["mathtext.rm"] = "Palatino"
matplotlib.rcParams["mathtext.it"] = "Palatino: italic"
matplotlib.rcParams["mathtext.bf"] = "Palatino:bold"

epoch = "epoch1"

infile = os.path.join("data", f"{epoch}.csv")

df = pd.read_csv(infile)

pretty_param = {"av": r"$A_V$", "temp1": r"$T_1$ (K)"}


for param in ["temp1"]:  # , "temp2", "scale1", "scale2"]:
    plt.figure(figsize=(width := 5, width / 1.4))

    df_temp = df[[param, f"{param}_chisq"]].dropna()

    x = df_temp[param]
    y = df_temp[f"{param}_chisq"]

    best_fit_chisq = np.min(y)

    if epoch == "epoch2":
        if param == "scale1":
            deg = 4
        elif param == "temp1":
            deg = 7
        elif param == "scale2":
            deg = 7
        else:
            deg = 4

    if epoch == "epoch1":
        if param == "scale2":
            deg = 4
        else:
            deg = 4

    if epoch == "epoch0":
        if param == "scale2":
            deg = 9
        else:
            deg = 4

    model = np.poly1d(np.polyfit(x, y, deg))
    x_model = np.linspace(np.min(x.values), np.max(x.values), 1000)

    sigma_line = np.full(shape=1000, fill_value=(1 + best_fit_chisq), dtype=float)

    intersection = np.argwhere(np.diff(np.sign(sigma_line - model(x_model)))).flatten()

    print(f"{param} min: {x_model[intersection[0]]:.6e}")
    print(f"{param} max: {x_model[intersection[1]]:.6e}")

    ax1 = plt.subplot(111)

    ax1.scatter(x, y)
    ax1.plot(x_model, model(x_model))
    ax1.set_xlabel(pretty_param[param], fontsize=12)
    ax1.set_ylabel(r"$\chi2$", fontsize=12)
    ax1.axhline(1 + best_fit_chisq, color="black")
    ax1.plot(x_model[intersection], sigma_line[intersection], "ro")
    plt.savefig(f"{epoch}_{param}.pdf")

    # plt.tight_layout()
