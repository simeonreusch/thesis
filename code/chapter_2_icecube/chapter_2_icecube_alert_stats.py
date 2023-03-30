#!/usr/bin/env python
# coding: utf-8

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import rc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc("font", **{"family": "serif", "serif": ["Palatino"]})
# rc("text", usetex=True)


def plot_overview():
    """
    Plot pie chart for observation rejection reasons
    """
    counts = [63, 36, 7]
    counts = np.asarray(counts)
    labels = ["Bronze (63)", "Gold (36)", "Retracted (7)"]
    colors = ["darkgoldenrod", "gold", "red"]

    explode = np.empty(len(counts))
    explode.fill(0.04)
    # explode[0] = 0.1

    # palette_color = sns.color_palette("dark")

    fig, ax = plt.subplots(figsize=(width := 3.5, width / 1.62))

    patches, texts, pcts = ax.pie(
        counts,
        labels=labels,
        explode=explode,
        colors=colors,
        startangle=90,
        labeldistance=1.13,
        autopct="%.0f%%",
        pctdistance=0.7,
    )
    for i, patch in enumerate(patches):
        texts[i].set_color(patch.get_facecolor())
    plt.setp(pcts, color="white", fontweight="bold", fontsize=10)
    plt.setp(texts, fontweight="bold")
    plt.tight_layout()

    outfile = "ic_he_alert_overview.pdf"

    plt.savefig(outfile)
    plt.close()


plot_overview()
