#!/usr/bin/env python3
# Author: Simeon Reusch (simeon.reusch@desy.de)
# License: BSD-3-Clause
import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import rc

rc("font", **{"family": "serif", "serif": ["Palatino"]})

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def classify(classes: list[str]) -> list[str]:
    """
    Sort into broader classes
    """
    classes_out = []
    broad_classes = {
        "AGN Variability": "AGN Variability",
        "AGN Variability?": "AGN Variability",
        "AGN Variability (FP)": "AGN Variability",
        "AGN?": "AGN Flare",
        "AGN": "AGN Flare",
        "G-type source": "AGN Variability",
        "QSO": "AGN Variability",
        "SNIa": "Transient",
        "SN Ia": "Transient",
        "SN Ic": "Transient",
        "SN II/IIb": "Transient",
        "SN IIP": "Transient",
        "SN II": "Transient",
        "Dwarf Nova": "Transient",
        "Star": "Star",
        "Star?": "Star",
        "TDE": "Transient",
        "Artifact": "Artifact",
        "Unclassified": "Unclassified",
        "???": "Unclassified",
    }

    for c in classes:
        classes_out.append(broad_classes[c])

    return classes_out


def plot_broad_classes(df: pd.DataFrame) -> None:
    """
    Plot the broad classes in a pie chart
    """
    counts = []
    labels = []
    for c in df.classif_broad.unique():
        counts.append(n := len(df.query("classif_broad == @c")))
        if c == "AGN Variability":
            c = "AGN\nVariability"
        labels.append(f"{c} ({n})")

    indices = list(np.argsort(counts))
    indices.reverse()
    counts = np.asarray(counts)[indices]
    labels = np.asarray(labels)[indices]

    explode = np.empty(len(counts))
    explode.fill(0.01)
    explode[4] = 0.1

    fig, ax = plt.subplots(figsize=(width := 5, width / 1.62))

    patches, texts, pcts = ax.pie(
        counts,
        labels=labels,
        explode=explode,
        startangle=90,
        labeldistance=1.13,
        autopct="%.0f%%",
        pctdistance=0.7,
    )
    for i, patch in enumerate(patches):
        texts[i].set_color(patch.get_facecolor())

    plt.setp(
        pcts,
        color="white",
        fontsize=10,
        fontweight="bold",
    )
    plt.setp(texts, fontweight="bold")
    plt.tight_layout()

    outfile_path = "classification_overview.pdf"

    plt.savefig(outfile_path)
    plt.close()


if __name__ == "__main__":
    infile_path = "fu_overview.csv"

    df = pd.read_csv(infile_path)
    no_gcned = np.sum(df["in_gcn"].values)
    # print(df["classif"].unique())
    logger.info(f"Transients published via GCN: {no_gcned}")

    df["classif_broad"] = classify(df["classif"].values)

    plot_broad_classes(df)
