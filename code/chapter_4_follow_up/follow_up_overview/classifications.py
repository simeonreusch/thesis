#!/usr/bin/env python3
# Author: Simeon Reusch (simeon.reusch@desy.de)
# License: BSD-3-Clause
import os

import numpy as np
import pandas as pd

infile = "fu_overview.csv"

df = pd.read_csv(infile)


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


no_gcned = np.sum(df["in_gcn"].values)
print(df["classif"].unique())

df["classif_broad"] = classify(df["classif"].values)

print(df)
