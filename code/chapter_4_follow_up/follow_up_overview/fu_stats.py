#!/usr/bin/env python
# coding: utf-8

import datetime
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb
from matplotlib import rc

rc("font", **{"family": "serif", "serif": ["Palatino"]})
# sb.set(font="Palatino")


fu_fn = os.path.abspath("neutrino_too_followup - OVERVIEW_FU.csv")
not_fu_fn = os.path.abspath("neutrino_too_followup - OVERVIEW_NOT_FU.csv")


def make_plot():
    fu = pd.read_csv(fu_fn, skipfooter=2)
    not_fu = pd.read_csv(not_fu_fn)

    pre_new_alertstream = ["HESE", "EHE", "HESE+EHE"]
    fu.query("Class not in @pre_new_alertstream", inplace=True)
    not_fu.query("Class not in @pre_new_alertstream", inplace=True)
    not_fu.query("Code != 1", inplace=True)

    not_fu["dates"] = [
        datetime.datetime.strptime(ev[2:-1], "%y%m%d").date() for ev in not_fu.Event
    ]
    fu["dates"] = [
        datetime.datetime.strptime(ev[2:-1], "%y%m%d").date() for ev in fu.Event.iloc
    ]
    not_fu["maintenance"] = not_fu["Code"] == 3

    not_fu["rejected"] = not_fu["Code"] == 4

    not_fu_other = not_fu.query("Code not in [3,4]")

    bins = np.array(
        [
            datetime.datetime.strptime(f"{y}0{m}01", "%y%m%d").date()
            for y in range(19, 24)
            for m in [1, 7]
        ]
    )

    bindif = bins[1:] - bins[:-1]
    binmids = bins[:-1] + (bindif) / 2
    colors = ["green", "red", "yellow", "lightgrey"]
    fig, ax = plt.subplots(nrows=2, sharex="all", figsize=(width := 6, width / 1.8))
    h, b, p = ax[0].hist(
        [
            fu.dates,
            not_fu.dates[not_fu.maintenance],
            not_fu.dates[not_fu.rejected],
            not_fu_other.dates,
        ],
        histtype="barstacked",
        label=["Followed up", "ZTF down", "Poor localization", "Other reason"],
        bins=bins,
        rwidth=0.9,
        color=colors,
    )

    total = h[-1]
    perc_fu = h[0] / total
    perc_not_fu_main = (h[1] - h[0]) / total
    perc_not_fu = (h[2] - h[1]) / total

    for i in range(len(h)):
        ax[1].bar(
            binmids,
            h[len(h) - i - 1] / h[-1],
            width=bindif * 0.9,
            color=colors[len(h) - i - 1],
        )

    # for i, p in enumerate(perc_not_fu_main):
    for i, p in enumerate(perc_fu):
        if p:
            x = binmids[i]
            y = perc_fu[i] - p / 2
            ax[1].annotate(
                f"{p*100:.0f}%", xy=(x, y), ha="center", va="center", color="white"
            )

    ax[0].set_ylabel("Count")
    ax[1].set_ylabel("Percentage")
    ax[0].legend(
        fontsize=9.5,
        loc="upper center",
        bbox_to_anchor=(0.48, 1.4),
        ncol=4,
        fancybox=True,
    )

    ax[0].set_xticks([])

    fn = os.path.join("follow_up_summary.pdf")
    print(fn)
    fig.tight_layout()
    fig.savefig(fn)
    plt.close()


if __name__ == "__main__":
    make_plot()
