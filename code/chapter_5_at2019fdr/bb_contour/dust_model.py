#!/usr/bin/env python3
# Author: Simeon Reusch (simeon.reusch@desy.de)
# License: BSD-3-Clause

import os, json
from astropy import units as u
from astropy import constants as const
import pandas as pd
import matplotlib as mpl
import numpy as np
from modelSED import utilities
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, convolve
from scipy.interpolate import splev, splrep
from lmfit import Model, Parameters, Minimizer, report_fit, minimize
from astropy.cosmology import FlatLambdaCDM


FIT = True
FITMETHOD = "lm"
PLOT = True


def plot_results_brute(result, best_vals=True, varlabels=None, output=None):
    """Visualize the result of the brute force grid search.

    The output file will display the chi-square value per parameter and contour
    plots for all combination of two parameters.

    Inspired by the `corner` package (https://github.com/dfm/corner.py).

    Parameters
    ----------
    result : :class:`~lmfit.minimizer.MinimizerResult`
        Contains the results from the :meth:`brute` method.

    best_vals : bool, optional
        Whether to show the best values from the grid search (default is True).

    varlabels : list, optional
        If None (default), use `result.var_names` as axis labels, otherwise
        use the names specified in `varlabels`.

    output : str, optional
        Name of the output PDF file (default is 'None')
    """
    npars = len(result.var_names)
    _fig, axes = plt.subplots(npars, npars, dpi=300)

    if not varlabels:
        varlabels = result.var_names
    if best_vals and isinstance(best_vals, bool):
        best_vals = result.params

    for i, par1 in enumerate(result.var_names):
        for j, par2 in enumerate(result.var_names):

            # parameter vs chi2 in case of only one parameter
            if npars == 1:
                axes.plot(result.brute_grid, result.brute_Jout, "o", ms=3)
                axes.set_ylabel(r"$\chi^{2}$")
                axes.set_xlabel(varlabels[i])
                if best_vals:
                    axes.axvline(best_vals[par1].value, ls="dashed", color="r")

            # parameter vs chi2 profile on top
            elif i == j and j < npars - 1:
                if i == 0:
                    axes[0, 0].axis("off")
                ax = axes[i, j + 1]
                red_axis = tuple([a for a in range(npars) if a != i])
                ax.plot(
                    np.unique(result.brute_grid[i]),
                    np.minimum.reduce(result.brute_Jout, axis=red_axis),
                    "o",
                    ms=3,
                )
                ax.set_ylabel(r"$\chi^{2}$")
                ax.yaxis.set_label_position("right")
                ax.yaxis.set_ticks_position("right")
                ax.set_xticks([])
                if best_vals:
                    ax.axvline(best_vals[par1].value, ls="dashed", color="r")

            # parameter vs chi2 profile on the left
            elif j == 0 and i > 0:
                ax = axes[i, j]
                red_axis = tuple([a for a in range(npars) if a != i])
                ax.plot(
                    np.minimum.reduce(result.brute_Jout, axis=red_axis),
                    np.unique(result.brute_grid[i]),
                    "o",
                    ms=3,
                )
                ax.invert_xaxis()
                ax.set_ylabel(varlabels[i])
                if i != npars - 1:
                    ax.set_xticks([])
                else:
                    ax.set_xlabel(r"$\chi^{2}$")
                if best_vals:
                    ax.axhline(best_vals[par1].value, ls="dashed", color="r")

            # contour plots for all combinations of two parameters
            elif j > i:
                ax = axes[j, i + 1]
                red_axis = tuple([a for a in range(npars) if a not in (i, j)])
                X, Y = np.meshgrid(
                    np.unique(result.brute_grid[i]), np.unique(result.brute_grid[j])
                )
                # lvls1 = np.linspace(result.brute_Jout.min(),
                #                     np.median(result.brute_Jout)/2.0, 7, dtype='int')
                # lvls2 = np.linspace(np.median(result.brute_Jout)/2.0,
                #                     np.median(result.brute_Jout), 3, dtype='int')
                # lvls = np.unique(np.concatenate((lvls1, lvls2)))
                ax.contourf(
                    X.T,
                    Y.T,
                    np.minimum.reduce(result.brute_Jout, axis=red_axis),
                    norm=mpl.colors.LogNorm(),
                )  # , lvls, norm=mpl.colors.LogNorm())
                ax.set_yticks([])
                if best_vals:
                    ax.axvline(best_vals[par1].value, ls="dashed", color="r")
                    ax.axhline(best_vals[par2].value, ls="dashed", color="r")
                    ax.plot(best_vals[par1].value, best_vals[par2].value, "rs", ms=3)
                if j != npars - 1:
                    ax.set_xticks([])
                else:
                    ax.set_xlabel(varlabels[i])
                if j - i >= 2:
                    axes[i, j].axis("off")

    if output is not None:
        plt.savefig(output)


if __name__ == "__main__":
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

    nice_fonts = {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": "Times New Roman",
    }
    mpl.rcParams.update(nice_fonts)
    mpl.rcParams["text.usetex"] = True
    mpl.rcParams["text.latex.preamble"] = [r"\usepackage{amsmath}"]  # for \text command

    DPI = 400
    FIG_WIDTH = 6
    GOLDEN_RATIO = 1 / 1.618
    ANNOTATION_FONTSIZE = 12
    AXIS_FONTSIZE = 14
    REDSHIFT = 0.2666

    MJD_OPT_PEAK = 58709.84
    MJD_IR_PEAK = 59074.01

    FITDIR = os.path.join("fit", "dust_model")

    if not os.path.exists(FITDIR):
        os.makedirs(FITDIR)

    infile_fitdf = "fit_lumi_radii.csv"

    PLOT_DIR = "plots"

    df_fit = pd.read_csv(infile_fitdf)

    opt_ir_delay_day = (MJD_IR_PEAK - MJD_OPT_PEAK) * u.day
    opt_ir_delay_s = opt_ir_delay_day.to(u.s)

    light_travel_distance = (opt_ir_delay_s * const.c).to(u.cm)

    fitted_max_optical_radius = np.max(df_fit["optical_radius"].values) * u.cm

    fitted_max_ir_radius = np.max(df_fit["infrared_radius"].values) * u.cm

    def equation_12(T=1850, R=0.1, R_err=None):
        """
        T in Kelvin
        Radius in parsec
        """
        L_abs = 5 * 10 ** 44 * (R / 0.1) ** 2 * (T / 1850) ** 5.8 * u.erg / u.s

        L_abs_dR = 10 * 10 ** 44 * (R) / 0.1 ** 2 * (T / 1850) ** 5.8 * u.erg / u.s

        if R_err is not None:
            L_abs_err = np.sqrt(L_abs_dR ** 2 * R_err ** 2)
        else:
            L_abs_err = None

        return L_abs, L_abs_err

    infile_lc = os.path.join("data", "lightcurves", "full_lightcurve.csv")
    lc = pd.read_csv(infile_lc)
    lc_g = lc.query("telescope == 'P48' and band == 'ZTF_g'")
    lc_g = lc_g.sort_values(by=["obsmjd"])

    lc_w1 = lc.query("telescope == 'WISE' and band == 'W1'")
    lc_w2 = lc.query("telescope == 'WISE' and band == 'W2'")
    wl_w1 = 33156.56
    wl_w2 = 46028.00
    wl_g = 4722.74

    obsmjd_w1 = lc_w1.obsmjd.values
    obsmjd_w1 = np.array([58710, 58911, 59074])
    obsmjd_w2 = obsmjd_w1
    obsmjd_g = lc_g.obsmjd.values

    mag_w1 = lc_w1.mag.values
    mag_w2 = lc_w2.mag.values
    mag_g = lc_g.mag.values

    mag_err_w1 = lc_w1.mag_err.values
    mag_err_w2 = lc_w2.mag_err.values
    mag_err_g = lc_g.mag_err.values

    flux_w1 = utilities.abmag_to_flux(mag_w1)
    flux_w2 = utilities.abmag_to_flux(mag_w2)
    flux_g = utilities.abmag_to_flux(mag_g)

    flux_err_w1 = utilities.abmag_err_to_flux_err(mag_w1, mag_err_w1)
    flux_err_w2 = utilities.abmag_err_to_flux_err(mag_w2, mag_err_w2)
    flux_err_g = utilities.abmag_err_to_flux_err(mag_g, mag_err_g)

    nu_fnu_w1, nu_fnu_err_w1 = utilities.flux_density_to_flux(
        wl_w1, flux_w1, flux_err_w1
    )
    nu_fnu_w2, nu_fnu_err_w2 = utilities.flux_density_to_flux(
        wl_w2, flux_w2, flux_err_w2
    )
    nu_fnu_g, nu_fnu_err_g = utilities.flux_density_to_flux(wl_g, flux_g, flux_err_g)

    # We fit the optical lightcurve with a spline
    spline_g = splrep(obsmjd_g, nu_fnu_g, s=4e-25)

    mjds = np.arange(58000, 59801, 1)
    spline_eval_g = splev(mjds, spline_g)
    spline_g = []

    for i, mjd in enumerate(mjds):
        if mjd < 58600 or mjd > 59450:
            spline_g.append(0)
        else:
            spline_g.append(spline_eval_g[i])

    def minimizer_function(params, x, data=None, data_err=None, **kwargs):

        do_plot = False

        delay = params["delay"]
        ltt = params["ltt"]
        amplitude = params["amplitude"]

        mjds = np.arange(58000, 59801, 1)

        _spline_g = kwargs["spline_g"]

        _boxfunc = []
        for i, mjd in enumerate(mjds):
            if mjd < (MJD_OPT_PEAK) or mjd > (MJD_OPT_PEAK + (2 * ltt)):
                _boxfunc.append(0)
            else:
                _boxfunc.append(1)

        _convolution = (
            convolve(_spline_g, _boxfunc, mode="same") / sum(_boxfunc) * amplitude
        )

        spline_conv = splrep(mjds + delay, _convolution, s=1e-30)

        residuals = []
        fitvals = []

        for i, flux in enumerate(data):

            mjd = x[i]
            j = np.where(mjds == mjd)

            flux_err = data_err[i]

            fitval = splev(mjd, spline_conv)
            fitvals.append(fitval)

            delta = fitval - flux

            res = delta / flux_err

            residuals.append(res)

        vals = []
        for mjd in mjds:
            vals.append(splev(mjd, spline_conv))

        residuals = np.array(residuals)

        chisq = np.sum(residuals ** 2)

        print(f"Chisquare = {chisq:.2f}")

        if do_plot:
            fig = plt.figure(dpi=DPI, figsize=(FIG_WIDTH, FIG_WIDTH * GOLDEN_RATIO))
            ax = fig.add_subplot(1, 1, 1)
            ax2 = ax.twinx()
            ax.set_xlim([58000 + 500, 59800])

            ax.set_yscale("log")
            ax.set_ylim([1e-14, 1e-11])
            ax.set_xlabel("Days since peak")
            ax.set_ylabel(r"$\nu F_{\nu}$ [erg/s/cm$^2$]")
            ax2.set_ylabel("Transfer function")

            ax.scatter(x, data)

            ax.plot(
                mjds,
                _spline_g,
                color="tab:green",
            )

            ax.plot(
                mjds + delay,
                [splev(mjd, spline_conv) for mjd in mjds],
                color="tab:blue",
            )

            ax.plot(mjds, vals, color="tab:red")

            ax.scatter(x, fitvals, color="tab:red")

            outfile = os.path.join(
                "test",
                f"del_{delay.value:.1f}_ltt_{ltt.value}_ampl_{amplitude.value:.2f}.png",
            )
            plt.savefig(outfile)

        return residuals

    params = Parameters()

    if FITMETHOD == "brute":

        params.add("delay", min=165, max=185)
        params.add("ltt", min=190, max=210)
        params.add("amplitude", min=1.2, max=1.5)

    else:

        params.add("delay", min=100, max=250)
        params.add("ltt", min=150, max=250)
        params.add("amplitude", min=1.1, max=1.5)

    x = obsmjd_w1
    data = nu_fnu_w1
    data_err = nu_fnu_err_w1

    x = np.insert(x, 0, 58600)
    data = np.insert(data, 0, 0)
    # looks good
    # data_err = np.insert(data_err, 0, 1e-14)
    # has errors
    # data_err = np.insert(data_err, 0, 1e-13)
    data_err = np.insert(data_err, 0, 1e-16)

    minimizer = Minimizer(
        userfcn=minimizer_function,
        params=params,
        fcn_args=(x, data, data_err),
        fcn_kws={"spline_g": spline_g},
        calc_covar=True,
    )

    if FIT:

        if FITMETHOD == "basinhopping":
            res = minimizer.minimize(method=FITMETHOD, Ns=30)
        else:
            res = minimizer.minimize(method=FITMETHOD)

        print(report_fit(res))

        if FITMETHOD == "brute":
            plot_results_brute(res, best_vals=True, varlabels=None, output="test.png")

        ltt = res.params["ltt"].value
        delay = res.params["delay"].value
        amplitude = res.params["amplitude"].value
        ltt_err = res.params["ltt"].stderr
        delay_err = res.params["delay"].stderr
        ampl_err = res.params["amplitude"].stderr

    else:
        delay = 178.399887
        delay_err = 5
        ltt = 198.228677
        ltt_err = 198 * 0.05
        amplitude = 1.35463143
        ampl_err = 0.1

    dust_distance_model = (ltt * u.day * const.c).to(u.cm)
    dust_distance_model_err = (ltt_err * u.day * const.c).to(u.cm)

    dust_distance_model_pc = dust_distance_model.to(u.pc)
    dust_distance_model_err_pc = dust_distance_model_err.to(u.pc)

    boxfunc = []
    for i, mjd in enumerate(mjds):
        if mjd < (MJD_OPT_PEAK) or mjd > (MJD_OPT_PEAK + (2 * ltt)):
            boxfunc.append(0)
        else:
            boxfunc.append(1)

    # We calculate the convolution
    convolution = convolve(spline_g, boxfunc, mode="same") / sum(boxfunc) * amplitude

    convolution_data = {"mjds": list(mjds + delay), "convolution": list(convolution)}
    convolution_outfile = os.path.join(FITDIR, "dust_model.json")

    with open(convolution_outfile, "w") as f:
        json.dump(convolution_data, f)

    spline_conv = splrep(mjds + delay, convolution, s=1e-30)
    vals = []
    for mjd in mjds:
        vals.append(splev(mjd, spline_conv))

    if PLOT:

        # And now we plot

        fig = plt.figure(dpi=DPI, figsize=(FIG_WIDTH, FIG_WIDTH * GOLDEN_RATIO))
        ax = fig.add_subplot(1, 1, 1)
        ax2 = ax.twinx()
        ax.set_xlim([58000 + 500, 59800])
        ax.set_yscale("log")
        ax.set_ylim([1e-14, 1e-11])
        ax.set_xlabel("Days since peak")
        ax.set_ylabel(r"$\nu F_{\nu}$ [erg/s/cm$^2$]")
        ax2.set_ylabel("Transfer function")
        ax.errorbar(
            obsmjd_g,
            nu_fnu_g,
            nu_fnu_err_g,
            color="tab:green",
            alpha=0.5,
            label="ZTF g-band",
            fmt=".",
        )
        ax.errorbar(
            obsmjd_w1,
            nu_fnu_w1,
            nu_fnu_err_w1,
            color="tab:blue",
            label="WISE W1",
            fmt=".",
            markersize=1,
        )

        ax.plot(mjds, spline_g, c="green")

        ax.plot(mjds, vals, color="tab:red")

        ax2.plot(
            mjds + delay,
            boxfunc,
            ls="dashed",
            c="black",
            alpha=0.3,
            label="transfer function",
        )

        ax.legend(loc=2)
        ax2.legend()
        outpath_pdf = os.path.join(PLOT_DIR, "dust_modeling.pdf")
        fig.savefig(outpath_pdf)

    ltt = ltt * u.d

    print("\n")
    print(f"--- TIME DELAYS ----")
    print(f"time delay between optical and IR peak: {opt_ir_delay_day:.0f}")
    print(f"time delay inferred from Sjoert's model (boxfunc/2): {ltt:.0f}")
    print("\n")
    print("----- DUST DISTANCE -----")
    print(f"inferred from light travel time: {light_travel_distance:.2e}")
    print(f"inferred from BB fit: {fitted_max_ir_radius:.2e}")
    print(
        f"inferred from Sjoert's model: {dust_distance_model:.2e} +/- {dust_distance_model_err:.2e}"
    )
    print(f"[{dust_distance_model_pc:.3f} +/- {dust_distance_model_err_pc:.3f}]")

    T_Tywin = 1850

    L_abs_paper, L_abs_paper_err = equation_12(
        T=T_Tywin,
        R=dust_distance_model_pc.value,
        R_err=dust_distance_model_err_pc.value,
    )

    print(L_abs_paper)
    print(L_abs_paper_err)

    log10L_abs_paper = np.log10(L_abs_paper.value)
    # L_abs_frombb = 1.02e45 * u.erg / u.s
    L_abs_frombb = 1.435e45 * u.erg / u.s
    # L_abs_frombb = 6.67e44 * u.erg / u.s
    log10_L_abs_frombb = np.log10(L_abs_frombb.value)

    L_dust_frombb = 4.247e44 * u.erg / u.s  # from epoch 1!
    # L_dust_frombb = 4.61e44 * u.erg / u.s # from epoch 1!

    print("\n")
    print("----- ENERGETICS (ALL FROM SJOERT'S MODEL) -------")
    print(f"R used for following calculations: {dust_distance_model_pc:.3f}")
    print(f"T used for following calculations: {T_Tywin} K")
    print("\n")
    print(f"L_abs (paper eq. 12) = {L_abs_paper:.2e}")
    print(f"L_abs (peak opt/UV BB luminosity) = {L_abs_frombb:.2e}")
    print(f"L_dust (peak IR BB luminosity) = {L_dust_frombb:.2e}")
    print("--------------------")

    # Now we integrate this over the optical lightcurve

    spline_g_full = splrep(obsmjd_g, nu_fnu_g, s=4e-25)

    time = spline_g_full[0] - min(spline_g_full[0])
    time = [(t * u.day).to(u.s).value for t in time]

    max_of_g = max(spline_g_full[1])

    new_spline_paper = spline_g_full[1] / max_of_g * L_abs_paper.value
    E_abs_paper = np.trapz(y=new_spline_paper, x=time) * u.erg
    log10E_abs_paper = np.log10(E_abs_paper.value)

    new_spline_frombb = spline_g_full[1] / max_of_g * L_abs_frombb.value
    E_abs_frombb = np.trapz(y=new_spline_frombb, x=time) * u.erg
    log10E_abs_frombb = np.log10(E_abs_frombb.value)

    print(f"E_abs (from paper eq. 12) = {E_abs_paper:.2e}")
    print(f"E_abs (from peak opt/UV BB luminosity) = {E_abs_frombb:.2e}")
    print("--------------------")

    time = mjds - min(mjds)
    time = [(t * u.day).to(u.s).value for t in time]

    d = cosmo.luminosity_distance(REDSHIFT)
    d = d.to(u.cm).value
    L_abs_mod = max_of_g * 4 * np.pi * d ** 2

    spline_dust = spline_g / max(spline_g) * L_dust_frombb.value

    E_dust = np.trapz(y=spline_dust, x=time)
    E_dust = E_dust * u.erg
    log10E_dust = np.log10(E_dust.value)

    print(f"E_dust (from peak IR BB lumi) = {E_dust:.2e}")
    print("--------------------")
    print(f"E_total = {E_dust+E_abs_frombb:.2e}")

    f_paper = E_dust / E_abs_paper
    f_frombb = E_dust / E_abs_frombb

    print(f"Covering factor (from paper eq. 12) = {f_paper:.2f}")
    print(f"Covering factor (from BB lumis) = {f_frombb:.2f}")
