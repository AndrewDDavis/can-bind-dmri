#!/usr/bin/env python3
# coding: utf-8
"""Plot outputs of nontensor_model_stats for use in the manuscript; for details, see
DMRI Pre-Processing Journal.md.txt.
"""

import os, sys, pickle
from glob import glob
import numpy as np
import pandas as pd

# import matplotlib
# matplotlib.use('Qt5Agg')       # on recommendation of Ernest; caused crash

import matplotlib.pyplot as plt

# Plot look
def set_plot_params():
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 'medium'
    plt.rcParams['xtick.labelsize'] = 'small'
    plt.rcParams['ytick.labelsize'] = 'small'

    plt.rcParams['lines.markersize'] = 3.25
    plt.rcParams['legend.framealpha'] = 0.9

    plt.rcParams['mathtext.fontset'] = 'custom'
    plt.rcParams['mathtext.it'] = 'STIXGeneral:italic'       # \mathit{} or default italic
    plt.rcParams['mathtext.bf'] = 'STIXGeneral:bold:italic'  # \mathbf{} bold italic
    plt.rcParams['mathtext.rm']	= 'STIXGeneral:regular'      # \mathrm{} Roman (upright)
    # plt.rcParams['mathtext.sf']   # \mathsf{} sans-serif
    # plt.rcParams['mathtext.tt']   # \mathtt{} Typewriter (monospace)
    # plt.rcParams['mathtext.cal']	# \mathcal{} calligraphic

    plt.rcParams['figure.constrained_layout.use'] = True
    plt.rcParams['figure.constrained_layout.w_pad'] = 1./72
    plt.rcParams['figure.constrained_layout.h_pad'] = 1./72
    plt.rcParams['figure.constrained_layout.wspace'] = 4./72
    plt.rcParams['figure.constrained_layout.hspace'] = 4./72

set_plot_params()

def define_colours():
    # B, R, G colour scheme
    colors = ['#336699', '#990033', '#669966']
    # Alternate 6-colour scheme
    # 21B6A8    177F75  B6212D  7F171F  B67721  7F5417

    # Blue & tan for the array
    blu = colors[0]
    tan = '#C2883A'
    mrn = '#7F171F'

    blu_dark = (28/255, 48/255, 94/255)      # dark blue
    blu_desat = (59/255, 72/255, 105/255)    # desaturated

    return colors, (blu, tan, mrn), (blu_dark, blu_desat)

colors, (blu, tan, mrn), (blu_dark, blu_desat) = define_colours()


# Read data
moddfs = {}
for f in glob('../dmri_model_stats-*.csv'):
    sid=f[20:31]

    # if sid != 'UCA_0074_01':
    #     continue

    moddf=pd.read_csv(f, header=0, index_col=0)
    moddfs[sid] = moddf


# Model plots array; one for each subject
for sid in moddfs:

    print(f'Array plots for {sid}...')

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2,
                                                 # sharex=True,
                                                 figsize=(7.0, 5.5),
                                                 dpi=150)
                                                 # constrained_layout=True)
    # fig.set_constrained_layout_pads(w_pad=2./72, h_pad=2./72,
                                    # hspace=6./72, wspace=6./72)

    # Tissue Fraction Plot
    x = np.arange(1,8)  # bar locations

    fa = moddfs[sid].loc['tensor', 'fsum_mean_wm']

    fsumwm_vals = [fa,
                   moddfs[sid].loc['ntmod_m1_n1_b1000', 'fsum_mean_wm'],
                   moddfs[sid].loc['ntmod_m2_n1_b1000', 'fsum_mean_wm'],
                   moddfs[sid].loc['ntmod_m3_n1_b1000', 'fsum_mean_wm'],
                   moddfs[sid].loc['ntmod_m1_n2_b1000', 'fsum_mean_wm'],
                   moddfs[sid].loc['ntmod_m2_n2_b1000', 'fsum_mean_wm'],
                   moddfs[sid].loc['ntmod_m3_n2_b1000', 'fsum_mean_wm']]

    fsumwm_std_vals = [0,
                       moddfs[sid].loc['ntmod_m1_n1_b1000', 'fsum_std_wm'],
                       moddfs[sid].loc['ntmod_m2_n1_b1000', 'fsum_std_wm'],
                       moddfs[sid].loc['ntmod_m3_n1_b1000', 'fsum_std_wm'],
                       moddfs[sid].loc['ntmod_m1_n2_b1000', 'fsum_std_wm'],
                       moddfs[sid].loc['ntmod_m2_n2_b1000', 'fsum_std_wm'],
                       moddfs[sid].loc['ntmod_m3_n2_b1000', 'fsum_std_wm']]

    # acr_fn = f'../dmri_model_stats_acr-{sid}.csv'
    # if os.path.isfile(acr_fn):
    #     acrdf=pd.read_csv(acr_fn, header=0, index_col=0)

    #     fa = acrdf.loc['tensor', 'fsum_mean_acr']

    #     fsumacr_vals = [fa,
    #                     acrdf.loc['ntmod_m1_n1_b1000', 'fsum_mean_acr'],
    #                     acrdf.loc['ntmod_m2_n1_b1000', 'fsum_mean_acr'],
    #                     acrdf.loc['ntmod_m3_n1_b1000', 'fsum_mean_acr'],
    #                     acrdf.loc['ntmod_m1_n2_b1000', 'fsum_mean_acr'],
    #                     acrdf.loc['ntmod_m2_n2_b1000', 'fsum_mean_acr'],
    #                     acrdf.loc['ntmod_m3_n2_b1000', 'fsum_mean_acr']]

    #     fsumacr_std_vals = [0,
    #                         acrdf.loc['ntmod_m1_n1_b1000', 'fsum_std_acr'],
    #                         acrdf.loc['ntmod_m2_n1_b1000', 'fsum_std_acr'],
    #                         acrdf.loc['ntmod_m3_n1_b1000', 'fsum_std_acr'],
    #                         acrdf.loc['ntmod_m1_n2_b1000', 'fsum_std_acr'],
    #                         acrdf.loc['ntmod_m2_n2_b1000', 'fsum_std_acr'],
    #                         acrdf.loc['ntmod_m3_n2_b1000', 'fsum_std_acr']]

    #     ax1.bar(x+0.25, fsumacr_vals, width=0.5, yerr=fsumacr_std_vals, color=tan, capsize=2.5,
    #             zorder=2)

    col_fn = f'../dmri_model_stats_linpla-{sid}.csv'
    if os.path.isfile(col_fn):
        coldf=pd.read_csv(col_fn, header=0, index_col=0)

        fa_col = coldf.loc['tensor', 'fsum_mean_col']
        fa_pla = coldf.loc['tensor', 'fsum_mean_pla']

        fsumcol_vals = [fa_col,
                        coldf.loc['ntmod_m1_n1_b1000', 'fsum_mean_col'],
                        coldf.loc['ntmod_m2_n1_b1000', 'fsum_mean_col'],
                        coldf.loc['ntmod_m3_n1_b1000', 'fsum_mean_col'],
                        coldf.loc['ntmod_m1_n2_b1000', 'fsum_mean_col'],
                        coldf.loc['ntmod_m2_n2_b1000', 'fsum_mean_col'],
                        coldf.loc['ntmod_m3_n2_b1000', 'fsum_mean_col']]

        # fsumcol_std_vals = [0,
        #                     coldf.loc['ntmod_m1_n1_b1000', 'fsum_std_col'],
        #                     coldf.loc['ntmod_m2_n1_b1000', 'fsum_std_col'],
        #                     coldf.loc['ntmod_m3_n1_b1000', 'fsum_std_col'],
        #                     coldf.loc['ntmod_m1_n2_b1000', 'fsum_std_col'],
        #                     coldf.loc['ntmod_m2_n2_b1000', 'fsum_std_col'],
        #                     coldf.loc['ntmod_m3_n2_b1000', 'fsum_std_col']]

        fsumpla_vals = [fa_pla,
                        coldf.loc['ntmod_m1_n1_b1000', 'fsum_mean_pla'],
                        coldf.loc['ntmod_m2_n1_b1000', 'fsum_mean_pla'],
                        coldf.loc['ntmod_m3_n1_b1000', 'fsum_mean_pla'],
                        coldf.loc['ntmod_m1_n2_b1000', 'fsum_mean_pla'],
                        coldf.loc['ntmod_m2_n2_b1000', 'fsum_mean_pla'],
                        coldf.loc['ntmod_m3_n2_b1000', 'fsum_mean_pla']]

        # fsumpla_std_vals = [0,
        #                     coldf.loc['ntmod_m1_n1_b1000', 'fsum_std_pla'],
        #                     coldf.loc['ntmod_m2_n1_b1000', 'fsum_std_pla'],
        #                     coldf.loc['ntmod_m3_n1_b1000', 'fsum_std_pla'],
        #                     coldf.loc['ntmod_m1_n2_b1000', 'fsum_std_pla'],
        #                     coldf.loc['ntmod_m2_n2_b1000', 'fsum_std_pla'],
        #                     coldf.loc['ntmod_m3_n2_b1000', 'fsum_std_pla']]

        # ax1.bar(x-0.20, fsumwm_vals, width=0.4, yerr=fsumwm_std_vals, color=blu, capsize=2.5,
        #         tick_label=['WDT', 'BS$_{\mathrm{ME}}$$^1$', 'BS$_{\mathrm{GD}}$$^1$', 'BZ$^1$', 'BS$_{\mathrm{ME}}$$^2$', 'BS$_{\mathrm{GD}}$$^2$', 'BZ$^2$'],
        #         zorder=1, label='All WM')

        # ax1.bar(x+0.1, fsumcol_vals, width=0.2, yerr=fsumcol_std_vals, color=tan, capsize=2.5,
        #         zorder=2, label='Collinear')

        # ax1.bar(x+0.3, fsumpla_vals, width=0.2, yerr=fsumpla_std_vals, color=mrn, capsize=2.5,
        #         zorder=3, label='Planar')

        # ^^ leave out SD values -- they may sow confusion because its spatial mean of values, not SD of mean

        ax1.bar(x-0.20, fsumwm_vals, width=0.4, color=blu,
                zorder=1, label='All WM')

        ax1.bar(x+0.1, fsumcol_vals, width=0.2, color=tan,
                zorder=2, label='Linear')

        ax1.bar(x+0.3, fsumpla_vals, width=0.2, color=mrn,
                zorder=3, label='Planar')

        ax1.set_xticks(x)
        ax1.set_xticklabels(['WDT', 'BS$_{\mathrm{ME}}$$^1$', 'BS$_{\mathrm{GD}}$$^1$', 'BZ$^1$', 'BS$_{\mathrm{ME}}$$^2$', 'BS$_{\mathrm{GD}}$$^2$', 'BZ$^2$'])
        ax1.legend(ncol=3, loc='upper left', fontsize='x-small', handletextpad=0.2, borderaxespad=0.2)

        ax1.set_ylabel(r'Restricted fraction ($f_{\mathrm{tot}}$)' '\n' 'or FA')
        # -> anisotropic fraction?

    else:
        # ax1.bar(x, fsumwm_vals, width=0.8, yerr=fsumwm_std_vals, color=blu, capsize=2.5,
        #         tick_label=['WDT', 'BS$_{\mathrm{ME}}$$^1$', 'BS$_{\mathrm{GD}}$$^1$', 'BZ$^1$', 'BS$_{\mathrm{ME}}$$^2$', 'BS$_{\mathrm{GD}}$$^2$', 'BZ$^2$'],
        #         zorder=1, label='All WM')

        ax1.bar(x, fsumwm_vals, width=0.8, color=blu,
                tick_label=['WDT', 'BS$_{\mathrm{ME}}$$^1$', 'BS$_{\mathrm{GD}}$$^1$', 'BZ$^1$', 'BS$_{\mathrm{ME}}$$^2$', 'BS$_{\mathrm{GD}}$$^2$', 'BZ$^2$'],
                zorder=1, label='All WM')

        ax1.set_ylabel(r'Restricted fraction ($f_{\mathrm{tot}}$)' '\n' 'or FA, in WM')


    ax1.grid(alpha=0.25, axis='y')
    ax1.set_axisbelow(True)


    # Time Plot
    x = np.arange(1,9)  # bar locations

    # Note ~ 4.5 s for the WLS tensor (whole brain=95813 voxels!)
    tt = 4.5/95813

    # For camino: 9 min 4 s for 178114 voxels
    tc = 544./178114

    heights = [tt,
               tc,
               moddfs[sid].loc['ntmod_m1_n1_b1000', 'time_spv'],
               moddfs[sid].loc['ntmod_m2_n1_b1000', 'time_spv'],
               moddfs[sid].loc['ntmod_m3_n1_b1000', 'time_spv'],
               moddfs[sid].loc['ntmod_m1_n2_b1000', 'time_spv'],
               moddfs[sid].loc['ntmod_m2_n2_b1000', 'time_spv'],
               moddfs[sid].loc['ntmod_m3_n2_b1000', 'time_spv']]

    errs = [0,
            0,
            moddfs[sid].loc['ntmod_m1_n1_b1000', 'time_spv_std'],
            moddfs[sid].loc['ntmod_m2_n1_b1000', 'time_spv_std'],
            moddfs[sid].loc['ntmod_m3_n1_b1000', 'time_spv_std'],
            moddfs[sid].loc['ntmod_m1_n2_b1000', 'time_spv_std'],
            moddfs[sid].loc['ntmod_m2_n2_b1000', 'time_spv_std'],
            moddfs[sid].loc['ntmod_m3_n2_b1000', 'time_spv_std']]

    # convert values from s/voxel to hrs/10^5 voxels
    heights = [x*10**5/60./60. for x in heights]
    errs = [x*10**5/60./60. for x in errs]

    # ax2.bar(x, heights, yerr=errs, color=blu, capsize=2.5,
    #         tick_label=['WDT', 'NLLS', 'BS$_{\mathrm{ME}}$$^1$', 'BS$_{\mathrm{GD}}$$^1$', 'BZ$^1$', 'BS$_{\mathrm{ME}}$$^2$', 'BS$_{\mathrm{GD}}$$^2$', 'BZ$^2$'])

    ax2.bar(x, heights, color=blu,
            tick_label=['WDT', 'NLLS', 'BS$_{\mathrm{ME}}$$^1$', 'BS$_{\mathrm{GD}}$$^1$', 'BZ$^1$', 'BS$_{\mathrm{ME}}$$^2$', 'BS$_{\mathrm{GD}}$$^2$', 'BZ$^2$'])

    ax2.grid(alpha=0.25, axis='y')
    ax2.set_axisbelow(True)
    ax2.set_ylabel("Processing time\n" r"(hours/(10$^5$ voxels))")


    # # Dispersion Plot
    # x = np.arange(1,7)  # bar locations

    # disp_vals = [moddfs[sid].loc['ntmod_m1_n1_b1000', 'd1disp_mean_wm'],
    #              moddfs[sid].loc['ntmod_m2_n1_b1000', 'd1disp_mean_wm'],
    #              moddfs[sid].loc['ntmod_m3_n1_b1000', 'd1disp_mean_wm'],
    #              moddfs[sid].loc['ntmod_m1_n2_b1000', 'd1disp_mean_wm'],
    #              moddfs[sid].loc['ntmod_m2_n2_b1000', 'd1disp_mean_wm'],
    #              moddfs[sid].loc['ntmod_m3_n2_b1000', 'd1disp_mean_wm']]

    # # disp_std_vals = [moddfs[sid].loc['ntmod_m1_n1_b1000', 'd1disp_std_wm'],
    # #                  moddfs[sid].loc['ntmod_m2_n1_b1000', 'd1disp_std_wm'],
    # #                  moddfs[sid].loc['ntmod_m3_n1_b1000', 'd1disp_std_wm'],
    # #                  moddfs[sid].loc['ntmod_m1_n2_b1000', 'd1disp_std_wm'],
    # #                  moddfs[sid].loc['ntmod_m2_n2_b1000', 'd1disp_std_wm'],
    # #                  moddfs[sid].loc['ntmod_m3_n2_b1000', 'd1disp_std_wm']]

    # ax2.bar(x, disp_vals, color=blu,
    #         tick_label=['BS$_{\mathrm{ME}}$$^1$', 'BS$_{\mathrm{GD}}$$^1$', 'BZ$^1$', 'BS$_{\mathrm{ME}}$$^2$', 'BS$_{\mathrm{GD}}$$^2$', 'BZ$^2$'])

    # ax2.grid(alpha=0.25, axis='y')
    # ax2.set_axisbelow(True)
    # ax2.set_ylabel('Dispersion for\nFO 1, in WM')

    # # remove tick for tensor here; or maybe use RD?
    # # xticks = ax2.xaxis.get_major_ticks()
    # # xticks[0].set_visible(False)


    # Diffusivity Plot
    x = np.arange(1,8)  # bar locations

    md = moddfs[sid].loc['tensor', 'd_mean_br']

    dbr_vals = [md,
                moddfs[sid].loc['ntmod_m1_n1_b1000', 'd_mean_br'],
                moddfs[sid].loc['ntmod_m2_n1_b1000', 'd_mean_br'],
                moddfs[sid].loc['ntmod_m3_n1_b1000', 'd_mean_br'],
                moddfs[sid].loc['ntmod_m1_n2_b1000', 'd_mean_br'],
                moddfs[sid].loc['ntmod_m2_n2_b1000', 'd_mean_br'],
                moddfs[sid].loc['ntmod_m3_n2_b1000', 'd_mean_br']]

    # ax3.bar(x, dbr_vals, yerr=dbr_std_vals, color=blu, capsize=2.5,
    #         tick_label=['WDT', 'BS$_{\mathrm{ME}}$$^1$', 'BS$_{\mathrm{GD}}$$^1$', 'BZ$^1$', 'BS$_{\mathrm{ME}}$$^2$', 'BS$_{\mathrm{GD}}$$^2$', 'BZ$^2$'])

    ax3.bar(x, dbr_vals, color=blu, capsize=2.5,
            tick_label=['WDT', 'BS$_{\mathrm{ME}}$$^1$', 'BS$_{\mathrm{GD}}$$^1$', 'BZ$^1$', 'BS$_{\mathrm{ME}}$$^2$', 'BS$_{\mathrm{GD}}$$^2$', 'BZ$^2$'])

    ax3.grid(alpha=0.25, axis='y')
    ax3.set_axisbelow(True)
    ax3.set_ylabel(r'Mean diffusivity ($d$),' '\n' + r'whole brain (μm²/ms)')


    # Add a SD(diff) plot instead of dispersion
    x = np.arange(1,7)  # bar locations

    dbr_std_vals = [moddfs[sid].loc['ntmod_m1_n1_b1000', 'd_std_br'],
                    moddfs[sid].loc['ntmod_m2_n1_b1000', 'd_std_br'],
                    moddfs[sid].loc['ntmod_m3_n1_b1000', 'd_std_br'],
                    moddfs[sid].loc['ntmod_m1_n2_b1000', 'd_std_br'],
                    moddfs[sid].loc['ntmod_m2_n2_b1000', 'd_std_br'],
                    moddfs[sid].loc['ntmod_m3_n2_b1000', 'd_std_br']]

    ax4.bar(x, dbr_std_vals, color=blu,
            tick_label=['BS$_{\mathrm{ME}}$$^1$', 'BS$_{\mathrm{GD}}$$^1$', 'BZ$^1$', 'BS$_{\mathrm{ME}}$$^2$', 'BS$_{\mathrm{GD}}$$^2$', 'BZ$^2$'])

    ax4.grid(alpha=0.25, axis='y')
    ax4.set_axisbelow(True)
    ax4.set_ylabel(r'Run-to-run SD($d$),' '\n' + r'whole brain (μm²/ms)')


    for ax in ax1, ax2, ax3, ax4:
        for tick in ax.get_xticklabels():
            tick.set_rotation(30)

    fig.savefig(f"mod-plots-array-{sid}.png", dpi=600)
    # fig.show()
    plt.close(fig)

# Nomenclature; use similar language to Permeability diffusivity imaging (PDI)
# Restricted compartment (R) and unrestricted compartment (U); so we have diffusivities
# D_r and D_u

# plotting variable of burn-in values
x = np.array([0, 1000, 2000, 5000, 10000, 15000, 20000, 25000, 50000])
x = x/1000.

# standardized plots
def plot_wfill(ax, x, y, yerr, lab, clr, ls='-'):
    ax.plot(x, y,
            ls=ls, marker='o', mec='k', mfc=clr, label=lab, color=clr)
    ax.fill_between(x, y-yerr, y+yerr,
                    alpha=0.4, edgecolor='None', facecolor=clr)

# def plot_2ax(ax, axR, x, y, yerr, lab, clr, ls='-'):
#     ax.plot(x, y, ls=ls, marker='o', mec='k', mfc='None', label=lab, color=clr)

#     axR.plot(x, yerr, ls='--', marker='None', label=lab+' SD', color=clr)


# Burn-in comparison plots; one for each subject
for sid in moddfs:

    print(f'BI plots for {sid}...')

    # Restricted Fraction Plots -- put them all on one axis; for supplemental
    fig1, ax1a = plt.subplots(nrows=1, ncols=1,
                              figsize=(7.0, 7.0),
                              dpi=150)


    fa = moddfs[sid].loc['tensor', 'fsum_mean_wm']
    ax1a.plot(0, fa, ls='None', marker='s', mec='k', mfc='None', label='WDT FA')


    fsumwm1 = moddfs[sid].loc[moddfs[sid].index.str.startswith('ntmod_m1_n2_b'), 'fsum_mean_wm']
    fsumwm2 = moddfs[sid].loc[moddfs[sid].index.str.startswith('ntmod_m2_n2_b'), 'fsum_mean_wm']
    fsumwm3 = moddfs[sid].loc[moddfs[sid].index.str.startswith('ntmod_m3_n2_b'), 'fsum_mean_wm']

    fsumwm1_std = moddfs[sid].loc[moddfs[sid].index.str.startswith('ntmod_m1_n2_b'), 'fsum_std_wm']
    fsumwm2_std = moddfs[sid].loc[moddfs[sid].index.str.startswith('ntmod_m2_n2_b'), 'fsum_std_wm']
    fsumwm3_std = moddfs[sid].loc[moddfs[sid].index.str.startswith('ntmod_m3_n2_b'), 'fsum_std_wm']

    plot_wfill(ax1a, x, fsumwm1, fsumwm1_std, 'BS$_{\mathrm{ME}}$$^2$ Tot', colors[0])
    plot_wfill(ax1a, x, fsumwm2, fsumwm2_std, 'BS$_{\mathrm{GD}}$$^2$ Tot', colors[1])
    plot_wfill(ax1a, x, fsumwm3, fsumwm3_std, 'BZ$^2$ Tot', colors[2])

    # fiber population 1
    f1wm1 = moddfs[sid].loc[moddfs[sid].index.str.startswith('ntmod_m1_n2_b'), 'f1_mean_wm']
    f1wm2 = moddfs[sid].loc[moddfs[sid].index.str.startswith('ntmod_m2_n2_b'), 'f1_mean_wm']
    f1wm3 = moddfs[sid].loc[moddfs[sid].index.str.startswith('ntmod_m3_n2_b'), 'f1_mean_wm']

    f1wm1_std = moddfs[sid].loc[moddfs[sid].index.str.startswith('ntmod_m1_n2_b'), 'f1_std_wm']
    f1wm2_std = moddfs[sid].loc[moddfs[sid].index.str.startswith('ntmod_m2_n2_b'), 'f1_std_wm']
    f1wm3_std = moddfs[sid].loc[moddfs[sid].index.str.startswith('ntmod_m3_n2_b'), 'f1_std_wm']

    plot_wfill(ax1a, x, f1wm1, f1wm1_std, 'BS$_{\mathrm{ME}}$$^2$ FO1', colors[0], '--')
    plot_wfill(ax1a, x, f1wm2, f1wm2_std, 'BS$_{\mathrm{GD}}$$^2$ FO1', colors[1], '--')
    plot_wfill(ax1a, x, f1wm3, f1wm3_std, 'BZ$^2$ FO1', colors[2], '--')

    # fiber population 2
    f2wm1 = moddfs[sid].loc[moddfs[sid].index.str.startswith('ntmod_m1_n2_b'), 'f2_mean_wm']
    f2wm2 = moddfs[sid].loc[moddfs[sid].index.str.startswith('ntmod_m2_n2_b'), 'f2_mean_wm']
    f2wm3 = moddfs[sid].loc[moddfs[sid].index.str.startswith('ntmod_m3_n2_b'), 'f2_mean_wm']

    f2wm1_std = moddfs[sid].loc[moddfs[sid].index.str.startswith('ntmod_m1_n2_b'), 'f2_std_wm']
    f2wm2_std = moddfs[sid].loc[moddfs[sid].index.str.startswith('ntmod_m2_n2_b'), 'f2_std_wm']
    f2wm3_std = moddfs[sid].loc[moddfs[sid].index.str.startswith('ntmod_m3_n2_b'), 'f2_std_wm']

    plot_wfill(ax1a, x, f2wm1, f2wm1_std, 'BS$_{\mathrm{ME}}$$^2$ FO2', colors[0], '-.')
    plot_wfill(ax1a, x, f2wm2, f2wm2_std, 'BS$_{\mathrm{GD}}$$^2$ FO2', colors[1], '-.')
    plot_wfill(ax1a, x, f2wm3, f2wm3_std, 'BZ$^2$ FO2', colors[2], '-.')

    # # brain -- removed for space and clarity
    # fa = moddfs[sid].loc['tensor', 'fsum_mean_br']

    # fsumbr1 = moddfs[sid].loc[moddfs[sid].index.str.startswith('ntmod_m1_n2_b'), 'fsum_mean_br']
    # fsumbr2 = moddfs[sid].loc[moddfs[sid].index.str.startswith('ntmod_m2_n2_b'), 'fsum_mean_br']
    # fsumbr3 = moddfs[sid].loc[moddfs[sid].index.str.startswith('ntmod_m3_n2_b'), 'fsum_mean_br']

    # fsumbr1_std = moddfs[sid].loc[moddfs[sid].index.str.startswith('ntmod_m1_n2_b'), 'fsum_std_br']
    # fsumbr2_std = moddfs[sid].loc[moddfs[sid].index.str.startswith('ntmod_m2_n2_b'), 'fsum_std_br']
    # fsumbr3_std = moddfs[sid].loc[moddfs[sid].index.str.startswith('ntmod_m3_n2_b'), 'fsum_std_br']

    # plot_wfill(ax1a, x, fsumbr1, fsumbr1_std, 'BS$_{\mathrm{ME}}$$^2$ Tot Br', colors[0], ':')
    # plot_wfill(ax1a, x, fsumbr2, fsumbr2_std, 'BS$_{\mathrm{GD}}$$^2$ Tot Br', colors[1], ':')
    # plot_wfill(ax1a, x, fsumbr3, fsumbr3_std, 'BZ$^2$ Tot Br', colors[2], ':')

    # ax1a.plot(0, fa, ls='None', marker='x', mec='k', mfc='None', label='FA Br')

    ax1a.set_xlabel(r'Burn-in ($\times 10^3$)')
    ax1a.set_ylabel(r'Restricted fraction ($f$) in WM')

    ax1a.grid(alpha=0.33, linewidth=0.5)
    for spine in ('right', 'top'):
        ax1a.spines[spine].set_visible(False)

    ax1a.legend(loc='best', fontsize='x-small')

    fig1.savefig(f"TF_plots-{sid}.png", dpi=600)
    # fig1.show()


    # Diffusivity + fsum plots for main manuscript...
    fig2, (ax2a, ax2b) = plt.subplots(nrows=2, ncols=1,
                                      sharex=True,
                                      figsize=(7.0, 5.5),
                                      dpi=150)

    # fig2, (ax2a, ax2aR, ax2b, ax2bR) = plt.subplots(nrows=4, ncols=1,
    #                                   gridspec_kw={'height_ratios':[2,1,2,1]},
    #                                   sharex=True,
    #                                   figsize=(7.0, 5.5),
    #                                   dpi=150)


    fig2.set_constrained_layout_pads(hspace=0., wspace=0.) # legend provides enough gap

    # Fsum
    plot_wfill(ax2a, x, fsumwm1, fsumwm1_std, 'BS$_{\mathrm{ME}}$$^2$', colors[0])
    plot_wfill(ax2a, x, fsumwm2, fsumwm2_std, 'BS$_{\mathrm{GD}}$$^2$', colors[1])
    plot_wfill(ax2a, x, fsumwm3, fsumwm3_std, 'BZ$^2$', colors[2])

    # plot with 2 y-axes to include run-to-run SD
    # this did not work well -- not a great use of space
    # ax2aR = ax2a.twinx()
    # # ax2aR.yaxis.tick_right()
    # # ax2aR.yaxis.set_label_position("right")
    # plot_2ax(ax2a, ax2aR, x, fsumwm1, fsumwm1_std, 'BS$_{\mathrm{ME}}$$^2$', colors[0])
    # plot_2ax(ax2a, ax2aR, x, fsumwm2, fsumwm2_std, 'BS$_{\mathrm{GD}}$$^2$', colors[1])
    # plot_2ax(ax2a, ax2aR, x, fsumwm3, fsumwm3_std, 'BZ$^2$', colors[2])


    # d WM
    mdwm = moddfs[sid].loc['tensor', 'd_mean_wm']
    # fn = moddfs[sid].loc['tensor', 'd1disp_wm']     # Frob-norm in d1disp column
    # ax2b.plot(0, fn, 'kx', label='MSN')

    dwm1 = moddfs[sid].loc[moddfs[sid].index.str.startswith('ntmod_m1_n2_b'), 'd_mean_wm']
    dwm2 = moddfs[sid].loc[moddfs[sid].index.str.startswith('ntmod_m2_n2_b'), 'd_mean_wm']
    dwm3 = moddfs[sid].loc[moddfs[sid].index.str.startswith('ntmod_m3_n2_b'), 'd_mean_wm']

    dwm1_std = moddfs[sid].loc[moddfs[sid].index.str.startswith('ntmod_m1_n2_b'), 'd_std_wm']
    dwm2_std = moddfs[sid].loc[moddfs[sid].index.str.startswith('ntmod_m2_n2_b'), 'd_std_wm']
    dwm3_std = moddfs[sid].loc[moddfs[sid].index.str.startswith('ntmod_m3_n2_b'), 'd_std_wm']

    # plot_wfill(ax2b, x, dwm1, dwm1_std, 'BS$_{\mathrm{ME}}$$^2$ WM', colors[0])
    # plot_wfill(ax2b, x, dwm2, dwm2_std, 'BS$_{\mathrm{GD}}$$^2$ WM', colors[1])
    # plot_wfill(ax2b, x, dwm3, dwm3_std, 'BZ$^2$ WM', colors[2])

    # d Brain
    mdbr = moddfs[sid].loc['tensor', 'd_mean_br']
    # fn = moddfs[sid].loc['tensor', 'd1disp_br']

    dbr1 = moddfs[sid].loc[moddfs[sid].index.str.startswith('ntmod_m1_n2_b'), 'd_mean_br']
    dbr2 = moddfs[sid].loc[moddfs[sid].index.str.startswith('ntmod_m2_n2_b'), 'd_mean_br']
    dbr3 = moddfs[sid].loc[moddfs[sid].index.str.startswith('ntmod_m3_n2_b'), 'd_mean_br']

    dbr1_std = moddfs[sid].loc[moddfs[sid].index.str.startswith('ntmod_m1_n2_b'), 'd_std_br']
    dbr2_std = moddfs[sid].loc[moddfs[sid].index.str.startswith('ntmod_m2_n2_b'), 'd_std_br']
    dbr3_std = moddfs[sid].loc[moddfs[sid].index.str.startswith('ntmod_m3_n2_b'), 'd_std_br']

    plot_wfill(ax2b, x, dbr1, dbr1_std, 'BS$_{\mathrm{ME}}$$^2$', colors[0])
    plot_wfill(ax2b, x, dbr2, dbr2_std, 'BS$_{\mathrm{GD}}$$^2$', colors[1])
    plot_wfill(ax2b, x, dbr3, dbr3_std, 'BZ$^2$', colors[2])

    # ax2bR = ax2b.twinx()
    # # ax2bR.yaxis.tick_right()
    # # ax2bR.yaxis.set_label_position("right")
    # plot_2ax(ax2b, ax2bR, x, dbr1, dbr1_std, 'BS$_{\mathrm{ME}}$$^2$', colors[0])
    # plot_2ax(ax2b, ax2bR, x, dbr2, dbr2_std, 'BS$_{\mathrm{GD}}$$^2$', colors[1])
    # plot_2ax(ax2b, ax2bR, x, dbr3, dbr3_std, 'BZ$^2$', colors[2])

    # ax2b.plot(0, mdwm, ls='None', marker='s', mec='k', mfc='None', label='WDT MD WM')
    ax2b.plot(0, mdbr, ls='None', marker='x', mec='k', mfc='None', label='WDT MD')

    # Labels
    ax2a.set_ylabel(r'Restricted fraction ($f_{\mathrm{tot}}$),' '\n' 'in WM')
    # ax2aR.set_ylabel(r'Run-to-run SD($\mathcal{\,f}_{tot}$),' '\n' 'in WM')

    ax2a.grid(alpha=0.33, linewidth=0.5)
    for spine in ('top', 'bottom', 'right'):
        ax2a.spines[spine].set_visible(False)
        # ax2aR.spines[spine].set_visible(False)
    ax2a.tick_params(bottom=False)

    ax2b.set_xlabel(r'Burn-in ($\times 10^3$)')
    ax2b.set_ylabel(r'Mean diffusivity ($d$),' '\n' r'whole brain (μm²/ms)')
    # ax2bR.set_ylabel(r'Run-to-run SD($\mathcal{d}$),' '\n' r'whole brain (μm$^2$/ms)')

    ax2b.grid(alpha=0.33, linewidth=0.5)
    for spine in ('top', 'right'):
        ax2b.spines[spine].set_visible(False)
        # ax2bR.spines[spine].set_visible(False)

    # put a legend between the subplots
    ax2b.legend(loc='lower center', bbox_to_anchor=(0.15, 1.03, .7, .10),
                ncol=4, borderaxespad=0., fontsize='x-small')
    # ax2b.legend(loc='upper right', ncol=4, borderaxespad=0., fontsize='x-small')

    # fig2.tight_layout(pad=0.5, w_pad=0.25, h_pad=0.25)#, rect=[0, 0, 0.90, 1.])
    fig2.savefig(f"Diff+fsum_plots-{sid}.png", dpi=600)
    # fig2.show()


    # Full diffusivity plots for supplemental
    figD, axD = plt.subplots(figsize=(7.0, 3.5), dpi=150)

    plot_wfill(axD, x, dwm1, dwm1_std, 'BS$_{\mathrm{ME}}$$^2$ WM', colors[0])
    plot_wfill(axD, x, dwm2, dwm2_std, 'BS$_{\mathrm{GD}}$$^2$ WM', colors[1])
    plot_wfill(axD, x, dwm3, dwm3_std, 'BZ$^2$ WM', colors[2])

    plot_wfill(axD, x, dbr1, dbr1_std, 'BS$_{\mathrm{ME}}$$^2$ BR', colors[0], ':')
    plot_wfill(axD, x, dbr2, dbr2_std, 'BS$_{\mathrm{GD}}$$^2$ BR', colors[1], ':')
    plot_wfill(axD, x, dbr3, dbr3_std, 'BZ$^2$ BR', colors[2], ':')

    axD.plot(0, mdwm, ls='None', marker='s', mec='k', mfc='None', label='WDT MD WM')
    axD.plot(0, mdbr, ls='None', marker='x', mec='k', mfc='None', label='WDT MD BR')

    axD.set_xlabel(r'Burn-in ($\times 10^3$)')
    axD.set_ylabel(r'Mean diffusivity ($d$, μm²/ms)')

    axD.grid(alpha=0.33, linewidth=0.5)
    for spine in ('right', 'top'):
        axD.spines[spine].set_visible(False)

    axD.legend(loc='lower center', bbox_to_anchor=(0.15, 1.03, .7, .10),
                ncol=3, borderaxespad=0., fontsize='x-small')

    figD.savefig(f"Diff_plots-{sid}.png", dpi=600)


    # Dispersion FO1 and STD(d) plots for main text
    fig3, (ax3a, ax3c) = plt.subplots(nrows=2, ncols=1,
                                      sharex=True,
                                      figsize=(7.0, 5.5),
                                      dpi=150)
    fig3.set_constrained_layout_pads(hspace=0., wspace=0.) # legend provides enough gap

    # Dispersion Plots (dyads<i>_dispersion) -- note this is dimensionless, comes from dot-product
    d1disp1 = moddfs[sid].loc[moddfs[sid].index.str.startswith('ntmod_m1_n2_b'), 'd1disp_mean_wm']
    d1disp2 = moddfs[sid].loc[moddfs[sid].index.str.startswith('ntmod_m2_n2_b'), 'd1disp_mean_wm']
    d1disp3 = moddfs[sid].loc[moddfs[sid].index.str.startswith('ntmod_m3_n2_b'), 'd1disp_mean_wm']

    # d1disp1_std = moddfs[sid].loc[moddfs[sid].index.str.startswith('ntmod_m1_n2_b'), 'd1disp_std_wm']
    # d1disp2_std = moddfs[sid].loc[moddfs[sid].index.str.startswith('ntmod_m2_n2_b'), 'd1disp_std_wm']
    # d1disp3_std = moddfs[sid].loc[moddfs[sid].index.str.startswith('ntmod_m3_n2_b'), 'd1disp_std_wm']

    ax3a.plot(x, d1disp1, 'o-', label='BS$_{\mathrm{ME}}$$^2$', mec='k', color=colors[0])
    ax3a.plot(x, d1disp2, 'o-', label='BS$_{\mathrm{GD}}$$^2$', mec='k', color=colors[1])
    ax3a.plot(x, d1disp3, 'o-', label='BZ$^2$', mec='k', color=colors[2])

    # plot_wfill(ax3a, x, d1disp1, d1disp1_std, 'BS$_{\mathrm{ME}}$$^2$', colors[0])
    # plot_wfill(ax3a, x, d1disp2, d1disp2_std, 'BS$_{\mathrm{GD}}$$^2$', colors[1])
    # plot_wfill(ax3a, x, d1disp3, d1disp3_std, 'BZ$^2$', colors[2])


    # # dyads 2
    d2disp1 = moddfs[sid].loc[moddfs[sid].index.str.startswith('ntmod_m1_n2_b'), 'd2disp_mean_wm']
    d2disp2 = moddfs[sid].loc[moddfs[sid].index.str.startswith('ntmod_m2_n2_b'), 'd2disp_mean_wm']
    d2disp3 = moddfs[sid].loc[moddfs[sid].index.str.startswith('ntmod_m3_n2_b'), 'd2disp_mean_wm']

    # ax3b.plot(x, d2disp1, 'o-', label='BS$_{\mathrm{ME}}$$^2$', mec='k', color=colors[0])
    # ax3b.plot(x, d2disp2, 'o-', label='BS$_{\mathrm{GD}}$$^2$', mec='k', color=colors[1])
    # ax3b.plot(x, d2disp3, 'o-', label='BZ$^2$', mec='k', color=colors[2])

    # ax3b.grid(alpha=0.25)
    # ax3b.legend(loc='best', fontsize='x-small')
    # ax3b.set_ylabel('Dispersion for\nFO 2, in WM')

    # d_std Plot
    d_modelstd_br2 = moddfs[sid].loc[moddfs[sid].index.str.startswith('ntmod_m2_n2_b'), 'd_modelstd_br']
    d_modelstd_br3 = moddfs[sid].loc[moddfs[sid].index.str.startswith('ntmod_m3_n2_b'), 'd_modelstd_br']

    ax3c.plot(x, 0*x, 'o-', label='BS$_{\mathrm{ME}}$$^2$', mec='k', color=colors[0])
    ax3c.plot(x, d_modelstd_br2, 'o-', label='BS$_{\mathrm{GD}}$$^2$', mec='k', color=colors[1])
    ax3c.plot(x, d_modelstd_br3, 'o-', label='BZ$^2$', mec='k', color=colors[2])


    ax3a.set_ylabel('Dispersion for\nFO 1, in WM')

    ax3a.grid(alpha=0.33, linewidth=0.5)
    for spine in ('right', 'top', 'bottom'):
        ax3a.spines[spine].set_visible(False)
    ax3a.tick_params(bottom=False)


    ax3c.grid(alpha=0.33, linewidth=0.5)
    for spine in ('right', 'top'):
        ax3c.spines[spine].set_visible(False)

    ax3c.set_xlabel(r'Burn-in ($\times 10^3$)')
    ax3c.set_ylabel(r'$d_{\mathrm{std}}$ (μm²/ms)')

    ax3c.legend(loc='lower center', bbox_to_anchor=(0.15, 1.03, .7, .10),
                ncol=3, borderaxespad=0., fontsize='x-small')

    fig3.savefig(f"Disp1+dstd-plots-{sid}.png", dpi=600)


    # Combined plot for main text: fsum, diff, Dispersion FO1, SD(d)
    figX, ((ax00, ax01), (ax10, ax11)) = plt.subplots(nrows=2, ncols=2,
                                                      sharex=True,
                                                      figsize=(7.0, 4.4),
                                                      dpi=150)
    figX.set_constrained_layout_pads(w_pad=1./72, h_pad=1./72, hspace=4./72, wspace=0./72) # units are inches; 1 pt = 1/72 of an inch

    bsme_lab = r'BS$_{\mathsf{ME}}$$^\mathsf{2}$'
    bsgd_lab = r'BS$_{\mathsf{GD}}$$^\mathsf{2}$'
    bz_lab = r'BZ$^\mathsf{2}$'

    plot_wfill(ax00, x, fsumwm1, fsumwm1_std, bsme_lab, colors[0])
    plot_wfill(ax00, x, fsumwm2, fsumwm2_std, bsgd_lab, colors[1])
    plot_wfill(ax00, x, fsumwm3, fsumwm3_std, bz_lab, colors[2])

    plot_wfill(ax10, x, dbr1, dbr1_std, bsme_lab, colors[0])
    plot_wfill(ax10, x, dbr2, dbr2_std, bsgd_lab, colors[1])
    plot_wfill(ax10, x, dbr3, dbr3_std, bz_lab, colors[2])

    ax10.plot(0, mdbr, ls='None', marker='x', mec='k', mfc='None', label='WDT MD')

    ax01.plot(x, d1disp1, 'o-', label=bsme_lab, mec='k', color=colors[0])
    ax01.plot(x, d1disp2, 'o-', label=bsgd_lab, mec='k', color=colors[1])
    ax01.plot(x, d1disp3, 'o-', label=bz_lab, mec='k', color=colors[2])

    ax11.plot(x, 0*x, 'o-', label=bsme_lab, mec='k', color=colors[0])
    ax11.plot(x, d_modelstd_br2, 'o-', label=bsgd_lab, mec='k', color=colors[1])
    ax11.plot(x, d_modelstd_br3, 'o-', label=bz_lab, mec='k', color=colors[2])

    # Labels
    ax00.set_ylabel('Restricted fraction\n' r'($\,f_{\mathrm{tot}}$), in WM')

    ax01.set_ylabel('Dispersion for\nFO 1, in WM')

    ax10.set_xlabel(r'Burn-in ($\times \mathsf{10^3}$)')
    ax10.set_ylabel(r'Mean diffusivity ($d$),' '\n' r'whole brain (μm²/ms)')

    ax11.set_xlabel(r'Burn-in ($\times \mathsf{10^3}$)')
    ax11.set_ylabel(r'$\mathbf{d}_{\mathrm{\mathbf{std}}}$ (μm²/ms)')

    # Grids, spines, ticks
    for ax in [ax00, ax10, ax01, ax11]:
        ax.grid(alpha=0.33, linewidth=0.5)

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    for ax in [ax00, ax01]:
        ax.spines['bottom'].set_visible(False)

        ax.tick_params(bottom=False)

    # Legends
    ax10.legend(loc='upper right', ncol=2, fontsize='x-small',
                columnspacing=1., handletextpad=0.5, borderaxespad=0.)

    ax11.legend(loc='upper center', ncol=3, fontsize='x-small',
                columnspacing=1., handletextpad=0.5, borderaxespad=0.)


    with open(f'Combined_plots-ftot_diff_disp1_dstd-{sid}-fig_obj.pkl', 'wb') as outfh:
        pickle.dump(figX, outfh)

    figX.savefig(f"Combined_plots-ftot_diff_disp1_dstd-{sid}.png", dpi=600)


    # Dispersion and STD(d) plots for supplemental
    figDisp, (axDispa, axDispb, axDispc) = plt.subplots(nrows=3, ncols=1, sharex=True,
                                                        figsize=(7.0, 7.0), dpi=150)
    figDisp.set_constrained_layout_pads(hspace=0.075, wspace=0.) # legend provides enough gap

    # Dispersion Plots (dyads<i>_dispersion) -- note this is dimensionless, comes from dot-product
    axDispa.plot(x, d1disp1, 'o-', label='BS$_{\mathrm{ME}}$$^2$', mec='k', color=colors[0])
    axDispa.plot(x, d1disp2, 'o-', label='BS$_{\mathrm{GD}}$$^2$', mec='k', color=colors[1])
    axDispa.plot(x, d1disp3, 'o-', label='BZ$^2$', mec='k', color=colors[2])

    # dyads 2
    axDispb.plot(x, d2disp1, 'o-', label='BS$_{\mathrm{ME}}$$^2$', mec='k', color=colors[0])
    axDispb.plot(x, d2disp2, 'o-', label='BS$_{\mathrm{GD}}$$^2$', mec='k', color=colors[1])
    axDispb.plot(x, d2disp3, 'o-', label='BZ$^2$', mec='k', color=colors[2])


    axDispa.set_ylabel('Dispersion for\nFO 1, in WM')

    axDispa.grid(alpha=0.33, linewidth=0.5)
    for spine in ('right', 'top', 'bottom'):
        axDispa.spines[spine].set_visible(False)
    axDispa.tick_params(bottom=False)

    axDispb.set_ylabel('Dispersion for\nFO 2, in WM')

    axDispb.grid(alpha=0.33, linewidth=0.5)
    for spine in ('right', 'top', 'bottom'):
        axDispb.spines[spine].set_visible(False)
    axDispb.tick_params(bottom=False)


    # d_std Plot
    axDispc.plot(x, 0*x, 'o-', label='BS$_{\mathrm{ME}}$$^2$', mec='k', color=colors[0])
    axDispc.plot(x, d_modelstd_br2, 'o-', label='BS$_{\mathrm{GD}}$$^2$', mec='k', color=colors[1])
    axDispc.plot(x, d_modelstd_br3, 'o-', label='BZ$^2$', mec='k', color=colors[2])


    axDispc.set_xlabel(r'Burn-in ($\times 10^3$)')
    axDispc.set_ylabel(r'$d_{\mathrm{std}}$  (μm²/ms)')

    axDispc.grid(alpha=0.33, linewidth=0.5)
    for spine in ('right', 'top'):
        axDispc.spines[spine].set_visible(False)

    axDispc.legend(loc='lower center', bbox_to_anchor=(0.15, 1.03, .7, .10),
                   ncol=3, borderaxespad=0., fontsize='x-small')

    figDisp.savefig(f"Disp-plots-{sid}.png", dpi=600)


    for f in fig1, fig2, fig3, figD, figDisp:
        plt.close(f)
