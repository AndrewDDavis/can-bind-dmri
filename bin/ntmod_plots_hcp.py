#!/usr/bin/env python3
# coding: utf-8
"""Plot outputs of ntmod_stats_hcp for use in the manuscript; for details, see
dMRI Pre-Processing Journal.md.txt.

Run from the HCP directory.
"""

import os, sys, pickle
from glob import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as tck

# Plot look
def set_plot_params():
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 'medium'
    plt.rcParams['axes.titlesize'] = 'medium'
    plt.rcParams['xtick.labelsize'] = 'small'
    plt.rcParams['ytick.labelsize'] = 'small'

    plt.rcParams['lines.markersize'] = 3.25
    plt.rcParams['legend.framealpha'] = 0.9

    # plt.rcParams['mathtext.fontset'] = 'STIX Two Text'
    plt.rcParams['mathtext.fontset'] = 'stix'         # to set individual subfamily values
    # plt.rcParams['mathtext.cal'] = 'stix:italic'
    # plt.rcParams['mathtext.cal'] = 'STIXGeneralItalic' # specify STIXGeneral on macOS

    plt.rcParams['figure.constrained_layout.use'] = True
    # plt.rcParams['figure.constrained_layout.w_pad'] = 3./72
    # plt.rcParams['figure.constrained_layout.h_pad'] = 3./72
    # plt.rcParams['figure.constrained_layout.wspace'] = 0.015
    # plt.rcParams['figure.constrained_layout.hspace'] = 0.015

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
# for fn in glob('ntmod_*_model_stats-*.csv'):
#     sid         = fn[-12:-4]                              # e.g. MGH_1031
#     sh_lab      = fn.split('_')[1]                        # e.g. b1000, 2shell
#     moddf       = pd.read_csv(fn, header=0, index_col=0)
#     moddfs[sid] = moddf

sids = ('MGH_1010', 'MGH_1019', 'MGH_1031')
sh_labs = ('b1000', 'b3000', '2shell')

# # just subj 1031 for now
# sid = sids[2]
# fn = f'ntmod_2shell_model_stats-{sid}.csv'

fn = f'ntmod_shellstr_model_stats-sidstr.csv'

for sid in sids:
    for sh_lab in sh_labs:

        fn_sh = fn.replace('shellstr', sh_lab).replace('sidstr', sid)
        moddf = pd.read_csv(fn_sh, header=0, index_col=0)

        moddfs[sid, sh_lab] = moddf

# sh_lab                = fn.split('_')[1]                        # e.g. b1000, 2shell

# Model plots array; one for each subject
for (sid, sh_lab) in moddfs.keys():

    if sh_lab != 'b1000':
        continue

    print(f'Array plots for {sid}...')

    # fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2,
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1,
                                   # sharex=True,
                                   figsize=(7.0, 5.5),
                                   dpi=150)
                                   # constrained_layout=True)
    # fig.set_constrained_layout_pads(w_pad=2./72, h_pad=2./72,
                                    # hspace=6./72, wspace=6./72)

    # Tissue Fraction Plot
    x = np.arange(1,13)  # bar locations

    fa_1 = moddfs[sid, 'b1000'].loc['tensor', 'fsum_mean_wm']
    fa_3 = moddfs[sid, 'b3000'].loc['tensor', 'fsum_mean_wm']
    fa_d = moddfs[sid, '2shell'].loc['tensor', 'fsum_mean_wm']

    fsumwm_vals = [fa_1, fa_3, fa_d,
                   moddfs[sid, 'b1000'].loc['m1_n2_bi50000', 'fsum_mean_wm'],
                   moddfs[sid, 'b3000'].loc['m1_n2_bi50000', 'fsum_mean_wm'],
                   moddfs[sid, '2shell'].loc['m1_n2_bi50000', 'fsum_mean_wm'],
                   moddfs[sid, 'b1000'].loc['m2_n2_bi50000', 'fsum_mean_wm'],
                   moddfs[sid, 'b3000'].loc['m2_n2_bi50000', 'fsum_mean_wm'],
                   moddfs[sid, '2shell'].loc['m2_n2_bi50000', 'fsum_mean_wm'],
                   moddfs[sid, 'b1000'].loc['m3_n2_bi50000', 'fsum_mean_wm'],
                   moddfs[sid, 'b3000'].loc['m3_n2_bi50000', 'fsum_mean_wm'],
                   moddfs[sid, '2shell'].loc['m3_n2_bi50000', 'fsum_mean_wm']]

    # fsumwm_std_vals = [0,
    #                    moddfs[(sid), ('b1000')].loc['m1_n2_bi1000', 'fsum_std_wm'],
    #                    moddfs[(sid), ('b1000')].loc['m2_n2_bi1000', 'fsum_std_wm'],
    #                    moddfs[(sid), ('b1000')].loc['m3_n2_bi1000', 'fsum_std_wm']]

    # ax1.bar(x, fsumwm_vals, width=0.8, yerr=fsumwm_std_vals, color=blu, capsize=2.5,
    #         tick_label=['WDT', 'BS1', 'BS2', 'BZ', 'BSS1', 'BSS2', 'BZZ'],
    #         zorder=1, label='All WM')

    ax1.bar(x, fsumwm_vals, width=0.8, color=blu,
            tick_label=['DT1', 'DT3', 'DTmsh', 'BSS1_1', 'BSS1_3', 'BSS1_msh', \
                        'BSS2_1', 'BSS2_3', 'BSS2_msh', 'BZZ_1', 'BZZ_3', 'BZZ_msh'],
            zorder=1, label='All WM')

    ax1.set_ylabel(r'Restricted fraction ($\mathit{\,f}_{\mathrm{tot}}$)' '\n' 'or FA, in WM')


    # Diffusivity Plot
    x = np.arange(1,13)  # bar locations

    md_1 = moddfs[sid, 'b1000'].loc['tensor', 'd_mean_br']
    md_3 = moddfs[sid, 'b3000'].loc['tensor', 'd_mean_br']
    md_d = moddfs[sid, '2shell'].loc['tensor', 'd_mean_br']

    dbr_vals = [md_1, md_3, md_d,
                moddfs[sid, 'b1000'].loc['m1_n2_bi50000', 'd_mean_br'],
                moddfs[sid, 'b3000'].loc['m1_n2_bi50000', 'd_mean_br'],
                moddfs[sid, '2shell'].loc['m1_n2_bi50000', 'd_mean_br'],
                moddfs[sid, 'b1000'].loc['m2_n2_bi50000', 'd_mean_br'],
                moddfs[sid, 'b3000'].loc['m2_n2_bi50000', 'd_mean_br'],
                moddfs[sid, '2shell'].loc['m2_n2_bi50000', 'd_mean_br'],
                moddfs[sid, 'b1000'].loc['m3_n2_bi50000', 'd_mean_br'],
                moddfs[sid, 'b3000'].loc['m3_n2_bi50000', 'd_mean_br'],
                moddfs[sid, '2shell'].loc['m3_n2_bi50000', 'd_mean_br']]

    # ax2.bar(x, dbr_vals, yerr=dbr_std_vals, color=blu, capsize=2.5,
    #         tick_label=['WDT', 'BS1', 'BS2', 'BZ', 'BSS1', 'BSS2', 'BZZ'])

    ax2.bar(x, dbr_vals, color=blu, capsize=2.5,
            tick_label=['DT1', 'DT3', 'DTmsh', 'BSS1_1', 'BSS1_3', 'BSS1_msh', \
                        'BSS2_1', 'BSS2_3', 'BSS2_msh', 'BZZ_1', 'BZZ_3', 'BZZ_msh'])

    ax2.set_ylabel(r'Mean diffusivity ($\mathit{d}$),' '\n' + r'whole brain (μm²/ms)')


    for ax in ax1, ax2:
        ax.grid(alpha=0.25, axis='y')
        ax.set_axisbelow(True)

        for tick in ax.get_xticklabels():
            tick.set_rotation(30)

    fig.suptitle('for BI = 50,000')

    # fig.show()
    with open(f'ntmod_plots_array-{sid}-fig_obj.pkl', 'wb') as outfh:
        pickle.dump(fig, outfh)

    fig.savefig(f"ntmod_plots_array-{sid}.png", dpi=600)
    continue


# Nomenclature; use similar language to Permeability diffusivity imaging (PDI)
# Restricted compartment (R) and unrestricted compartment (U); so we have diffusivities
# D_r and D_u

# plotting variable of burn-in values
x = np.array([1000, 2000, 10000, 50000])
x = x/1000.

# standardized plots
def plot_wfill(ax, x, y, yerr, lab, clr, ls='-'):
    ax.plot(x, y, ls=ls, marker='o', mec='k', mfc='None', label=lab, color=clr)
    ax.fill_between(x, y-yerr, y+yerr,
                    alpha=0.4, edgecolor='None', facecolor=clr)

# def plot_2ax(ax, axR, x, y, yerr, lab, clr, ls='-'):
#     ax.plot(x, y, ls=ls, marker='o', mec='k', mfc='None', label=lab, color=clr)

#     axR.plot(x, yerr, ls='--', marker='None', label=lab+' SD', color=clr)


# Burn-in comparison plots; one for each subject
for idx, (sid, sh_lab) in enumerate(moddfs):

    # if sid != 'UCA_0074_01':
    #     continue

    print(f'BI plots for {sid}, {sh_lab}...')

    # Diffusivity + fsum plots for main manuscript...
    if sh_lab == 'b1000':
        ls = '-'
        fig2, ((ax00, ax01, ax02), (ax10, ax11, ax12)) = plt.subplots(nrows=2, ncols=3,
                                                                      sharex=True, sharey='row',
                                                                      figsize=(7.48, 4.), dpi=150)
        fig2.set_constrained_layout_pads(hspace=0., wspace=0.) # legend provides enough gap

        f_ax = ax00
        d_ax = ax10
        sh_str = 'b=1000 s/mm²'

    elif sh_lab == 'b3000':
        ls = ':'
        f_ax = ax01
        d_ax = ax11
        sh_str = 'b=3000 s/mm²'

    else:
        ls = '-.'
        f_ax = ax02
        d_ax = ax12
        sh_str = 'Multi-shell'


    # Fsum
    fsumwm1 = moddfs[sid, sh_lab].loc[moddfs[sid, sh_lab].index.str.startswith('m1_n2_bi'), 'fsum_mean_wm']
    fsumwm2 = moddfs[sid, sh_lab].loc[moddfs[sid, sh_lab].index.str.startswith('m2_n2_bi'), 'fsum_mean_wm']
    fsumwm3 = moddfs[sid, sh_lab].loc[moddfs[sid, sh_lab].index.str.startswith('m3_n2_bi'), 'fsum_mean_wm']

    fsumwm1_std = moddfs[sid, sh_lab].loc[moddfs[sid, sh_lab].index.str.startswith('m1_n2_bi'), 'fsum_std_wm']
    fsumwm2_std = moddfs[sid, sh_lab].loc[moddfs[sid, sh_lab].index.str.startswith('m2_n2_bi'), 'fsum_std_wm']
    fsumwm3_std = moddfs[sid, sh_lab].loc[moddfs[sid, sh_lab].index.str.startswith('m3_n2_bi'), 'fsum_std_wm']

    plot_wfill(f_ax, x, fsumwm1, fsumwm1_std, r'BS$_{\mathrm{ME}}$²', colors[0], ls=ls)
    plot_wfill(f_ax, x, fsumwm2, fsumwm2_std, r'BS$_{\mathrm{GD}}$²', colors[1], ls=ls)
    plot_wfill(f_ax, x, fsumwm3, fsumwm3_std, r'BZ²', colors[2], ls=ls)


    # # d WM
    # mdwm = moddfs[sid, sh_lab].loc['tensor', 'd_mean_wm']

    # dwm1 = moddfs[sid, sh_lab].loc[moddfs[sid, sh_lab].index.str.startswith('m1_n2_bi'), 'd_mean_wm']
    # dwm2 = moddfs[sid, sh_lab].loc[moddfs[sid, sh_lab].index.str.startswith('m2_n2_bi'), 'd_mean_wm']
    # dwm3 = moddfs[sid, sh_lab].loc[moddfs[sid, sh_lab].index.str.startswith('m3_n2_bi'), 'd_mean_wm']

    # dwm1_std = moddfs[sid, sh_lab].loc[moddfs[sid, sh_lab].index.str.startswith('m1_n2_bi'), 'd_std_wm']
    # dwm2_std = moddfs[sid, sh_lab].loc[moddfs[sid, sh_lab].index.str.startswith('m2_n2_bi'), 'd_std_wm']
    # dwm3_std = moddfs[sid, sh_lab].loc[moddfs[sid, sh_lab].index.str.startswith('m3_n2_bi'), 'd_std_wm']

    # plot_wfill(ax2b, x, dwm1, dwm1_std, r'BS$_{\mathrm{ME}}$2', colors[0], ls=ls)
    # plot_wfill(ax2b, x, dwm2, dwm2_std, r'BS$_{\mathrm{GD}}$2', colors[1], ls=ls)
    # plot_wfill(ax2b, x, dwm3, dwm3_std, r'BZ2', colors[2], ls=ls)

    # # ax2b.plot(0, mdwm, ls='None', marker='x', mec='k', mfc='None', label='WDT MD')


    # d Brain
    mdbr = moddfs[sid, sh_lab].loc['tensor', 'd_mean_br']

    dbr1 = moddfs[sid, sh_lab].loc[moddfs[sid, sh_lab].index.str.startswith('m1_n2_bi'), 'd_mean_br']
    dbr2 = moddfs[sid, sh_lab].loc[moddfs[sid, sh_lab].index.str.startswith('m2_n2_bi'), 'd_mean_br']
    dbr3 = moddfs[sid, sh_lab].loc[moddfs[sid, sh_lab].index.str.startswith('m3_n2_bi'), 'd_mean_br']

    dbr1_std = moddfs[sid, sh_lab].loc[moddfs[sid, sh_lab].index.str.startswith('m1_n2_bi'), 'd_std_br']
    dbr2_std = moddfs[sid, sh_lab].loc[moddfs[sid, sh_lab].index.str.startswith('m2_n2_bi'), 'd_std_br']
    dbr3_std = moddfs[sid, sh_lab].loc[moddfs[sid, sh_lab].index.str.startswith('m3_n2_bi'), 'd_std_br']

    plot_wfill(d_ax, x, dbr1, dbr1_std, r'BS$_{\mathrm{ME}}$²', colors[0], ls=ls)
    plot_wfill(d_ax, x, dbr2, dbr2_std, r'BS$_{\mathrm{GD}}$²', colors[1], ls=ls)
    plot_wfill(d_ax, x, dbr3, dbr3_std, r'BZ²', colors[2], ls=ls)

    # ax2c.plot(0, mdbr, ls='None', marker='x', mec='k', mfc='None', label='WDT MD')

    # Labels and Titles
    f_ax.set_title(f'{sh_str}')
    d_ax.set_xlabel(r'Burn-in ($\times 10^3$)')

    # ax2b.set_ylabel(r'Mean diffusivity ($\mathcal{d}$),' '\n' r'in WM (μm²/ms)')
    # d_ax.set_ylabel(r'Mean diffusivity ($\mathcal{d}$),' '\n' r'whole brain (μm²/ms)')

    # put a legend between the subplots
    # ax00.legend(loc='lower center', bbox_to_anchor=(0.15, -0.15, .7, .10),
    # ax2b.legend(loc='upper right', ncol=4, borderaxespad=0., fontsize='x-small')

    if sh_lab == '2shell':

        ax00.legend(loc='lower right', ncol=1, fontsize='small')

        ax00.set_ylabel('Restricted fraction \n' r'($\mathit{\,f}_{\mathrm{tot}}$), in WM')
        ax10.set_ylabel(r'Mean diffusivity ($\mathit{d}$),' '\n' r'whole brain (μm²/ms)')

        for ax in [ax00, ax01, ax02, ax10, ax11, ax12]:
            ax.yaxis.set_minor_locator(tck.AutoMinorLocator())
            ax.tick_params(which='minor', left=False)

            ax.grid(alpha=0.3, linewidth=0.5)
            ax.grid(which='minor', alpha=0.15, linewidth=0.5)

        # turn off unused ticks
        # ax00.xaxis.set_visible(False)
        # ax.spines['right'].set_visible(False)
        ax00.tick_params(bottom=False)
        ax01.tick_params(bottom=False, left=False)
        ax02.tick_params(bottom=False, left=False)
        ax11.tick_params(left=False)
        ax12.tick_params(left=False)

        # fig2.show()
        # input('hit any key')
        # fig2.canvas.draw()

        with open(f'ntmod_burnin_plots-{sid}-fig_obj.pkl', 'wb') as outfh:
            pickle.dump(fig2, outfh)

        fig2.savefig(f'ntmod_burnin_plots-{sid}.png', dpi=600)
