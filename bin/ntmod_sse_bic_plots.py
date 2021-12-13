#!/usr/bin/env python3
# coding: utf-8
"""Plot MSE and BIC values for use in the manuscript.

Values came from dmri_calc_ssebic, then combined in the spreadsheet dmri_SSE_BIC_allsubjs.ods. For
details, see DMRI Pre-Processing Journal.md.txt.

Run from ~/dmriproj/group_results/nontensor_models_analysis
"""

import os, sys
# from glob import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Plot look
def set_plot_params():
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 'medium'
    plt.rcParams['xtick.labelsize'] = 'small'
    plt.rcParams['ytick.labelsize'] = 'small'

    plt.rcParams['lines.markersize'] = 3.25
    plt.rcParams['legend.framealpha'] = 0.9

    # plt.rcParams['mathtext.fontset'] = 'dejavusans'
    plt.rcParams['mathtext.fontset'] = 'custom'     # to set individual subfamily values
    # plt.rcParams['mathtext.cal'] = 'stix:italic'  # not found on macOS
    # plt.rcParams['mathtext.cal'] = 'cm'           # computer modern -- did not work here, but works as math_fontfamily=... below

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

    # Blue, tan, maroon for the bars
    blu = colors[0]
    tan = '#C2883A'
    mrn = '#7F171F'

    blu_dark = (28/255, 48/255, 94/255)      # dark blue
    blu_desat = (59/255, 72/255, 105/255)    # desaturated

    return colors, (blu, tan, mrn), (blu_dark, blu_desat)

colors, (blu, tan, mrn), (blu_dark, blu_desat) = define_colours()


# Read data
sites = ('All', 'CAM', 'MCU', 'QNS', 'TGH', 'UBC', 'UCA')
sdfs = {}
for i,s in enumerate(sites):
    df = pd.read_excel("dmri_SSE_BIC_allsubjs.ods", engine="odf", sheet_name=i)

    sdfs[s] = df

av_df = sdfs['All']


# Set up figure
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2,
                               # sharex=True,
                               figsize=(7.0, 3.5),
                               dpi=150)
# fig.set_constrained_layout_pads(w_pad=2./72, h_pad=2./72,
                                # hspace=6./72, wspace=6./72)

x = np.arange(1,6)  # bar locations
xtick_labels = ['WDT', 'BS$_{\mathrm{ME}}$$^1$', 'BS$_{\mathrm{ME}}$$^2$', 'BZ$^1$', 'BZ$^2$']

w=0.8       # overall bar group width
p=0.6       # proportion of width for All_WM
bigbar_wdth = p*w
bigbar_ctrs = x+0.5*w*(p-1)
smbar_wdth = 0.5*w*(1-p)
smbar_ctrs1 = x+0.25*w*(3*p-1)
smbar_ctrs2 = x+0.25*w*(p+1)

# SSE
ax1.bar(bigbar_ctrs, av_df['SSE_WM'], width=bigbar_wdth, color=blu,
        zorder=1, label='All WM')

# also plot individual site values
for s in sites[1:]:
    sdf = sdfs[s]
    ax1.plot(bigbar_ctrs, sdf['SSE_WM'], 'ko', zorder=2, alpha=0.5)


ax1.bar(smbar_ctrs1, av_df['SSE_lin-WM'], width=smbar_wdth, color=tan,
                zorder=2, label='Linear')

ax1.bar(smbar_ctrs2, av_df['SSE_pla-WM'], width=smbar_wdth, color=mrn,
                zorder=3, label='Planar')

ax1.set_xticks(x)
ax1.set_xticklabels(xtick_labels)

# ax1.legend(ncol=1, loc='upper left', bbox_to_anchor=(1.0, 1.0),
#            fontsize='x-small', labelspacing=0.75, framealpha=1.0, edgecolor='none')

ax1.legend(ncol=1, loc='upper right', fontsize='x-small',
           framealpha=1.0, borderaxespad=0., labelspacing=0.75, handletextpad=0.5)

ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)

ax1.grid(alpha=0.25, axis='y')
ax1.set_axisbelow(True)

ax1.set_ylabel('Sum of squared errors (SSE)')


# BIC
ax2.bar(bigbar_ctrs, av_df['BIC_WM'], width=bigbar_wdth, color=blu,
        zorder=1, label='All WM')

# also plot individual site values
for s in sites[1:]:
    sdf = sdfs[s]
    ax2.plot(bigbar_ctrs, sdf['BIC_WM'], 'ko', zorder=2, alpha=0.5)


ax2.bar(smbar_ctrs1, av_df['BIC_lin-WM'], width=smbar_wdth, color=tan,
                zorder=2, label='Linear')

ax2.bar(smbar_ctrs2, av_df['BIC_pla-WM'], width=smbar_wdth, color=mrn,
                zorder=3, label='Planar')

ax2.set_xticks(x)
ax2.set_xticklabels(xtick_labels)
# ax2.legend(ncol=3, loc='upper right', fontsize='x-small', handletextpad=0.2, borderaxespad=0.2)

ax2.set_ylabel('Bayesian information criterion (BIC)')

ax2.grid(alpha=0.25, axis='y')
ax2.set_axisbelow(True)

ylims = list(ax2.get_ylim())
ylims[1] = -100.
ax2.set_ylim(ylims)

# for ax in ax1, ax2:
#     for tick in ax.get_xticklabels():
#         tick.set_rotation(30)

# should pickle here too
fig.savefig(f"ntmod_sse_bic_plots.png", dpi=600)
plt.close(fig)
