#!/usr/bin/env python3
# coding: utf-8
"""Plot outputs of ntmod_stats_suppl for use in the manuscript; for details, see
dMRI Pre-Processing Journal.md.txt.

This script deals specifically with the CBN01-supplemental data.

Run from the CBN01-suppl dir.

Provide 1 arg (integer) for random seed if desired.
"""

import os, sys
from glob import glob
import pickle
import numpy as np
from scipy import stats as st
import pandas as pd
import matplotlib.pyplot as plt

# Plot look
def set_plot_params():
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 'medium'
    plt.rcParams['xtick.labelsize'] = 'small'
    plt.rcParams['ytick.labelsize'] = 'small'

    plt.rcParams['lines.markersize'] = 4.5
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

_, _, (blu_dark, blu_desat) = define_colours()


# Arg
if len(sys.argv) > 1:
    rnd_seed = int(sys.argv[1])

else:
    rnd_seed = 0

np.random.seed(rnd_seed)   # for repeatability of data-point locations


# Read data
fn = f'ntmod_stats-suppl_combined.csv'
df = pd.read_csv(fn, header=0)

subjids = np.unique(df.subjid)
N = len(subjids)

# For each subject, plot mean(d) and run-to-run SD(d) for models 1 and 2, WM and brain
# - for a simpler plot, and to match the other figures, just plot the whole-brain values
# fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(3.0, 4.0), dpi=150)
fig, ax = plt.subplots(figsize=(3.0, 4.0), dpi=150)

x_ticklocs = [0.8, 1.2]#, 1.8, 2.2]
x_list = [N*[x_ticklocs[0]], N*[x_ticklocs[1]]]#, N*[x_ticklocs[2]], N*[x_ticklocs[3]]]

m1_tf = (df.params == 'm1_n2_bi1000')
m2_tf = (df.params == 'm2_n2_bi1000')

d_m1_wm = df.loc[m1_tf, 'd_mean_wm']
dstd_m1_wm = df.loc[m1_tf, 'd_std_wm']

d_m1_br = df.loc[m1_tf, 'd_mean_br']
dstd_m1_br = df.loc[m1_tf, 'd_std_br']

d_m2_wm = df.loc[m2_tf, 'd_mean_wm']
dstd_m2_wm = df.loc[m2_tf, 'd_std_wm']

d_m2_br = df.loc[m2_tf, 'd_mean_br']
dstd_m2_br = df.loc[m2_tf, 'd_std_br']

# print some results
ratio_std_m2m1 = dstd_m2_br.values/dstd_m1_br.values

print("ratios of dstd_2 to dstd_1 in br:")
print(np.round(ratio_std_m2m1, 3))
print(f"mean/sd of ratios: {ratio_std_m2m1.mean():.3f} +/- {ratio_std_m2m1.std():.3f}")

ratio_d_m2m1 = d_m2_br.values/d_m1_br.values

print("ratios of d_2 to d_1 in br:")
print(np.round(ratio_d_m2m1, 3))
print(f"mean/sd of ratios: {ratio_d_m2m1.mean():.3f} +/- {ratio_d_m2m1.std():.3f}")

sys.exit(0)

def jitter(x, y, scl=0.04, rtype='uni'):
    '''give random variation to points so they don't completely overlap'''

    if rtype == 'uni':
        jit = np.random.uniform(low=-2*scl, high=2*scl, size=len(x))

    elif rtype == 'norm':
        # random variation
        jit = np.random.normal(scale=scl, size=len(x))

    else:
        raise ValueError(f'unrecognized rtype: {rtype}')

    # on the vertical, allow more jitter in high percentile regions
    gkde = st.gaussian_kde(y)
    ypdf = gkde.pdf(y)
    y_sc = ypdf/ypdf.max()

    x = np.array(x) + y_sc*jit

    return x


# define plot appearance params
alpha = 0.67
mkclr = blu_dark + (alpha,)
ebclr = blu_desat + (alpha,)
plt_dict = {'fmt':'o',
            'mec':mkclr, 'mfc':mkclr, 'ms':2,
            'ecolor':ebclr, 'elinewidth':1, 'mew':1, 'capsize':3}


# for i, (y, yerr) in enumerate(zip([d_m1_wm, d_m2_wm, d_m1_br, d_m2_br],
#                                   [dstd_m1_wm, dstd_m2_wm, dstd_m1_br, dstd_m2_br])):
# ax.clear()
for i, (y, yerr) in enumerate(zip([d_m1_br, d_m2_br], [dstd_m1_br, dstd_m2_br])):

    x = x_list[i]

    # if (i < 2):
    #     ax=ax1
    # else:
    #     ax=ax2

    # standardized plot of non-overlapping values with errorbars
    ax.errorbar(jitter(x, y, rtype='norm'), y, yerr=yerr, **plt_dict)


# labels
ax.set_ylabel(r'Diffusivity ($\mathit{d}$, μm$^\mathsf{2}$/ms)')
# ax2.yaxis.tick_right()

ax.set_xticks(x_ticklocs[0:2])
ax.set_xticklabels([r'BS$_{\mathsf{\mathrm{ME}}}$$^\mathsf{2}$', r'BS$_{\mathsf{\mathrm{GD}}}$$^\mathsf{2}$'])
ax.set_xlim(x_ticklocs[0]-0.15, x_ticklocs[1]+0.15)

ax.set_xlabel('Model')
ax.xaxis.labelpad = 8       # was 4

# ax.set_xticks(x_ticklocs[0:4])
# ax.set_xticklabels(['M1 WM', 'M2 WM', 'M1 Br', 'M2 Br'])
# ax.set_xlim(x_ticklocs[0]-0.15, x_ticklocs[3]+0.15)

# grid and minor grid
ax.minorticks_on()
ax.xaxis.set_tick_params(which='minor', bottom=False)
ax.yaxis.set_tick_params(which='minor', left=False)

ax.grid(alpha=0.3, axis='y')
ax.grid(which='minor', alpha=0.15, axis='y')
ax.set_axisbelow(True)


# wait if trying to find a good seed
yn = 'y'
if len(sys.argv) > 1:
    fig.show()
    yn = input("y to save this plot, enter to move on: ")

if yn == 'y':
    outfile_bn = f'ntmod_stats-suppl_diff_plot_sd{rnd_seed}'
    # save figure as png and pickle
    with open(outfile_bn + '.pkl', 'wb') as outfile:
        pickle.dump(fig, outfile)

    # later can be loaded as:
    # with open('ntmod_stats-suppl_diff_plot.pkl', 'rb') as infile:
    #     fig2 = pickle.load(infile)
    # ref: https://stackoverflow.com/questions/51264460/save-python-matplotlib-figure-as-pickle

    fig.savefig(outfile_bn + '.png', dpi=600)
