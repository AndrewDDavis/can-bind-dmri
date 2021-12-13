# coding: utf-8

# run this within an ipython session in the ~/Documents/Research_Projects/CAN-BIND_DTI dir.
# e.g. %run src/lmer_fdr_group_effect_and_tukey_grids.py
# outputs group_results/merged_skel_v01v02v03-excl_outl/lmer_fdr_group_effect_and_tukey_grids.pdf

import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Set up figure
# Make fig 90 mm wide for biol. psych.
plt.rcParams['font.size'] = 10.
plt.rcParams['axes.titlepad'] = 18.0
# plt.rcParams['axes.labelpad'] = 2.0
# plt.rcParams['ytick.major.pad'] = 0.0
# plt.rcParams['xtick.major.pad'] = -4.0
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['axes.titlesize'] = 9
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=[5.51181, 4.5], dpi=150)


# Find dir
if os.path.isdir('group_results/merged_skel_v01v02v03'):
    iodir = 'group_results/merged_skel_v01v02v03/'
elif os.path.isdir('group_results/merged_skel_v01v02v03-excl_outl'):
    iodir = 'group_results/merged_skel_v01v02v03-excl_outl/'
else:
    raise ValueError('no iodir found')


# Group effect p-vals
# read input p-values for ROIs with significant group effects
gedf = pd.read_csv(iodir + 'lmer_fdr_LLR_group_effect_pvals.csv', header=0, index_col=0)
lgedf = -1*np.log10(gedf)


# Use pcolormesh and tweak the axes setup
im1 = ax1.pcolormesh(lgedf, vmin=1.0, vmax=2.3, edgecolors='0.80', lw=0.5, cmap=plt.cm.Blues)

ax1.set_xticks(np.arange(0.5, len(lgedf.columns), 1))
ax1.set_xticklabels(lgedf.columns)
ax1.set_yticks(np.arange(0.5, len(lgedf.index), 1))
ax1.set_yticklabels(lgedf.index)

ax1.invert_yaxis()
ax1.xaxis.tick_top()
ax1.tick_params(axis='y', left=False, pad=0)
ax1.tick_params(axis='x', top=False, pad=-2.5)

ax1.set_title(u"Group Effects (CTRL/RESP/NONR)")

# colorbar
fig.colorbar(im1, ax=ax1, extend='both',
    fraction=0.03, orientation='horizontal', pad=0.03, aspect=16)

# annotate with pvalues in the boxes -- after tight layout improves?
for y in range(lgedf.shape[0]):
    for x in range(lgedf.shape[1]):
        pval = gedf.iloc[y, x]
        tcolor = 'black'
        if pval < 0.011:
            tcolor = 'white'

        ax1.text(x + 0.5, y + 0.575, "{:.3f}".format(pval), ha='center', va='center', fontsize=8, color=tcolor)


# Tukey HSD N-R pvals
# read input p-values
nrdf = pd.read_csv(iodir + 'lmer_fdr_tukeyhsd_NRpvals.csv', header=0, index_col=0)
lnrdf = -1*np.log10(nrdf)

assert np.all(nrdf.index == gedf.index)


# Use pcolormesh and tweak the axes setup
im2 = ax2.pcolormesh(lnrdf, vmin=1.0, vmax=2.3, edgecolors='0.80', lw=0.5, cmap=plt.cm.Oranges)

ax2.set_xticks(np.arange(0.5, len(lnrdf.columns), 1))
ax2.set_xticklabels(lnrdf.columns)
ax2.set_yticks(np.arange(0.5, len(lnrdf.index), 1))
ax2.set_yticklabels(lnrdf.index)

ax2.invert_yaxis()
ax2.xaxis.tick_top()
ax2.yaxis.tick_right()
ax2.tick_params(axis='y', right=False, pad=0)
ax2.tick_params(axis='x', top=False, pad=-2.5)

ax2.set_title(u"RESP–NONR Differences")

# colorbar
fig.colorbar(im2, ax=ax2, extend='both',
    fraction=0.03, orientation='horizontal', pad=0.03, aspect=16)

# annotate with pvalues in the boxes
for y in range(lnrdf.shape[0]):
    for x in range(lnrdf.shape[1]):
        pval = nrdf.iloc[y, x]
        if not np.isnan(pval):
            tcolor = 'black'
            if pval < 0.011:
                tcolor = 'white'
            ax2.text(x + 0.5, y + 0.575, "{:.3f}".format(pval), ha='center', va='center', fontsize=8, color=tcolor)


fig.tight_layout()
fig.subplots_adjust(wspace=0.050, left=0.10, right=0.90, bottom=0.05, top=0.91)
fig.canvas.draw()

fig.savefig(iodir + 'lmer_fdr_group_effect_and_tukey_grids.pdf', dpi=1000)

# fig.show()

# alternative ideas:
# # matshow
# fig, ax = plt.subplots()
# im = ax.matshow(lgedf)
# plt.colorbar(im, ax=ax)
# plt.yticks(np.arange(0.5, len(lgedf.index), 1), lgedf.index)
# plt.xticks(np.arange(0.5, len(lgedf.columns), 1), lgedf.columns)

# # seaborn heatmap
# import seaborn as sns
# fig, ax = plt.subplots(figsize=[3.25, 5.0], dpi=150)
# sns.heatmap(data=lgedf, ax=ax, cmap=plt.cm.Blues, vmin=0, vmax=2.6, linewidths=0.5, annot=True, cbar=True)
