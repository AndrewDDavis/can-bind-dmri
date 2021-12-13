# coding: utf-8

# run this within an ipython session in the ~/Documents/Research_Projects/CAN-BIND_DTI dir.
# e.g. %run src/groupdiff-AOV_fdr_grid.py
# outputs group_results/merged_skel_v01-excl_outl/groupdiff-AOV_fdr_grids.pdf

import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Set up figure
# Make fig 90 mm wide for 1-column in biol. psych.
plt.rcParams['font.size'] = 10.
plt.rcParams['axes.titlepad'] = 18.0
# plt.rcParams['axes.labelpad'] = 2.0
# plt.rcParams['ytick.major.pad'] = 0.0
# plt.rcParams['xtick.major.pad'] = -4.0
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['axes.titlesize'] = 9
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=[5.51181, 2.0], dpi=150)
# fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=[3.543, 3.5], dpi=150)


# Find I/O dir
if os.path.isdir('group_results/merged_skel_v01'):
    iodir = 'group_results/merged_skel_v01/'
elif os.path.isdir('group_results/merged_skel_v01-excl_outl'):
    iodir = 'group_results/merged_skel_v01-excl_outl/'
else:
    raise ValueError('no iodir found')


# TRMT-CTRL p-vals
# read input p-values
tcdf = pd.read_csv(iodir + 'groupdiff-AOV_fdr_pvals.csv',
	               header=2, index_col=0, usecols=[0] + range(1,5),
	               names=['ROI', 'FA', 'MD', 'AD', 'RD'])
ltcdf = -1*np.log10(tcdf)


# Use pcolormesh and tweak the axes setup
im1 = ax1.pcolormesh(ltcdf, vmin=1.0, vmax=2.3,
	                 edgecolors='0.80', lw=0.5, cmap=plt.cm.Blues)

ax1.set_xticks(np.arange(0.5, len(ltcdf.columns), 1))
ax1.set_xticklabels(ltcdf.columns)
ax1.set_yticks(np.arange(0.5, len(ltcdf.index), 1))
ax1.set_yticklabels(ltcdf.index)

ax1.invert_yaxis()
ax1.xaxis.tick_top()
ax1.tick_params(axis='y', left=False, pad=0)
ax1.tick_params(axis='x', top=False, pad=-2.5)

ax1.set_title(u"MDD vs CTRL")

# colorbar
fig.colorbar(im1, ax=ax1, extend='both',
             fraction=0.08, orientation='horizontal', pad=0.03, aspect=16)

# annotate with pvalues in the boxes -- after tight layout improves?
for y in range(ltcdf.shape[0]):
    for x in range(ltcdf.shape[1]):
        pval = tcdf.iloc[y, x]
        tcolor = 'black'
        if pval < 0.011:
            tcolor = 'white'

        ax1.text(x + 0.5, y + 0.575, "{:.3f}".format(pval),
        	     ha='center', va='center', fontsize=8, color=tcolor)


# Trigroup ANOVA p-vals
# read input p-values for ROIs with significant group effects
tgdf = pd.read_csv(iodir + 'groupdiff-AOV_fdr_pvals.csv',
	               header=2, index_col=0, usecols=[0] + range(5,9),
	               names=['ROI', 'FA', 'MD', 'AD', 'RD'])
ltgdf = -1*np.log10(tgdf)

assert np.all(tgdf.index == tcdf.index)


# Use pcolormesh and tweak the axes setup
im2 = ax2.pcolormesh(ltgdf, vmin=1.0, vmax=2.3,
					 edgecolors='0.80', lw=0.5, cmap=plt.cm.Oranges)

ax2.set_xticks(np.arange(0.5, len(ltgdf.columns), 1))
ax2.set_xticklabels(ltgdf.columns)
ax2.set_yticks(np.arange(0.5, len(ltgdf.index), 1))
ax2.set_yticklabels(ltgdf.index)

ax2.invert_yaxis()
ax2.xaxis.tick_top()
ax2.yaxis.tick_right()
ax2.tick_params(axis='y', right=False, pad=0)
ax2.tick_params(axis='x', top=False, pad=-2.5)

ax2.set_title(u"CTRL/RESP/NONR ANOVA")

# colorbar
fig.colorbar(im2, ax=ax2, extend='both',
             fraction=0.08, orientation='horizontal', pad=0.03, aspect=16)

# annotate with pvalues in the boxes
for y in range(ltgdf.shape[0]):
    for x in range(ltgdf.shape[1]):
        pval = tgdf.iloc[y, x]
        tcolor = 'black'
        if pval < 0.011:
            tcolor = 'white'

        ax2.text(x + 0.5, y + 0.575, "{:.3f}".format(pval),
                 ha='center', va='center', fontsize=8, color=tcolor)


fig.tight_layout(rect=(0, 0, 1, 1))
fig.subplots_adjust(wspace=0.050, left=0.09, right=0.91, bottom=0.11, top=0.80)
fig.canvas.draw()

fig.savefig(iodir + 'groupdiff-AOV_fdr_grids.pdf', dpi=1000)

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
