#!/usr/bin/env python3
# coding: utf-8
"""plot values from camino and bedpostx ball-stick model fits.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Plot setup
plt.rcParams['font.size'] = 9
plt.rcParams['axes.titlesize'] = 'medium'
plt.rcParams['axes.labelsize'] = 'medium'
plt.rcParams['xtick.labelsize'] = 'small'
plt.rcParams['ytick.labelsize'] = 'small'
plt.rcParams['lines.markersize'] = 3.

# match colour with other plots
clr = '#336699'

# read data
ddf = pd.read_csv("model_comparison-ball_stick-camino_vs_bpx-d_vals.csv",
                  header=0,
                  sep=r'\s+')
fdf = pd.read_csv("model_comparison-ball_stick-camino_vs_bpx-f_vals.csv",
                  header=0,
                  sep=r'\s+')


fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2,
                               figsize=(5.5, 3.5),  # 2/3 of page width with images beside it
                               dpi=150)


def ba_plot(ax, data, *args, **kwargs):
    """Accepts two-column data, creates Bland-Altman plot (AKA Tukey Mean-Difference plot)."""

    mean = data.mean(axis=1)        # column vector of means
    diff = data.bpx - data.camino   # column vector of differences
    md = np.mean(diff)
    sd = np.std(diff)

    ax.plot(mean, diff, 'o', color=clr, mew=1.0, alpha=0.5, zorder=2, *args, **kwargs)

    hline_kwargs = {'color': 'black', 'linestyle': '--', 'alpha': 0.4, 'zorder': 1}
    for yloc in [md, md + 1.96*sd, md - 1.96*sd]:
        ax.axhline(yloc, **hline_kwargs)

    low_list = np.nonzero(diff < md - 1.96*sd)[0].tolist()
    print("indices below limits of agreement: {} ({})".format(low_list, diff[low_list]))
    high_list = np.nonzero(diff > md + 1.96*sd)[0].tolist()
    print("indices above limits of agreement: {} ({})".format(high_list, diff[high_list]))

    return low_list, high_list

# diffusivity
ll, hl = ba_plot(ax1, ddf.loc[:, ['bpx', 'camino']])

ax1.set_xlim((0, 3.5))
ax1.set_title(r'Diffusivity (μm$^2$/ms)')
ax1.set_ylabel('Difference (MCMC−NLLS)')
ax1.set_xlabel('Mean')

print("d low_list:")
print(ddf.loc[ll])

print("d high_list:")
print(ddf.loc[hl])


# tissue-fraction
ll, hl = ba_plot(ax2, fdf.loc[:, ['bpx', 'camino']])

ax2.set_xlim((0, 1))
ax2.set_title('Restricted Fraction')
ax2.set_ylabel('Difference (MCMC−NLLS)')
ax2.set_xlabel('Mean')

print("f low_list:")
print(fdf.loc[ll])

print("f high_list:")
print(fdf.loc[hl])


fig.tight_layout()
fig.savefig('model_comparison-BA-plot.png', dpi=600)
# fig.show()
plt.close(fig)

