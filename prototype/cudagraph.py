#!/usr/bin/env python3

import sys
import csv
import numpy as np
import re
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg') # For headless use
plt.rc('text', usetex=True)

def plot_lib(ax, f, mem, style):
    with open(f) as csvfile:
        rd = csv.DictReader(csvfile)
        Hs = list(map(int, rd.fieldnames[1:]))

        for r in rd:
            M = r['M']
            runtimes = [int(r[str(H)]) for H in Hs]
            label = M
            if label[0] in ['_', '=']:
                label = '$M' + label + '$'

            w = 2

            if label == 'Auto':
                style = '-'
                w = 4

            ax.plot(Hs, runtimes,
                    linestyle=style,
                    linewidth=w,
                    label='{} ({})'.format(label, mem))

_, outputfile = sys.argv

hdw_shared_1_file='hdw_local_1.csv'
hdw_global_1_file='hdw_global_1.csv'
cas_shared_1_file='cas_local_1.csv'
cas_global_1_file='cas_global_1.csv'
xcg_shared_1_file='xcg_local_1.csv'
xcg_global_1_file='xcg_global_1.csv'

hdw_shared_63_file='hdw_local_63.csv'
hdw_global_63_file='hdw_global_63.csv'
cas_shared_63_file='cas_local_63.csv'
cas_global_63_file='cas_global_63.csv'
xcg_shared_63_file='xcg_local_63.csv'
xcg_global_63_file='xcg_global_63.csv'

fig, axes = plt.subplots(2,3, figsize=(10,5))
plt.subplots_adjust(hspace=0.3)

def ax_props(ax, title, ylabel=False, xlabel=True):
    if ylabel:
        ax.set_ylabel('Runtime in $\mu{}s$')
    if xlabel:
        ax.set_xlabel('H')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title(title)

plot_lib(axes[0,0], hdw_shared_1_file, 'shared', ':')
plot_lib(axes[0,0], hdw_global_1_file, 'global', '--')
ax_props(axes[0,0], 'HDW, race factor=1', ylabel=True, xlabel=False)

plot_lib(axes[0,1], cas_shared_1_file, 'shared', ':')
plot_lib(axes[0,1], cas_global_1_file, 'global', '--')
ax_props(axes[0,1], 'CAS, race factor=1', ylabel=False, xlabel=False)

plot_lib(axes[0,2], xcg_shared_1_file, 'shared', ':')
plot_lib(axes[0,2], xcg_global_1_file, 'global', '--')
ax_props(axes[0,2], 'XCG, race factor=1', ylabel=False, xlabel=False)

plot_lib(axes[1,0], hdw_shared_63_file, 'shared', ':')
plot_lib(axes[1,0], hdw_global_63_file, 'global', '--')
ax_props(axes[1,0], 'HDW, race factor=63', ylabel=True, xlabel=True)

plot_lib(axes[1,1], cas_shared_63_file, 'shared', ':')
plot_lib(axes[1,1], cas_global_63_file, 'global', '--')
ax_props(axes[1,1], 'CAS, race factor=63', ylabel=False, xlabel=True)

plot_lib(axes[1,2], xcg_shared_63_file, 'shared', ':')
plot_lib(axes[1,2], xcg_global_63_file, 'global', '--')
ax_props(axes[1,2], 'XCG, race factor=63', ylabel=False, xlabel=True)


axes[0,0].legend(loc='upper center', framealpha=1, ncol=6, fancybox=False,
                 bbox_to_anchor=(1.75, 1.5))

plt.savefig(outputfile, bbox_inches='tight')
