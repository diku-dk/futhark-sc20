#!/usr/bin/env python3

import sys
import csv
import json
import numpy as np
import re

def read_futhark_results(mem, rf, f):
    all_results = json.load(open(f))
    hwd_results = all_results['prototype.fut:hwd']
    cas_results = all_results['prototype.fut:cas']
    xcg_results = all_results['prototype.fut:xcg']

    def on_results(results):
        Hs = {}
        for dataset in results['datasets']:
            H = int(re.search('([0-9]+)', dataset).group(1))
            Hs[H] = int(np.mean(results['datasets'][dataset]['runtimes'])*1000)
        return Hs

    return (on_results(hwd_results),
            on_results(cas_results),
            on_results(xcg_results))

def pH(H):
    if H > 1024*1024:
        return str(round(H / (1024*1024))) + 'M'
    elif H > 1024:
        return str(round(H / 1024)) + 'K'
    else:
        return str(H)

def pNum(x):
    if x > 10:
        return str(int(x)) + "."
    elif x > 1:
        return "{:.01f}".format(x)
    else:
        return "{:.02f}".format(x).lstrip("0")

def table_for(what, f, fut):

    with open(f) as csvfile:
        rd = csv.DictReader(csvfile)
        Hs = rd.fieldnames[1:]
        print(r'\begin{tabular}{|%s}\hline' % ('l|' * (len(Hs)+1)))
        print(r'\textsc{%s} ' % what, end='')

        for H in Hs:
            if H == Hs[0]:
                print(r'& \textbf{H=%s}' % pH(int(H)), end='')
            else:
                print(r' & %s' % pH(int(H)), end='')

        best = {}
        lines = []
        for r in rd:
            M = r['M']
            for H in Hs:
                if H in best:
                    best[H] = min(best[H], int(r[H]))
                else:
                    best[H] = int(r[H])
            lines += [r]

        print(r' \\\hline')

        def printHs(r):
            for H in Hs:
                runtime=int(r[H])
                runtime_s = pNum(runtime/1000)

                if abs(1-(float(runtime) / float(best[H]))) < 0.02:
                    print(r' & \textbf{%s}' % runtime_s, end='')
                else:
                    print(r' & %s' % runtime_s, end='')

        for r in lines:
            M = r['M']
            if M == 'Ours':
                print('Ours', end='')
            else:
                print('$M%s$' % M, end='')
            printHs(r)
            print(r' \\')
        print('Futhark', end='')
        printHs(r)
        print(r' \\')

        print(r'\hline')
        print(r'\end{tabular}')

mem=sys.argv[1]
rf=sys.argv[2]

hwd_file='cuda-prototype/hwd_{}_{}.csv'.format(mem, rf)
cas_file='cuda-prototype/cas_{}_{}.csv'.format(mem, rf)
xcg_file='cuda-prototype/xcg_{}_{}.csv'.format(mem, rf)

fut_hwd, fut_cas, fut_xcg = read_futhark_results(mem, rf, 'futhark/prototype.json')

table_for('hwd', hwd_file, fut_hwd)
table_for('cas', cas_file, fut_cas)
table_for('xcg', xcg_file, fut_xcg)
