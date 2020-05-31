#!/usr/bin/env python3

import json
import numpy as np

fut_results = json.load(open('histograms.json'))

def dataset(op, H, RF):
    if op == 'xcg':
        return '%di32 %di32 [50000000]i32 [50000000]i32' % (H, RF)
    else:
        return '%di32 %di32 [50000000]i32' % (H, RF)

def report(op):
    op_upper = op.upper()
    cub_fname = 'cub-%s.csv' % op
    res = {}
    for line in open(cub_fname).read().splitlines():
        H, cub_runtime = line.split(', ')
        H = float(H)
        cub_runtime = float(cub_runtime)
        fut = fut_results['histograms.fut:%s' % op]['datasets']
        fut_rf1_runtime = np.mean(fut[dataset(op, H, 1)]['runtimes'])
        fut_rf63_runtime = np.mean(fut[dataset(op, H, 63)]['runtimes'])
        res[H] = {'cub': cub_runtime,
                  'rf1': fut_rf1_runtime,
                  'rf63': fut_rf63_runtime}

    print(op_upper)

    print(9*' ', end='')
    for H in res:
        if H > 1000:
            s=' H=%dK' % (H/1000)
        else:
            s=' H=%d' % H
        print('%8s' % s, end='')
    print('')


    print('CUB      ', end='')
    for H in res:
        print(' %7.2f' % (res[H]['cub'] / 1000), end='')
    print('')

    print('FUT RF=1 ', end='')
    for H in res:
        cub=(res[H]['cub'] / 1000)
        speedup=cub/(res[H]['rf1'] / 1000)
        print(' %6.1fx' % speedup, end='')
    print('')

    print('FUT RF=63', end='')
    for H in res:
        cub=(res[H]['cub'] / 1000)
        speedup=cub/(res[H]['rf63'] / 1000)
        print(' %6.1fx' % speedup, end='')
    print('')

report('hdw')
print('')
report('cas')
print('')
report('xcg')
print('')
