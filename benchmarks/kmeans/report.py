#!/usr/bin/env python3

import json
import numpy as np

k5_kmcuda = float(open('k5.seconds').read())
k1024_kmcuda = float(open('k1024.seconds').read())

futhark_results = json.load(open('kmeans.json'))

k5_futhark = np.mean(list(map(float, futhark_results['kmeans.fut']['datasets']['data/k5.in']['runtimes'])))/1e6
k1024_futhark = np.mean(list(map(float, futhark_results['kmeans.fut']['datasets']['data/k1024.in']['runtimes'])))/1e6

print('k=5 kmcuda: ', k5_kmcuda)
print('k=5 Futhark:', k5_futhark)
print('Speedup: %.2f' % (k5_kmcuda / k5_futhark))
print('')
print('k=1024  kmcuda: ', k1024_kmcuda)
print('k=1024  Futhark:', k1024_futhark)
print('Speedup: %.2f' % (k1024_kmcuda / k1024_futhark))
