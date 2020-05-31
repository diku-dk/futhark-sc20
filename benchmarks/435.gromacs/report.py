#!/usr/bin/env python3

import json
import numpy as np

cas_cuda = float(open('cuda/cas.seconds').read())*1e6
hdw_cuda = float(open('cuda/hdw.seconds').read())*1e6

cas_futhark_results = json.load(open('gromacs-inl1100-cas.json'))
hdw_futhark_results = json.load(open('gromacs-inl1100-hdw.json'))

cas_futhark = np.mean(list(map(float, cas_futhark_results['gromacs-inl1100-cas.fut']['datasets']['data/all-huge.in']['runtimes'])))
hdw_futhark = np.mean(list(map(float, hdw_futhark_results['gromacs-inl1100-hdw.fut']['datasets']['data/all-huge.in']['runtimes'])))

print('HDW CUDA:   ', hdw_cuda)
print('HDW Futhark:', hdw_futhark)
print('Speedup: %.2f' % (hdw_cuda / hdw_futhark))
print('')
print('CAS CUDA:   ', cas_cuda)
print('CAS Futhark:', cas_futhark)
print('Speedup: %.2f' % (cas_cuda / cas_futhark))
