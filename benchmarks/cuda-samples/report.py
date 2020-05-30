#!/usr/bin/env python3

import json
import numpy as np

histogram_64_cuda = float(open('histogram64.seconds').read())*1e6
histogram_256_cuda = float(open('histogram256.seconds').read())*1e6

futhark_results = json.load(open('cuda-samples.json'))

histogram_64_futhark = np.mean(list(map(float, futhark_results['cuda-samples.fut:bytes']['datasets']['[64]i32 63i32 [67108864]u8']['runtimes'])))
histogram_256_futhark = np.mean(list(map(float, futhark_results['cuda-samples.fut:bytes']['datasets']['[256]i32 255i32 [67108864]u8']['runtimes'])))

print('H=64  CUDA:   ', histogram_64_cuda)
print('H=64  Futhark:', histogram_64_futhark)
print('Speedup: %.2f' % (histogram_64_cuda / histogram_64_futhark))
print('')
print('H=256  CUDA:   ', histogram_256_cuda)
print('H=256  Futhark:', histogram_256_futhark)
print('Speedup: %.2f' % (histogram_256_cuda / histogram_256_futhark))
