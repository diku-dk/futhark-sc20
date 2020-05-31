# k-means benchmark

A Futhark implementation of k-means clustering written with
generalized histograms ([kmeans.fut](kmeans.fut)), compared to an
implementation that makes calls to the
[kmcuda](https://github.com/src-d/kmcuda) library
([kmeans-kmcuda.c](kmeans-kmcuda.c)).

The Makefile here should automatically download and compile the kmcuda
library.  Note that kmcuda appears very picky about the NVIDIA compute
architecture, which must match the hardware being used, and specified
at compile time.  You may need to modify the `KMCUDA_CUDA_ARCH`
variable in [Makefile](Makefile) to suit your GPU.

## TL;DR

Run `make` to produce a summary of the runtimes, including a speedup,
corresponding to the entries on Table III.
