# Benchmarks for *Compiling Generalized Histograms for GPU*

This directory contains Futhark implementations of various benchmarks
that make use of generalized histograms.  While we have made an effort
to make the Futhark parts easily runnable, most third party reference
implementations are large, complex, and fragile, and unfortunately not
packaged.

* [cub](cub/) contains Futhark and CUB implementations of histogram
  computations, corresponding to section IV.B of the paper.

* [435.gromacs](435.gromacs/) contains a Futhark port of an inner
  loop from the 435.gromacs benchmark from SPEC CPU2006.  This
  corresponds to section IV.C and part of Table III.

* [img-reg](img-reg/) contains a Futhark port of an image registration
  program, corresponding to section V.A.4 and part of Table III.

* [kmeans](kmeans/) contains a Futhark implementation of K-means
  clustering, corresponding to section IV.D.1 and part of Table III.

* [parboil](parboil/) contains a Futhark implementation of the
  *histo* and *tpacf* benchmarks from the Parboil benchmark suite,
  corresponding to section IV.D.2 and part of Table III.

* [cuda-samples](cuda-samples/) contains a Futhark implementation of
  the CUDA SDK samples of histograms, corresponding to section IV.D.3
  and part of Table III.

## Running

Running `make` in this directory will produce human-readable runtime
numbers **for the Futhark implementations only**.  To compare with
reference implementations (when applicable), run `make` in each
directory separately.
