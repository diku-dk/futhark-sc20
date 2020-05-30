# Artifact for *Compiling Generalized Histograms for GPU*

Each directory of this artifact contains a `README.md` explaining the
use of its contents (or a link to further information).

The artifact is divided into three main parts:

* The [library](library/) directory contains a CUDA library
  implementation of our generalized histograms, that can be used by
  other CUDA programs.  It also contains documentation and an example
  program.  It will not be discussed further.

* The [prototype](prototype) directory contains the CUDA prototoype
  that was used for the design space exploration of Section II and its
  validation in section IV.A, including reproducing Figure 7.

* The [benchmarks](benchmarks/) directory contains the benchmark
  programs used for Section IV in the paper.
