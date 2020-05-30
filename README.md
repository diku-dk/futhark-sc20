# Artifact for *Compiling Generalized Histograms for GPU*

The experiments in the paper comprise various GPU programs that
generally perform their own internal timing. In principle,
reproduction should be as easy as compiling and running them. Most of
the GPU code is written in ordinary CUDA, but some is written in
Futhark, the data-parallel language that is also partially the subject
of the paper. An appropriate version of the Futhark compiler is also
needed. The paper results have been produced with Futhark 0.15.8.

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

## Paper environment

We have run the experiments in the paper on a server-class computer
with an Intel(R) Xeon(R) CPU E5-2650 v2 CPU, but more importantly an
NVIDIA RTX 2080 Ti GPU. The operating system was RHEL 7.7 and we used
CUDA 10.1 for the GPU interaction.  You will need a UTF-8 locale.
