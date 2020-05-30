# CUDA Samples benchmark

[cuda-samples.fut](cuda-samples.fut) contains a Futhark implementation
of the `histogram` example from the CUDA SDK (usually located in
`/usr/local/cuda/samples/3_Imaging/histogram/`).  It contains two
versions: `words` and `bytes`.  The numbers we report in the paper is
from `bytes`, which performs (marginally) better on the hardware we
have tried.

[cuda-samples-orig.cu](cuda-samples-orig.cu) contains a concatenation
of the files making up the original CUDA SDK program, with a few
changes to compile without SDK-specific libraries, and to collect
machine-readable runtime results.

## TL;DR

Run `make` to produce a summary of the runtimes, including a speedup,
corresponding to the entries on Table III.
