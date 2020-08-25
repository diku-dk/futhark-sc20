# Image registration

Implementation of the image registration benchmark from section V.A.4.
`img-reg-appl-orig.py` contains the original implementation written
with PyTorch.  The Futhark implementation is in `imgRetHisto.fut`.
Only the `mkImgRetHisto` entry point is used, and it is accessed
through the wrapper module `wrapFuthark.py`, which is ultimately
called by `img-reg-appl-wfut.py`.

**Beware: this benchmark runs on random data and does not perform any
validation.**

## TL;DR

Run `make` to reproduce the relevant part of Table III.  You need
Python 3 and PyTorch with a working CUDA setup.
