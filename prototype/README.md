# The CUDA prototype for generalized histograms

If you have a working CUDA setup, then running `make` should produce
the file `figure_7.pdf`, corresponding to Figure 7 in the paper.  Note
that our figure was generated on an RTX 2080 Ti, so your results may
vary.

The graph generation script, `cudagraph.py`, requires Python 3, NumPy,
and Matplotlib.  You also need enough LaTeX for Matplotlib's LaTeX
backend to work.  On most Linux distributions, the "`texlive-full`"
package will suffice.

## TL;DR

Run `make` to generate `figure_7.pdf`.
