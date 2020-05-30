# Common Makefile options used by Futhark implementations.

FUTHARK=futhark

FUTHARK_BACKEND=cuda

# Note that these options are specific to the 'cuda' backend, so you
# may want to comment them out when using 'opencl'.
FUTHARK_BENCH_OPTIONS=--pass-option=--nvrtc-option=-arch=compute_35 --pass-option=--default-num-groups=272

# The 435.gromacs benchmark overrides this because it is too low there.
FUTHARK_BENCH_RUNS=100

%.json: %.fut
	$(FUTHARK) bench $^ $(FUTHARK_BENCH_OPTIONS) -r $(FUTHARK_BENCH_RUNS) --backend=$(FUTHARK_BACKEND) --json $@
	@echo
