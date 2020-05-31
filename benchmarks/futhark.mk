# Common Makefile options used by Futhark implementations.

LOCAL_FUTHARK=$(shell [ -x ../../bin/futhark ] && echo yes)

ifeq ($(LOCAL_FUTHARK),yes)
	FUTHARK=../../bin/futhark
else
	FUTHARK=futhark
endif

FUTHARK_BACKEND=cuda

# Note that these options are specific to the 'cuda' backend, so you
# may want to comment them out when using 'opencl'.
FUTHARK_BENCH_OPTIONS=--pass-option=--nvrtc-option=-arch=compute_35 --pass-option=--default-num-groups=272

# Some benchmarks override this if they run very quickly and need more
# iterations for stable results.
FUTHARK_BENCH_RUNS=100

%.json: %.fut
	$(FUTHARK) bench $^ $(FUTHARK_BENCH_OPTIONS) -r $(FUTHARK_BENCH_RUNS) --backend=$(FUTHARK_BACKEND) --json $@
	@echo
