include ../futhark.mk

FUTHARK_BENCH_RUNS=5000

run: gromacs-inl1100-hdw.json gromacs-inl1100-cas.json cuda/hdw.seconds cuda/cas.seconds
	python3 report.py

gromacs-inl1100-hdw.json: gromacs-inl1100-hdw.fut
	$(FUTHARK) bench --backend=$(FUTHARK_BACKEND) $(FUTHARK_BENCH_OPTIONS) -r 5000 $< --json $@

FUTHARK_OPENCL_BENCH_OPTIONS=--pass-option=--default-num-groups=$(FUTHARK_NUM_GROUPS)

gromacs-inl1100-cas.json: gromacs-inl1100-cas.fut
	$(FUTHARK) bench --backend=opencl $(FUTHARK_OPENCL_BENCH_OPTIONS) -r 5000 $< --json $@

cuda/hdw.seconds:
	make -C cuda hdw.seconds

cuda/cas.seconds:
	make -C cuda cas.seconds

.PHONY: clean

clean:
	rm -f *.json
	make -C cuda clean
