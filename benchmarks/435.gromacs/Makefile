include ../futhark.mk

run: gromacs-inl1100-hdw.json gromacs-inl1100-cas.json

gromacs-inl1100-hdw.json: gromacs-inl1100-hdw.fut
	$(FUTHARK) bench --backend=$(FUTHARK_BACKEND) $(FUTHARK_BENCH_OPTIONS) -r 5000 $< --json $@

gromacs-inl1100-cas.json: gromacs-inl1100-cas.fut
	$(FUTHARK) bench --backend=$(FUTHARK_BACKEND) $(FUTHARK_BENCH_OPTIONS) -r 5000 $< --json $@

.PHONY: clean
	rm -f *.json
