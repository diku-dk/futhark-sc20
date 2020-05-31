.NOTPARALLEL:

include futhark.mk

JSONS=435.gromacs/gromacs-inl1100-hdw.json 435.gromacs/gromacs-inl1100-cas.json kmeans/kmeans.json parboil/tpacf.json parboil/histo.json cuda-samples/cuda-samples.json

run: $(JSONS)

435.gromacs/gromacs-inl1100-hdw.json:
	make -C 435.gromacs gromacs-inl1100-hdw.json

435.gromacs/gromacs-inl1100-cas.json:
	make -C 435.gromacs gromacs-inl1100-cas.json

kmeans/kmeans.json:
	make -C kmeans kmeans.json

parboil/tpacf.json:
	make -C parboil tpacf.json

parboil/histo.json:
	make -C parboil histo.json

cuda-samples/cuda-samples.json:
	make -C cuda-samples cuda-samples.json

.PHONY: clean

clean:
	rm -f $(JSONS)
