include futhark.mk

JSONS=435.gromacs/inl1100-6perit.json 435.gromacs/inl1100-withscan.json kmeans/kmeans.json parboil/tpacf.json parboil/histo.json cuda-samples/cuda-samples.json

run: $(JSONS)

.PHONY: clean

clean:
	rm -f $(JSONS)
