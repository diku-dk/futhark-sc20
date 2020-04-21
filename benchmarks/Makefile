all: cuda_table_local_1.tex cuda_table_local_64.tex cuda_table_global_1.tex cuda_table_global_64.tex cuda_table.pdf

cuda_table_local_1.tex: cuda-prototype/hwd_local_1.csv cuda-prototype/cas_local_1.csv cuda-prototype/xcg_local_1.csv futhark/prototype.json
	python3 cudatable.py local 1 > $@
cuda_table_local_64.tex: cuda-prototype/hwd_local_64.csv cuda-prototype/cas_local_64.csv cuda-prototype/xcg_local_64.csv futhark/prototype.json
	python3 cudatable.py local 64 > $@

cuda_table_global_1.tex: cuda-prototype/hwd_global_1.csv cuda-prototype/cas_global_1.csv cuda-prototype/xcg_global_1.csv futhark/prototype.json
	python3 cudatable.py global 1 > $@
cuda_table_global_64.tex: cuda-prototype/hwd_global_64.csv cuda-prototype/cas_global_64.csv cuda-prototype/xcg_global_64.csv futhark/prototype.json
	python3 cudatable.py global 64 > $@

cuda_table.pdf: cudagraph.py futhark/prototype.json cuda-prototype/hwd_local_1.csv cuda-prototype/cas_local_1.csv cuda-prototype/xcg_local_1.csv cuda-prototype/hwd_global_1.csv cuda-prototype/cas_global_1.csv cuda-prototype/xcg_global_1.csv futhark/prototype.json cuda-prototype/hwd_local_64.csv cuda-prototype/cas_local_64.csv cuda-prototype/xcg_local_64.csv cuda-prototype/hwd_global_64.csv cuda-prototype/cas_global_64.csv cuda-prototype/xcg_global_64.csv
	 python3 cudagraph.py $@

.PHONY: clean

clean:
	(cd cuda-prototype; make clean)
	rm -f *.tex *.pdf
