all: cuda_table_local_1.tex cuda_table_local_63.tex cuda_table_global_1.tex cuda_table_global_63.tex cuda_table.pdf

cuda_table_local_1.tex: cuda-prototype/hdw_local_1.csv cuda-prototype/cas_local_1.csv cuda-prototype/xcg_local_1.csv futhark/prototype.json
	python3 cudatable.py local 1 > $@
cuda_table_local_63.tex: cuda-prototype/hdw_local_63.csv cuda-prototype/cas_local_63.csv cuda-prototype/xcg_local_63.csv futhark/prototype.json
	python3 cudatable.py local 63 > $@

cuda_table_global_1.tex: cuda-prototype/hdw_global_1.csv cuda-prototype/cas_global_1.csv cuda-prototype/xcg_global_1.csv futhark/prototype.json
	python3 cudatable.py global 1 > $@
cuda_table_global_63.tex: cuda-prototype/hdw_global_63.csv cuda-prototype/cas_global_63.csv cuda-prototype/xcg_global_63.csv futhark/prototype.json
	python3 cudatable.py global 63 > $@

cuda_table.pdf: cudagraph.py futhark/prototype.json cuda-prototype/hdw_local_1.csv cuda-prototype/cas_local_1.csv cuda-prototype/xcg_local_1.csv cuda-prototype/hdw_global_1.csv cuda-prototype/cas_global_1.csv cuda-prototype/xcg_global_1.csv futhark/prototype.json cuda-prototype/hdw_local_63.csv cuda-prototype/cas_local_63.csv cuda-prototype/xcg_local_63.csv cuda-prototype/hdw_global_63.csv cuda-prototype/cas_global_63.csv cuda-prototype/xcg_global_63.csv
	 python3 cudagraph.py $@

.PHONY: clean

clean:
	(cd cuda-prototype; make clean)
	rm -f *.tex *.pdf
