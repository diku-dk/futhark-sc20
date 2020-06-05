include ../futhark.mk

FUTHARK_BENCH_RUNS=10

KMCUDA=kmcuda-6.2.3

# Some old systems in wide use (RHEL...) require 'cmake3' to access
# cmake version 3.  Let's hack this.
ifeq ($(shell which cmake3),)
CMAKE=cmake
else
CMAKE=cmake3
endif

KMCUDA_CUDA_ARCH=75

run: k5.seconds k1024.seconds kmeans.json
	python3 report.py

benchmark_data: benchmark_data.c
	gcc -o $@ $< -O -std=c99

data: data/k1024.in data/k5.in

data/k1024.in: benchmark_data
	mkdir -p data
	rm -f $@
	futhark dataset -b -g 1024 -g '[100000][256]f32' > $@

data/k5.in: benchmark_data
	mkdir -p data
	rm -f $@
	futhark dataset -b -g 5 -g '[100000][256]f32' > $@

kmeans.json: kmeans.fut data
	$(FUTHARK) bench --backend=opencl $(FUTHARK_OPENCL_BENCH_OPTIONS) -r $(FUTHARK_BENCH_RUNS) $< --json $@

kmeans-kmcuda: kmeans-kmcuda.c $(KMCUDA)/src/libKMCUDA.so
	gcc $< -o $@ -O3 -std=c99 -I$(KMCUDA)/src $(KMCUDA)/src/libKMCUDA.so

%.seconds: kmeans-kmcuda data
	LD_LIBRARY_PATH=$(KMCUDA)/src/ ./kmeans-kmcuda data/$*.in $@

$(KMCUDA)/src/libKMCUDA.so: v6.2.3.tar.gz
	tar xf $<
	sed 's/-Werror//' -i $(KMCUDA)/src/CMakeLists.txt
	(cd $(KMCUDA)/src && \
	 $(CMAKE) . -DDISABLE_PYTHON=y -DDISABLE_R=y -DCUDA_ARCH=$(KMCUDA_CUDA_ARCH) && \
	 make -j)

v6.2.3.tar.gz:
	wget https://github.com/src-d/kmcuda/archive/v6.2.3.tar.gz

.PHONY: clean

clean:
	rm -f *.json kmeans.c kmeans benchmark_data v6.2.3.tar.gz
	rm -rf data
	rm -rf $(KMCUDA)
