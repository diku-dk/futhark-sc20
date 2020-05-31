.NOTPARALLEL:

include ../futhark.mk

FUTHARK_BENCH_RUNS=1000

.PHONY: run

run: histogram64.seconds histogram256.seconds cuda-samples.json
	python3 report.py

cuda-samples-orig: cuda-samples-orig.cu
	nvcc -o $@ $< -O3 --compiler-options=-Wall

histogram64.seconds histogram256.seconds: cuda-samples-orig
	./cuda-samples-orig

.PHONY: clean

clean:
	rm -f *.json *.c cuda-samples cuda-samples-orig *.seconds
