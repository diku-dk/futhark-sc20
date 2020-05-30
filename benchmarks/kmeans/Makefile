include ../futhark.mk

all: kmeans.json

benchmark_data: benchmark_data.c
	gcc -o $@ $< -O

data: data/k1024.in data/k5.in

data/k1024.in: benchmark_data
	mkdir -p data
	rm -f $@
	echo 0 >> $@
	echo 1024 >> $@
	echo 1000 >> $@
	./benchmark_data 100000 256 >> $@

data/k5.in: benchmark_data
	mkdir -p data
	rm -f $@
	echo 0 >> $@
	echo 5 >> $@
	echo 1000 >> $@
	./benchmark_data 100000 256 >> $@

kmeans.json: kmeans.fut data

.PHONY: clean

clean:
	rm -f *.json kmeans.c kmeans benchmark_data
	rm -rf data
