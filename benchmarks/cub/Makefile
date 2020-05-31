include ../futhark.mk

CUB=cub-1.8.0

all: cub-hdw.csv cub-cas.csv cub-xcg.csv histograms.json
	python3 report.py

cub-hdw.csv: genhisto_hdw
	./cub-bench.sh ./genhisto_hdw > $@

cub-cas.csv: genhisto_cas
	./cub-bench.sh ./genhisto_cas > $@

cub-xcg.csv: genhisto_xcg
	./cub-bench.sh ./genhisto_xcg > $@

%: %.cu *.h $(CUB)
	nvcc -I$(CUB)/cub -o $@ $<

cub-1.8.0: | 1.8.0.tar.gz
	tar xf 1.8.0.tar.gz

1.8.0.tar.gz:
	wget https://github.com/NVlabs/cub/archive/1.8.0.tar.gz

clean:
	rm -f genhisto_cas genhisto_hdw genhisto_xcg *.csv
	rm -rf cub-1.8.0
	rm -f 1.8.0.tar.gz
