.NOTPARALLEL:

COMPILER?=nvcc
CFLAGS?=-O3 --compiler-options=-Wall
LDFLAGS?=-lOpenCL

PROGRAM=genhisto

.PHONY: clean all run

all: table_7.pdf

$(PROGRAM): histo-main.cu histo-wrap.cu.h histo-kernels.cu.h
	$(COMPILER) $(CFLAGS) -o $(PROGRAM) histo-main.cu

run: hdw_local_1.csv hdw_global_1.csv hdw_local_63.csv hdw_global_63.csv

hdw_local_1.csv cas_local_1.csv xcg_local_1.csv: $(PROGRAM)
	./$(PROGRAM) local 1 hdw_local_1.csv cas_local_1.csv xcg_local_1.csv

hdw_global_1.csv cas_global_1.csv xcg_global_1.csv: $(PROGRAM)
	./$(PROGRAM) global 1 hdw_global_1.csv cas_global_1.csv xcg_global_1.csv

hdw_local_63.csv cas_local_63.csv xcg_local_63.csv: $(PROGRAM)
	./$(PROGRAM) local 63 hdw_local_63.csv cas_local_63.csv xcg_local_63.csv

hdw_global_63.csv cas_global_63.csv xcg_global_63.csv: $(PROGRAM)
	./$(PROGRAM) global 63 hdw_global_63.csv cas_global_63.csv xcg_global_63.csv

table_7.pdf: cudagraph.py hdw_local_1.csv cas_local_1.csv xcg_local_1.csv hdw_global_1.csv cas_global_1.csv xcg_global_1.csv hdw_local_63.csv cas_local_63.csv xcg_local_63.csv hdw_global_63.csv cas_global_63.csv xcg_global_63.csv
	 python3 cudagraph.py $@

clean:
	rm -f $(PROGRAM) *.o *.csv
	rm -f *.tex *.pdf
