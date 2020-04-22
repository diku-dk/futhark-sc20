COMPILER?=nvcc
CFLAGS?=-O3 --compiler-options=-Wall
LDFLAGS?=-lOpenCL

PROGRAM=genhisto

.PHONY: clean all run

$(PROGRAM): histo-main.cu histo-wrap.cu.h histo-kernels.cu.h
	$(COMPILER) $(CFLAGS) -o $(PROGRAM) histo-main.cu

all: $(PROGRAM)

run: $(PROGRAM)
	./$(PROGRAM) local 1 hdw_local_1.csv cas_local_1.csv xcg_local_1.csv
	./$(PROGRAM) global 1 hdw_global_1.csv cas_global_1.csv xcg_global_1.csv
	./$(PROGRAM) local 63 hdw_local_63.csv cas_local_63.csv xcg_local_63.csv
	./$(PROGRAM) global 63 hdw_global_63.csv cas_global_63.csv xcg_global_63.csv

clean:
	rm -f $(PROGRAM) *.o *.csv
