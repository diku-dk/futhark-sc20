COMPILER?=nvcc
CFLAGS?=-O3 --compiler-options=-Wall
LDFLAGS?=-lOpenCL

PROGRAM=genhisto

.PHONY: clean all run

$(PROGRAM): histo-main.cu histo-wrap.cu.h histo-kernels.cu.h
	$(COMPILER) $(CFLAGS) -o $(PROGRAM) histo-main.cu

all: $(PROGRAM)

run: $(PROGRAM)
	./$(PROGRAM) local 1 hwd_local_1.csv cas_local_1.csv xcg_local_1.csv
	./$(PROGRAM) global 1 hwd_global_1.csv cas_global_1.csv xcg_global_1.csv
	./$(PROGRAM) local 64 hwd_local_64.csv cas_local_64.csv xcg_local_64.csv
	./$(PROGRAM) global 64 hwd_global_64.csv cas_global_64.csv xcg_global_64.csv

clean:
	rm -f $(PROGRAM) *.o *.csv
