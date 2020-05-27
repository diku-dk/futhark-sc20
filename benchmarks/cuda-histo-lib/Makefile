COMPILER?=nvcc
CFLAGS?=-O3 --compiler-options=-Wall -arch=compute_35
LDFLAGS?=-lOpenCL

PROGRAM=genhisto

.PHONY: clean all run

$(PROGRAM): histo-main.cu histo-wrap.cu.h histo-kernels.cu.h
	$(COMPILER) $(CFLAGS) -o $(PROGRAM) histo-main.cu

all: $(PROGRAM)

run: $(PROGRAM)
	./$(PROGRAM) local  hdw_local_1.csv cas_local_1.csv xcg_local_1.csv hdw_local_63.csv cas_local_63.csv xcg_local_63.csv
	./$(PROGRAM) global hdw_global_1.csv cas_global_1.csv xcg_global_1.csv hdw_global_63.csv cas_global_63.csv xcg_global_63.csv

clean:
	rm -f $(PROGRAM) *.o *.csv
