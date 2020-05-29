COMPILER?=nvcc
CFLAGS?=-O3 --compiler-options=-Wall --compiler-options=-Wextra -arch=compute_35 -std=c++11
LDFLAGS?=-lOpenCL

PROGRAM=example

.PHONY: clean all run

example: example.cu genhist.cu.h
	$(COMPILER) $(CFLAGS) -o $(PROGRAM) example.cu

all: $(PROGRAM)

run: $(PROGRAM)
	./$(PROGRAM) local
	./$(PROGRAM) global

clean:
	rm -f $(PROGRAM) *.o *.csv
