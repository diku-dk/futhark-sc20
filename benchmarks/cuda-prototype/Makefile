COMPILER?=nvcc
CFLAGS?=-O3 --compiler-options=-Wall
LDFLAGS?=-lOpenCL

PROGRAM=genhisto

.PHONY: clean all run

$(PROGRAM): histo-main.cu histo-wrap.cu.h histo-kernels.cu.h
	$(COMPILER) $(CFLAGS) -o $(PROGRAM) histo-main.cu

all: $(PROGRAM)

run: $(PROGRAM)
	./$(PROGRAM) local 1
	./$(PROGRAM) global 1
	./$(PROGRAM) local 64
	./$(PROGRAM) global 64

clean:
	rm -f $(PROGRAM) *.o
