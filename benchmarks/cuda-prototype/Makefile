COMPILER?=nvcc
OPT_FLAGS?=-O3
LDFLAGS?=-lOpenCL

PROGRAM=genhisto

.PHONY: clean all run

$(PROGRAM): histo-main.cu histo-wrap.cu.h histo-kernels.cu.h
	$(COMPILER) $(OPT_FLAGS) -o $(PROGRAM) histo-main.cu

all: $(PROGRAM)

run: $(PROGRAM)
	./$(PROGRAM) local 1
	./$(PROGRAM) global 1
	./$(PROGRAM) local 64
	./$(PROGRAM) global 64

clean:
	rm -f $(PROGRAM) *.o
