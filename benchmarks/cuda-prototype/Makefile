COMPILER?=nvcc
OPT_FLAGS?=-O3
LDFLAGS?=-lOpenCL

PROGRAM=genhisto

.PHONY: clean all run

$(PROGRAM): histo-main.cu histo-wrap.cu.h histo-kernels.cu.h
	$(COMPILER) $(OPT_FLAGS) -o $(PROGRAM) histo-main.cu

all: $(PROGRAM)

run: $(PROGRAM)
	./$(PROGRAM)

race-1: histo-main.cu histo-wrap.cu.h histo-kernels.cu.h
	$(COMPILER) $(OPT_FLAGS) -D RF=1 -o $(PROGRAM) histo-main.cu
	./$(PROGRAM)

race-10: histo-main.cu histo-wrap.cu.h histo-kernels.cu.h
	$(COMPILER) $(OPT_FLAGS) -D RF=10 -o $(PROGRAM) histo-main.cu
	./$(PROGRAM)

race-30: histo-main.cu histo-wrap.cu.h histo-kernels.cu.h
	$(COMPILER) $(OPT_FLAGS) -D RF=30 -o $(PROGRAM) histo-main.cu
	./$(PROGRAM)

clean:
	rm -f $(PROGRAM) *.o
