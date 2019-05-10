COMPILER?=nvcc
OPT_FLAGS?=-O3
LDFLAGS?=-lOpenCL

PROGRAM=genhisto

all: histo-main.cu histo-wrap.cu.h histo-kernels.cu.h
	$(COMPILER) $(OPT_FLAGS) -o $(PROGRAM) histo-main.cu
	./$(PROGRAM)

race-1: histo-main.cu histo-wrap.cu.h histo-kernels.cu.h
	$(COMPILER) $(OPT_FLAGS) -D RACE_FACT=1 -o $(PROGRAM) histo-main.cu
	./$(PROGRAM)

race-10: histo-main.cu histo-wrap.cu.h histo-kernels.cu.h
	$(COMPILER) $(OPT_FLAGS) -D RACE_FACT=10 -o $(PROGRAM) histo-main.cu
	./$(PROGRAM)

race-30: histo-main.cu histo-wrap.cu.h histo-kernels.cu.h
	$(COMPILER) $(OPT_FLAGS) -D RACE_FACT=30 -o $(PROGRAM) histo-main.cu
	./$(PROGRAM)

.PHONY: clean

clean:
	rm -f $(PROGRAM) *.o
