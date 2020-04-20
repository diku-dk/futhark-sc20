COMPILER?=nvcc
CFLAGS?=-O4 
#-arch=compute_35 -prec-sqrt=true
LDFLAGS?=-lm

PROGRAM=gromacs-main

.PHONY: clean all run

$(PROGRAM): gromacs-main.cu  gromacs-kernels.cu.h gromacs-wrap.cu.h util.h scan-host-skel.cu.h scan-kernels.cu.h
	$(COMPILER) $(CFLAGS) -o $(PROGRAM) gromacs-main.cu

all: $(PROGRAM)

run: $(PROGRAM)
	./$(PROGRAM) data/huge-indarr.txt.in 36000 1600000 32000

clean:
	rm -f $(PROGRAM) *.o *.csv
