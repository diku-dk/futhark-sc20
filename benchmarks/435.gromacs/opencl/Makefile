COMPILER?=gcc
CFLAGS?=-O3
LDFLAGS?=-lOpenCL -lm

PROGRAM=gromacs-main

.PHONY: clean all run

$(PROGRAM): gromacs-main.cpp clutils.h gromacs-wrap.h SetupOpenCL.h util.h
	$(COMPILER) $(CFLAGS) -o $(PROGRAM) gromacs-main.cpp $(LDFLAGS)

all: $(PROGRAM)

run: $(PROGRAM)
	./$(PROGRAM) ../cuda/data/huge-indarr.txt.in 36000 1600000 32000

clean:
	rm -f $(PROGRAM) *.o *.csv
