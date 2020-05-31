COMPILER?=nvcc
CFLAGS?=-O3
LDFLAGS?=-lm

PROGRAM_HDW=gromacs-main-hdw
PROGRAM_CAS=gromacs-main-cas

.PHONY: clean all run

$(PROGRAM_HDW): gromacs-main.cu  gromacs-kernels.cu.h gromacs-wrap.cu.h util.h scan-host-skel.cu.h scan-kernels.cu.h
	$(COMPILER) $(CFLAGS) -D WITH_HDW=1 -o $(PROGRAM_HDW) gromacs-main.cu

$(PROGRAM_CAS): gromacs-main.cu  gromacs-kernels.cu.h gromacs-wrap.cu.h util.h scan-host-skel.cu.h scan-kernels.cu.h
	$(COMPILER) $(CFLAGS) -D WITH_HDW=0 -o $(PROGRAM_CAS) gromacs-main.cu

all: $(PROGRAM_HDW) $(PROGRAM_CAS)

hdw.seconds: $(PROGRAM_HDW)
	./$(PROGRAM_HDW) data/huge-indarr.txt.in 36000 1600000 32000 $@

cas.seconds: $(PROGRAM_CAS)
	./$(PROGRAM_CAS) data/huge-indarr.txt.in 36000 1600000 32000 $@

clean:
	rm -f $(PROGRAM_HDW) $(PROGRAM_CAS) *.o *.seconds
