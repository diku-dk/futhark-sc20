COMPILER?=nvcc
CFLAGS?=-O4 
#-arch=compute_35 -prec-sqrt=true
LDFLAGS?=-lm

PROGRAM_HWD=gromacs-main-hwd
PROGRAM_CAS=gromacs-main-cas

.PHONY: clean all run

$(PROGRAM_HWD): gromacs-main.cu  gromacs-kernels.cu.h gromacs-wrap.cu.h util.h scan-host-skel.cu.h scan-kernels.cu.h
	$(COMPILER) $(CFLAGS) -D WITH_HDW=1 -o $(PROGRAM_HWD) gromacs-main.cu

$(PROGRAM_CAS): gromacs-main.cu  gromacs-kernels.cu.h gromacs-wrap.cu.h util.h scan-host-skel.cu.h scan-kernels.cu.h
	$(COMPILER) $(CFLAGS) -D WITH_HDW=0 -o $(PROGRAM_CAS) gromacs-main.cu

all: $(PROGRAM_HWD) $(PROGRAM_CAS)

run: $(PROGRAM_HWD) $(PROGRAM_CAS)
	./$(PROGRAM_HWD) data/huge-indarr.txt.in 36000 1600000 32000
	./$(PROGRAM_CAS) data/huge-indarr.txt.in 36000 1600000 32000

clean:
	rm -f $(PROGRAM_HWD) $(PROGRAM_CAS) *.o *.csv
