COMPILER?=nvcc
CFLAGS?=-O4 -arch=compute_35 -Wno-deprecated-declarations -DNONPP
LDFLAGS?=-lOpenCL

# nvcc -O4 -arch=compute_35 -Wno-deprecated-declarations -DNONPP test_image_8b_1C.cu -I../ -o test_image_8b_1C

PRG1H64  = test_image_8b_1C_H64
PRG1H256 = test_image_8b_1C_H256
PRG2H64  = test_image_8b_4C_H64
PRG2H256 = test_image_8b_4C_H256

.PHONY: clean all run

default: all

$(PRG1H64): test_image_8b_1C.cu cuda_histogram.h
	$(COMPILER) $(CFLAGS) -DTESTMAXIDX=64 -o $(PRG1H64) test_image_8b_1C.cu

$(PRG1H256): test_image_8b_1C.cu cuda_histogram.h
	$(COMPILER) $(CFLAGS) -DTESTMAXIDX=256 -o $(PRG1H256) test_image_8b_1C.cu

$(PRG2H64): test_image_8b_4C.cu cuda_histogram.h
	$(COMPILER) $(CFLAGS) -DTESTMAXIDX=64 -o $(PRG2H64) test_image_8b_4C.cu

$(PRG2H256): test_image_8b_4C.cu cuda_histogram.h
	$(COMPILER) $(CFLAGS) -DTESTMAXIDX=256 -o $(PRG2H256) test_image_8b_4C.cu

all: $(PRG1H64) $(PRG1H256) $(PRG2H64) $(PRG2H256)

run: $(PRG1H64) $(PRG1H256) $(PRG2H64) $(PRG2H256)
	./$(PRG1H64)  logo_small.raw --ch3
	./$(PRG1H256) logo_small.raw --ch3
	./$(PRG2H64)  logo_small.raw --ch3
	./$(PRG2H256) logo_small.raw --ch3

clean:
	rm -f $(PRG1H64) $(PRG1H256) $(PRG2H64) $(PRG2H256) *.o *.csv
