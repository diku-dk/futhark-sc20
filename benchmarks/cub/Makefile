CXX        = nvcc -O3
CUB?="../../../cub-1.8.0/cub"

EXECUTABLES=genhisto_i32 genhisto_i32_sat genhisto_f64

all: $(EXECUTABLES)

.cu.o: $(SRC) $(HELPERS)
	$(CXX) -I$(CUB) -c $@ $<

%: %.cu *.h
	$(CXX) -I$(CUB) -o $@ $<

clean:
	rm -f $(EXECUTABLES) *.o

