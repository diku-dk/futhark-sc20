CXX        = nvcc -O3
CUB="../../cub-1.8.0/cub"

SRC1 = histo-atomicIncrem.cu
HELPERS1 = helper.cu.h
EXECUTABLE1 = histo-atomicIncrem

SRC2 = histo-saturIncrem.cu
HELPERS2 = helper.cu.h
EXECUTABLE2 = histo-saturIncrem

SRC3 = histo-keyValf64Add.cu
HELPERS3 = helper-keyValf64Add.cu.h
EXECUTABLE3 = histo-keyValf64Add

default: compile

.cu.o: $(SRC) $(HELPERS)
	$(CXX) -I$(CUB) -c $@ $<

compile: $(EXECUTABLE1) $(EXECUTABLE2) $(EXECUTABLE3)

$(EXECUTABLE1): $(SRC1) $(HELPERS1)
	$(CXX) -I$(CUB) -o $(EXECUTABLE1) $(SRC1)

$(EXECUTABLE2): $(SRC2) $(HELPERS2)
	$(CXX) -I$(CUB) -o $(EXECUTABLE2) $(SRC2)

$(EXECUTABLE3): $(SRC3) $(HELPERS3)
	$(CXX) -I$(CUB) -o $(EXECUTABLE3) $(SRC3)

clean:
	rm -f $(EXECUTABLE1) $(EXECUTABLE2) $(EXECUTABLE3)

