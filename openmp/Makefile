CXXFLAGS ?= -Wall -std=c++11 -fopenmp
LDFLAGS ?= -fopenmp

BIN := saxpy histogram

.PHONY: all
all:  $(BIN)

$(BIN): %:%.cc
	$(CXX) -o $@ $(CXXFLAGS) $< $(LDFLAGS)

.PHONY: clean
clean:
	rm -f *.o $(BIN)
