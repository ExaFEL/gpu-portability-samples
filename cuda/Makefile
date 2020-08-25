NVCC ?= nvcc

NVCCFLAGS ?=
LDFLAGS ?=

BIN := saxpy

.PHONY: all
all:  $(BIN)

$(BIN): %:%.cu
	$(NVCC) -o $@ $(NVCCFLAGS) $< $(LDFLAGS)

.PHONY: clean
clean:
	rm -f *.o $(BIN)
