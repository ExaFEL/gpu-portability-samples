NVCC ?= nvcc

NVCCFLAGS ?=
LDFLAGS ?=

BIN := saxpy histogram

.PHONY: all
all:  $(BIN)

$(BIN): %:%.cu
	$(NVCC) -o $@ $(NVCCFLAGS) $< $(LDFLAGS)

.PHONY: clean
clean:
	rm -f *.o $(BIN)
