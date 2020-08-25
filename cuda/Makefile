NVCC ?= nvcc

DEBUG ?= 0

NVCCFLAGS ?=

LDFLAGS ?=

ifeq ($(strip $(DEBUG)),0)
	NVCCFLAGS += -O3
else
	NVCCFLAGS += -O0
endif

BIN := saxpy

.PHONY: all
all:  $(BIN)

$(BIN): %:%.cu
	$(NVCC) -o $@ $(NVCCFLAGS) $< $(LDFLAGS)

.PHONY: clean
clean:
	rm -f *.o $(BIN)
