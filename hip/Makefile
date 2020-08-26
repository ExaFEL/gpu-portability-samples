HIPCC ?= hipcc

HIPCCFLAGS ?=
HIPCCFLAGS += -Wall
LDFLAGS ?=

BIN := saxpy histogram

.PHONY: all
all:  $(BIN)

$(BIN): %:%.cc
	$(HIPCC) -o $@ $(HIPCCFLAGS) $< $(LDFLAGS)

.PHONY: clean
clean:
	rm -f *.o $(BIN)
