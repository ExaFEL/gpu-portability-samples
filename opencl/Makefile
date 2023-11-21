CC ?= gcc
CXX ?= g++

# Note: this has to be a Clang with the experimental SPIR-V target enabled
OPENCL_CC ?= clang
OPENCL64_FLAGS ?=
OPENCL64_FLAGS += -Wall -fintegrated-objemitter -target spirv64v1.0
OPENCL32_FLAGS ?=
OPENCL32_FLAGS += -Wall -fintegrated-objemitter -target spirv32v1.0
CXXFLAGS ?=
CXXFLAGS += -Wall -std=c++11
LDFLAGS ?=
LDFLAGS += -lOpenCL

BIN := saxpy histogram

ifeq ($(USE_SPIRV),1)
  SPIRV_DEPS := $(BIN:%=%_kernel32.spv) $(BIN:%=%_kernel64.spv)
  CXXFLAGS += -DUSE_SPIRV=1
else
  SPIRV_DEPS :=
  CXXFLAGS += -DUSE_SPIRV=0
endif

.PHONY: all
all:  $(BIN)

$(BIN): %:%_main.o $(SPIRV_DEPS)
	$(CXX) -o $@ $(CXXFLAGS) $< $(LDFLAGS)

%.o:%.cc
	$(CXX) -c -o $@ $(CXXFLAGS) $<

%64.spv:%.cl
	$(OPENCL_CC) -c -o $@ $(OPENCL64_FLAGS) $<

%32.spv:%.cl
	$(OPENCL_CC) -c -o $@ $(OPENCL32_FLAGS) $<

.PHONY: clean
clean:
	rm -f *.o *.ll *.spv $(BIN)
