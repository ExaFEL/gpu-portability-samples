CC ?= gcc
CXX ?= g++

# Note: this has to be a Clang with the experimental SPIR-V target enabled
OPENCL_CC ?= clang
OPENCL64_FLAGS ?=
OPENCL64_FLAGS += -Wall -fintegrated-objemitter -target spirv64v1.0
OPENCL32_FLAGS ?=
OPENCL32_FLAGS += -Wall -fintegrated-objemitter -target spirv32v1.0
LLVM_SPIRV ?= llvm-spirv
CXXFLAGS ?=
CXXFLAGS += -Wall -std=c++11
LDFLAGS ?=
LDFLAGS += -lOpenCL

BIN := saxpy histogram

ifeq ($(USE_SPIRV),1)
  SPIRV_DEPS := $(BIN:%=%_kernel32.spv) $(BIN:%=%_kernel64.spv) $(BIN:%=%_kernel32_t.spv) $(BIN:%=%_kernel64_t.spv) $(BIN:%=%_kernel32_t.bc) $(BIN:%=%_kernel64_t.bc)
  BIN += saxpy_t histogram_t
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

saxpy_t_main.o: saxpy_main.cc
	$(CXX) -c -o $@ $(CXXFLAGS) -DLOAD_SPIRV_T $<

histogram_t_main.o: histogram_main.cc
	$(CXX) -c -o $@ $(CXXFLAGS) -DLOAD_SPIRV_T $<

%64.spv:%.cl
	$(OPENCL_CC) -c -o $@ $(OPENCL64_FLAGS) $<

%32.spv:%.cl
	$(OPENCL_CC) -c -o $@ $(OPENCL32_FLAGS) $<

%64_t.bc:%.cl
	$(OPENCL_CC) -c -o $@ $(OPENCL64_FLAGS) -emit-llvm $<

%32_t.bc:%.cl
	$(OPENCL_CC) -c -o $@ $(OPENCL32_FLAGS) -emit-llvm $<

%_t.spv:%_t.bc
	$(LLVM_SPIRV) $< -o $@

.PHONY: clean
clean:
	rm -f *.o *.bc *.spv $(BIN)
