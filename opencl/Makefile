CC ?= gcc
CXX ?= g++

# Note: this has to be a Clang with the experimental SPIR-V target enabled
OPENCL_CC ?= clang
OPENCL64_FLAGS ?=
OPENCL64_FLAGS += -Wall -fintegrated-objemitter -target spirv64v1.0
OPENCL32_FLAGS ?=
OPENCL32_FLAGS += -Wall -fintegrated-objemitter -target spirv32v1.0
OPENCL64_T_FLAGS ?=
OPENCL64_T_FLAGS += -Wall -cl-std=CL1.2 -target spir64 -emit-llvm -Xclang -finclude-default-header -flto
OPENCL32_T_FLAGS ?=
OPENCL32_T_FLAGS += -Wall -cl-std=CL1.2 -target spir   -emit-llvm -Xclang -finclude-default-header -flto
LLVM_SPIRV ?= llvm-spirv
CXXFLAGS ?=
CXXFLAGS += -Wall -std=c++11
LDFLAGS ?=
LDFLAGS += -lOpenCL

BIN := saxpy histogram

ifeq ($(USE_SPIRV),1)
  SPIRV_DEPS := $(BIN:%=%_kernel32.spv) $(BIN:%=%_kernel64.spv) $(BIN:%=%_kernel32_t.spv) $(BIN:%=%_kernel64_t.spv) $(BIN:%=%_kernel32_t.ll) $(BIN:%=%_kernel64_t.ll)
  BIN += saxpy_t
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

%64.spv:%.cl
	$(OPENCL_CC) -c -o $@ $(OPENCL64_FLAGS) $<

%32.spv:%.cl
	$(OPENCL_CC) -c -o $@ $(OPENCL32_FLAGS) $<

%64_t.ll:%.cl
	$(OPENCL_CC) -c -o $@ $(OPENCL64_T_FLAGS) $<

%32_t.ll:%.cl
	$(OPENCL_CC) -c -o $@ $(OPENCL32_T_FLAGS) $<

%_t.spv:%_t.ll
	$(LLVM_SPIRV) $< -o $@

.PHONY: clean
clean:
	rm -f *.o *.ll *.spv $(BIN)
