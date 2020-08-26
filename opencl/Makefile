CC := clang
CXX := clang++
LLVM_SPIRV ?= llvm-spirv

OPENCL64_FLAGS ?= -Wall -cl-std=CL1.2 -target spir64 -emit-llvm -Xclang -finclude-default-header -flto
OPENCL32_FLAGS ?= -Wall -cl-std=CL1.2 -target spir   -emit-llvm -Xclang -finclude-default-header -flto
CXXFLAGS ?= -Wall -std=c++11
LDFLAGS ?= -lOpenCL

BIN := saxpy

.PHONY: all
all:  $(BIN)

$(BIN): $(BIN:%=%_main.o) $(BIN:%=%_kernel32.spv) $(BIN:%=%_kernel64.spv)
	$(CXX) -o $@ $(NVCCFLAGS) $< $(LDFLAGS)

%.o:%.cc
	$(CXX) -c -o $@ $(CXXFLAGS) $<

%64.ll:%.cl
	$(CC) -c -o $@ $(OPENCL64_FLAGS) $<

%32.ll:%.cl
	$(CC) -c -o $@ $(OPENCL32_FLAGS) $<

%.spv:%.ll
	$(LLVM_SPIRV) $< -o $@

.PHONY: clean
clean:
	rm -f *.o *.ll *.spv $(BIN)
