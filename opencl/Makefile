CC := clang
CXX := clang++
LLVM_SPIRV ?= llvm-spirv

OPENCL64_FLAGS ?= -cl-std=CL1.2 -target spir64 -emit-llvm -Xclang -finclude-default-header -flto
OPENCL32_FLAGS ?= -cl-std=CL1.2 -target spir   -emit-llvm -Xclang -finclude-default-header -flto
CXXFLAGS ?= -std=c++11
LDFLAGS ?= -lOpenCL

BIN := saxpy
OBJ := main.o
OPENCL_OBJ := kernel64.spv

.PHONY: all
all:  $(BIN)

$(BIN): $(OBJ) $(OPENCL_OBJ)
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
