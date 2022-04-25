module load rocm/5.1.0

export PATH="$PATH:$HOME/llvm-spirv/install/bin"

export OPENCL64_FLAGS="-I $ROCM_PATH/opencl/include"
export OPENCL32_FLAGS="-I $ROCM_PATH/opencl/include"
export CXXFLAGS="-I $ROCM_PATH/opencl/include -DUSE_SPIRV=0"
export LDFLAGS="-L $ROCM_PATH/opencl/lib"

export USE_SPIRV=0
