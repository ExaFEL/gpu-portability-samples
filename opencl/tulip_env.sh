module load rocm/4.0.0

export PATH="$PATH:$HOME/llvm-spirv/install/bin"

export OPENCL64_FLAGS="-I $ROCM_PATH/opencl/include"
export OPENCL32_FLAGS="-I $ROCM_PATH/opencl/include"
export CXXFLAGS="-I $ROCM_PATH/opencl/include -DUSE_SPIRV=0"
