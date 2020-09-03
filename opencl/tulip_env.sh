module load rocm

export OPENCL64_FLAGS="-I $ROCM_PATH/opencl/include"
export OPENCL32_FLAGS="-I $ROCM_PATH/opencl/include"
export CXXFLAGS="-I $ROCM_PATH/opencl/include"
