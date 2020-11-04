module load oneapi # Important: must come before hipcl

module use /soft/modulefiles
module load hipcl

export HIPCC=clang++
export HIPCCFLAGS=-std=c++11
export LDFLAGS="-lhipcl -lOpenCL"
