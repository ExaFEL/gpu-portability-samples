module load rocm/3.9.0

export CXX=clang++
export CXXFLAGS="-fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx908"
