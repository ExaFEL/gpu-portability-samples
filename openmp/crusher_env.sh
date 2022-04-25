module load rocm/5.1.0

export CXX=amdclang++
export CXXFLAGS="-fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx90a"
