module load oneapi/2020.10.30.001 # compilers for kokkos
module load cmake
module load kokkos

# only needed for kokkos
export CMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH:$KOKKOS_HOME"
export CXXFLAGS='-std=c++17 -fiopenmp -fopenmp-targets=spir64="-mllvm -vpo-paropt-enable-64bit-opencl-atomics=true" -Wno-openmp-mapping'
