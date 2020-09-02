module load cmake
module load kokkos/openmptarget_icpx_with_host_omp # kokkos/icpx
module load omp/2020.07.30.001 # for compiler needed by kokkoks/icpx

export CXXFLAGS="-std=c++17 -fiopenmp -fopenmp-targets=spir64 -D__STRICT_ANSI__"
export LDFLAGS="-fiopenmp -fopenmp-targets=spir64"

# only needed for kokkos/openmptarget_icpx_with_host_omp
export CMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH:$KOKKOS_HOME"
