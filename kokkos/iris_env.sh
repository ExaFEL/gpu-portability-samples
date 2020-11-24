module load oneapi/2020.10.30.001 # compilers for kokkos
module load cmake
module load kokkos

# only needed for kokkos
export CMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH:$KOKKOS_HOME"
