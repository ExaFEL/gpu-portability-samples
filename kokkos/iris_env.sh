module load oneapi # compilers for kokkos/openmptarget_icpx
module load cmake
module load kokkos/openmptarget_icpx

# only needed for kokkos/openmptarget_icpx
export CMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH:$KOKKOS_HOME"
