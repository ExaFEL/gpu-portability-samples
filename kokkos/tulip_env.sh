module load cmake
module load rocm/3.7.0

# settings for Kokkos build
export CXX=hipcc
export CXXFLAGS=--amdgpu-target=gfx908
export CMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH:$PWD/kokkos/install"
