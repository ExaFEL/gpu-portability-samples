module load cmake
module load rocm/5.1.0

# settings for Kokkos build
export CXX=hipcc
export CXXFLAGS=--amdgpu-target=gfx90a
export CMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH:$PWD/kokkos/install"
