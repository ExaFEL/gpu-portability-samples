cmake_minimum_required(VERSION 3.10)
project(GPUPortabilitySamples)

find_package(Kokkos REQUIRED)

add_executable(saxpy saxpy.cc)
target_link_libraries(saxpy Kokkos::kokkos)

add_executable(histogram histogram.cc)
target_link_libraries(histogram Kokkos::kokkos)
