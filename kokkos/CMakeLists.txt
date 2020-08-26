cmake_minimum_required(VERSION 3.10)
project(saxpy)

find_package(Kokkos REQUIRED)

add_executable(saxpy saxpy.cc)
target_link_libraries(saxpy Kokkos::kokkos)
