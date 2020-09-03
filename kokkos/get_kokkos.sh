#!/bin/bash

set -e

git clone -b develop https://github.com/kokkos/kokkos.git
cd kokkos
mkdir build
cd build
cmake -DKokkos_ARCH_VEGA906=on \
    -DKokkos_ENABLE_HIP=on \
    -DKokkos_ENABLE_SERIAL=on \
    -DKokkos_ENABLE_HIP_RELOCATABLE_DEVICE_CODE=off \
    -DCMAKE_INSTALL_PREFIX="$PWD/../install" \
    ..
make install -j8
