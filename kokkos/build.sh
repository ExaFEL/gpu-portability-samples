#!/bin/bash

set -e

if [[ ! -d build ]]; then
    mkdir build
    pushd build
    cmake ..
    popd
fi

make -C build
