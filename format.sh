#!/bin/bash

set -e

clang-format -i */*.cc
clang-format -i */*.cl
clang-format -i */*.cu
