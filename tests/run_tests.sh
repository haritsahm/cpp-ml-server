#!/bin/bash

cmake -DRUN_TESTS=ON  -DENABLE_TRITON=OFF -DENABLE_ONNXRT=ON -DCMAKE_BUILD_TYPE=Debug .. && make
ctest --output-on-failure --progress --verbose
lcov --capture --directory . --output-file coverage.info
mkdir -p /mnt/coverage && cp coverage.info /mnt/coverage/coverage.info
