#!/usr/bin/bash

mkdir build && cd build

if [ $1 = "all" ]; then
   cmake -DENABLE_TRITON=ON -DENABLE_ONNXRT=ON ..
fi

if [ $1 = "triton" ]; then
   cmake -DENABLE_TRITON=ON -DENABLE_ONNXRT=OFF ..
fi

if [ $1 = "onnxrt" ]; then
   cmake -DENABLE_TRITON=OFF -DENABLE_ONNXRT=ON ..
fi

make -j$(nproc)
