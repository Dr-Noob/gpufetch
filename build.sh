#!/bin/bash

# gpufetch build script
set -e

rm -rf build/ gpufetch
mkdir build/
cd build/

if [ "$1" == "debug" ]
then
  BUILD_TYPE="Debug"
else
  BUILD_TYPE="Release"
fi

# In case you have CUDA installed but it is not detected,
# - set CMAKE_CUDA_COMPILER to your nvcc binary:
# - set CMAKE_CUDA_COMPILER_TOOLKIT_ROOT to the CUDA root dir
# for example:
# cmake -DCMAKE_BUILD_TYPE=$BUILD_TYPE -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DCMAKE_CUDA_COMPILER_TOOLKIT_ROOT=/usr/local/cuda/ ..

# In case you want to explicitely disable a backend, you can:
# Disable CUDA backend:
# cmake -DCMAKE_BUILD_TYPE=$BUILD_TYPE -DENABLE_CUDA_BACKEND=OFF ..
# Disable Intel backend:
# cmake -DCMAKE_BUILD_TYPE=$BUILD_TYPE -DENABLE_INTEL_BACKEND=OFF ..

cmake -DCMAKE_BUILD_TYPE=$BUILD_TYPE ..

os=$(uname)
if [ "$os" == 'Linux' ]; then
  make -j$(nproc)
elif [ "$os" == 'FreeBSD' ]; then
  gmake -j4
fi

cd -
ln -s build/gpufetch .
