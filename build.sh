#!/bin/bash

print_help() {
  cat << EOF
Usage: $0 <backends> [build_type]

  <backends>    MANDATORY. Comma-separated list of 
                backends to enable.
                Valid options: hsa, intel, cuda
                Example: hsa,cuda

  [build_type]  OPTIONAL. Build type. Valid options:
                debug, release (default: release)

Examples:
  $0 hsa,intel debug
  $0 cuda
  $0 hsa,intel,cuda release
EOF
}

# gpufetch build script
set -e

rm -rf build/ gpufetch
mkdir build/
cd build/

if [ "$1" == "--help" ]
then
  echo "gpufetch build script"
  echo
  print_help
  exit 0
fi

if [[ $# -lt 1 ]]; then
  echo "ERROR: At least one backend must be specified."
  echo
  print_help
  exit 1
fi

# Determine if last argument is build type
LAST_ARG="${!#}"
if [[ "$LAST_ARG" == "debug" || "$LAST_ARG" == "release" ]]; then
  BUILD_TYPE="$LAST_ARG"
  BACKEND_ARG="${1}"
else
  BUILD_TYPE="release"
  BACKEND_ARG="${1}"
fi

# Split comma-separated backends into an array
IFS=',' read -r -a BACKENDS <<< "$BACKEND_ARG"

# Validate build type 
if [[ "$BUILD_TYPE" != "debug" && "$BUILD_TYPE" != "release" ]]
then
  echo "Error: Invalid build type '$BUILD_TYPE'."
  echo "Valid options are: debug, release"
  exit 1
fi

# From lower to upper case
CMAKE_FLAGS="-DCMAKE_BUILD_TYPE=${BUILD_TYPE^}"

# Validate backends
VALID_BACKENDS=("hsa" "intel" "cuda")

for BACKEND in "${BACKENDS[@]}"; do
  case "$BACKEND" in
    hsa)
      CMAKE_FLAGS+=" -DENABLE_HSA_BACKEND=ON"
      ;;
    intel)
      CMAKE_FLAGS+=" -DENABLE_INTEL_BACKEND=ON"
      ;;
    cuda)
      CMAKE_FLAGS+=" -DENABLE_CUDA_BACKEND=ON"
      ;;
    *)
      echo "ERROR: Invalid backend '$BACKEND'."
      echo "Valid options: ${VALID_BACKENDS[*]}"
      exit 1
      ;;
  esac
done

# You can also manually specify the compilation flags.
# If you need to, just run the cmake command directly
# instead of using this script.
#
# Here you will find some help:
#
# In case you have CUDA installed but it is not detected,
# - set CMAKE_CUDA_COMPILER to your nvcc binary:
# - set CMAKE_CUDA_COMPILER_TOOLKIT_ROOT to the CUDA root dir
# for example:
# cmake -DCMAKE_BUILD_TYPE=$BUILD_TYPE -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DCMAKE_CUDA_COMPILER_TOOLKIT_ROOT=/usr/local/cuda/ ..
#
# In case you want to explicitely disable a backend, you can:
# Disable CUDA backend:
# cmake -DCMAKE_BUILD_TYPE=$BUILD_TYPE -DENABLE_CUDA_BACKEND=OFF ..
# Disable HSA backend:
# cmake -DCMAKE_BUILD_TYPE=$BUILD_TYPE -DENABLE_HSA_BACKEND=OFF ..
# Disable Intel backend:
# cmake -DCMAKE_BUILD_TYPE=$BUILD_TYPE -DENABLE_INTEL_BACKEND=OFF ..

echo "$0: Running cmake $CMAKE_FLAGS"
echo 
cmake $CMAKE_FLAGS ..

os=$(uname)
if [ "$os" == 'Linux' ]; then
  make -j$(nproc)
elif [ "$os" == 'FreeBSD' ]; then
  gmake -j4
fi

cd -
ln -s build/gpufetch .
