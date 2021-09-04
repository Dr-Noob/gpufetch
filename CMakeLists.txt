cmake_minimum_required(VERSION 3.10)
include(CheckLanguage)
include(ExternalProject)

project(gpufetch CXX)

set(SRC_DIR "src")
set(COMMON_DIR "${SRC_DIR}/common")
set(CUDA_DIR "${SRC_DIR}/cuda")

if(NOT WIN32)
  string(ASCII 27 Esc)
  set(ColorReset "${Esc}[m")
  set(ColorBold  "${Esc}[1m")
  set(Red         "${Esc}[31m")
  set(Green	  "${Esc}[32m")
  set(BoldRed     "${Esc}[1;31m")
  set(BoldGreen   "${Esc}[1;32m")
  set(BoldYellow  "${Esc}[1;33m")
endif()

check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
else()
  message(FATAL_ERROR "${BoldRed}[ERROR]${ColorReset} Unable to find CUDA compiler. You may use -DCMAKE_CUDA_COMPILER and -DCMAKE_CUDA_COMPILER_TOOLKIT_ROOT if CUDA is installed but not detected by CMake")
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
find_package(PCIUTILS)
if(NOT ${PCIUTILS_FOUND})
  message(STATUS "${BoldYellow}pciutils not found, downloading and building a local copy...${ColorReset}")

  # Download and build pciutils
  set(PCIUTILS_INSTALL_LOCATION ${CMAKE_BINARY_DIR}/pciutils-install)
  ExternalProject_Add(pciutils
    GIT_REPOSITORY https://github.com/pciutils/pciutils
    CONFIGURE_COMMAND ""
    BUILD_COMMAND make SHARED=no
    BUILD_IN_SOURCE true
    INSTALL_COMMAND make PREFIX=${PCIUTILS_INSTALL_LOCATION} install-lib
  )

  include_directories(${PCIUTILS_INSTALL_LOCATION}/include)
  link_directories(${PCIUTILS_INSTALL_LOCATION}/lib)
else()
  include_directories(${PCIUTILS_INCLUDE_DIR})
  link_libraries(${PCIUTILS_LIBRARIES})
endif()

set(SANITY_FLAGS "-Wfloat-equal -Wshadow -Wpointer-arith")
set(CMAKE_CXX_FLAGS "${SANITY_FLAGS} -Wall -Wextra -pedantic -fstack-protector-all -pedantic")

# https://en.wikipedia.org/w/index.php?title=CUDA&section=5#GPUs_supported
# https://raw.githubusercontent.com/PointCloudLibrary/pcl/master/cmake/pcl_find_cuda.cmake
if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL "11.0")
  set(CMAKE_CUDA_ARCHITECTURES 35 37 50 52 53 60 61 62 70 72 75 80 86)
elseif(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL "10.0")
  set(CMAKE_CUDA_ARCHITECTURES 30 32 35 37 50 52 53 60 61 62 70 72 75)
elseif(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL "9.0")
  set(CMAKE_CUDA_ARCHITECTURES 30 32 35 37 50 52 53 60 61 62 70 72)
elseif(${CMAKE_CUDA_COMPILER_VERSION} VERSION_EQUAL "8.0")
  set(CMAKE_CUDA_ARCHITECTURES 20 21 30 32 35 37 50 52 53 60 61 62)
endif()

link_directories(${CMAKE_CUDA_COMPILER_TOOLKIT_ROOT}/targets/x86_64-linux/lib)

add_library(cuda_backend STATIC ${CUDA_DIR}/cuda.cpp ${CUDA_DIR}/uarch.cpp ${CUDA_DIR}/pci.cpp)
add_executable(gpufetch ${COMMON_DIR}/main.cpp ${COMMON_DIR}/args.cpp ${COMMON_DIR}/gpu.cpp ${COMMON_DIR}/pci.cpp ${COMMON_DIR}/global.cpp ${COMMON_DIR}/printer.cpp)

if(NOT ${PCIUTILS_FOUND})
  add_dependencies(cuda_backend pciutils)
  add_dependencies(gpufetch pciutils)
endif()

target_include_directories(cuda_backend PUBLIC ${CMAKE_CUDA_COMPILER_TOOLKIT_ROOT}/samples/common/inc ${CMAKE_CUDA_COMPILER_TOOLKIT_ROOT}/targets/x86_64-linux/include)

target_link_libraries(cuda_backend cudart)
target_link_libraries(gpufetch cuda_backend pci z)

install(TARGETS gpufetch DESTINATION bin)
