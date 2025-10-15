cmake_minimum_required(VERSION 3.10)
include(CheckLanguage)
include(ExternalProject)

project(gpufetch CXX)

set(SRC_DIR "src")
set(COMMON_DIR "${SRC_DIR}/common")
set(CUDA_DIR "${SRC_DIR}/cuda")
set(HSA_DIR "${SRC_DIR}/hsa")
set(INTEL_DIR "${SRC_DIR}/intel")

# Enable Intel backend by default
if(NOT DEFINED ENABLE_INTEL_BACKEND)
    set(ENABLE_INTEL_BACKEND true)
endif()

if(ENABLE_CUDA_BACKEND)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    # Must link_directories early so add_executable(gpufetch ...) gets the right directories
    link_directories(cuda_backend ${CMAKE_CUDA_COMPILER_TOOLKIT_ROOT}/targets/x86_64-linux/lib)
  else()
    set(ENABLE_CUDA_BACKEND false)
  endif()
endif()

if(ENABLE_HSA_BACKEND)
  find_package(ROCmCMakeBuildTools)
  if (ROCmCMakeBuildTools_FOUND)
    find_package(hsa-runtime64 1.0 REQUIRED)    
    link_directories(hsa_backend hsa-runtime64::hsa-runtime64)

    # Find HSA headers
    # ROCm does not seem to provide this, which is quite frustrating.
    find_path(HSA_INCLUDE_DIR
      NAMES hsa/hsa.h
      HINTS
          $ENV{ROCM_PATH}/include     # allow users override via env variable
          /opt/rocm/include           # common default path
          /usr/include
          /usr/local/include
    )

    if(NOT HSA_INCLUDE_DIR)
      message(STATUS "${BoldYellow}HSA not found, disabling HSA backend${ColorReset}")
      set(ENABLE_HSA_BACKEND false)
    endif()
  else()
    # rocm-cmake is not installed, try to manually find neccesary files.
    if (NOT DEFINED ROCM_PATH)
      set(ROCM_PATH "/opt/rocm" CACHE PATH "Path to ROCm")
    endif()

    find_path(HSA_INCLUDE_DIR hsa/hsa.h HINTS ${ROCM_PATH}/include)
    find_library(HSA_LIBRARY hsa-runtime64 HINTS ${ROCM_PATH}/lib ${ROCM_PATH}/lib64)

    if (HSA_INCLUDE_DIR and HSA_LIBRARY)
      message(STATUS "${BoldYellow}Found ROCm manually${ColorReset}")
    else()
      set(ENABLE_HSA_BACKEND false)
      message(STATUS "${BoldYellow}ROCm not found${ColorReset}")
    endif()
  endif()
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
    BUILD_COMMAND make SHARED=no HWDB=no
    BUILD_IN_SOURCE true
    INSTALL_COMMAND make PREFIX=${PCIUTILS_INSTALL_LOCATION} install-lib
  )

  include_directories(${PCIUTILS_INSTALL_LOCATION}/include)
  link_directories(${PCIUTILS_INSTALL_LOCATION}/lib)
else()
  include_directories(${PCIUTILS_INCLUDE_DIR})
  link_libraries(${PCIUTILS_LIBRARIES})
  # Needed for linking libpci in FreeBSD
  link_directories(/usr/local/lib/)
endif()

add_executable(gpufetch ${COMMON_DIR}/main.cpp ${COMMON_DIR}/args.cpp ${COMMON_DIR}/gpu.cpp ${COMMON_DIR}/pci.cpp ${COMMON_DIR}/sort.cpp ${COMMON_DIR}/global.cpp ${COMMON_DIR}/printer.cpp ${COMMON_DIR}/master.cpp ${COMMON_DIR}/uarch.cpp)
set(SANITY_FLAGS -Wfloat-equal -Wshadow -Wpointer-arith -Wall -Wextra -pedantic -fstack-protector-all -pedantic)
target_compile_features(gpufetch PRIVATE cxx_std_11)
target_compile_options(gpufetch PRIVATE ${SANITY_FLAGS})

if(ENABLE_INTEL_BACKEND)
  target_compile_definitions(gpufetch PUBLIC BACKEND_INTEL)

  add_library(intel_backend STATIC ${INTEL_DIR}/intel.cpp ${INTEL_DIR}/pci.cpp ${INTEL_DIR}/uarch.cpp ${INTEL_DIR}/udev.cpp ${INTEL_DIR}/cpuid.cpp)

  if(NOT ${PCIUTILS_FOUND})
    add_dependencies(intel_backend pciutils)
  endif()

  target_link_libraries(gpufetch intel_backend)
endif()

if(ENABLE_CUDA_BACKEND)
  target_compile_definitions(gpufetch PUBLIC BACKEND_CUDA)

  # https://en.wikipedia.org/w/index.php?title=CUDA&section=5#GPUs_supported
  # https://raw.githubusercontent.com/PointCloudLibrary/pcl/master/cmake/pcl_find_cuda.cmake
  if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL "11.1")
    set(CMAKE_CUDA_ARCHITECTURES 35 37 50 52 53 60 61 62 70 72 75 80 86)
  elseif(${CMAKE_CUDA_COMPILER_VERSION} EQUAL "11.0")
    set(CMAKE_CUDA_ARCHITECTURES 30 32 35 37 50 52 53 60 61 62 70 72 75 80)
  elseif(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL "10.0")
    set(CMAKE_CUDA_ARCHITECTURES 30 32 35 37 50 52 53 60 61 62 70 72 75)
  elseif(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL "9.0")
    set(CMAKE_CUDA_ARCHITECTURES 30 32 35 37 50 52 53 60 61 62 70 72)
  elseif(${CMAKE_CUDA_COMPILER_VERSION} VERSION_EQUAL "8.0")
    set(CMAKE_CUDA_ARCHITECTURES 20 21 30 32 35 37 50 52 53 60 61 62)
  endif()

  add_library(cuda_backend STATIC ${CUDA_DIR}/cuda.cpp ${CUDA_DIR}/uarch.cpp ${CUDA_DIR}/pci.cpp)

  if(NOT ${PCIUTILS_FOUND})
    add_dependencies(cuda_backend pciutils)
  endif()

  target_include_directories(cuda_backend PUBLIC ${CMAKE_CUDA_COMPILER_TOOLKIT_ROOT}/targets/x86_64-linux/include)

  target_link_libraries(cuda_backend PRIVATE cudart)
  target_link_libraries(gpufetch cuda_backend)
endif()

if(ENABLE_HSA_BACKEND)
  target_compile_definitions(gpufetch PUBLIC BACKEND_HSA)

  add_library(hsa_backend STATIC ${HSA_DIR}/hsa.cpp ${HSA_DIR}/uarch.cpp)

  if(NOT ${PCIUTILS_FOUND})
    add_dependencies(hsa_backend pciutils)
  endif()

  target_include_directories(hsa_backend PRIVATE "${HSA_INCLUDE_DIR}")
  message(STATUS "Found HSA: ${HSA_INCLUDE_DIR}")

  target_link_libraries(hsa_backend PRIVATE ${HSA_LIBRARY})
  target_link_libraries(gpufetch hsa_backend)
endif()

target_link_libraries(gpufetch pci z)
install(TARGETS gpufetch DESTINATION bin)

if(NOT WIN32)
  string(ASCII 27 Esc)
  set(ColorReset "${Esc}[m")
  set(ColorBold  "${Esc}[1m")
  set(Red         "${Esc}[31m")
  set(Green       "${Esc}[32m")
  set(BoldRed     "${Esc}[1;31m")
  set(BoldGreen   "${Esc}[1;32m")
  set(BoldYellow  "${Esc}[1;33m")
endif()

message(STATUS "----------------------")
message(STATUS "gpufetch build report:")
if(ENABLE_CUDA_BACKEND)
  message(STATUS "CUDA backend: ${BoldGreen}ON${ColorReset}")
else()
  message(STATUS "CUDA backend: ${BoldRed}OFF${ColorReset}")
endif()
if(ENABLE_HSA_BACKEND)
  message(STATUS "HSA backend: ${BoldGreen}ON${ColorReset}")
else()
  message(STATUS "HSA backend: ${BoldRed}OFF${ColorReset}")
endif()
if(ENABLE_INTEL_BACKEND)
  message(STATUS "Intel backend: ${BoldGreen}ON${ColorReset}")
else()
  message(STATUS "Intel backend: ${BoldRed}OFF${ColorReset}")
endif()
message(STATUS "----------------------")
