#include <helper_cuda.h>
#include <cuda_runtime.h>

#include "cuda.hpp"
#include "../common/global.hpp"

struct gpu_info* get_gpu_info() {
  struct gpu_info* gpu = (struct gpu_info*) emalloc(sizeof(struct gpu_info));

  int dev = 0;
  cudaSetDevice(dev);
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, dev);

  gpu->name = (char *) emalloc(sizeof(char) * (strlen(deviceProp.name) + 1));
  strcpy(gpu->name, deviceProp.name);

  return gpu;
}
