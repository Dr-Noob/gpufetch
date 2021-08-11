#include <helper_cuda.h>
#include <cuda_runtime.h>

#include "cuda.hpp"
#include "uarch.hpp"
#include "../common/global.hpp"

struct cache* get_cache_info(struct gpu_info* gpu) {
  struct cache* cach = (struct cache*) emalloc(sizeof(struct cache));

  return cach;
}

struct topology* get_topology_info(struct gpu_info* gpu) {
  struct topology* topo = (struct topology*) emalloc(sizeof(struct topology));

  return topo;
}

struct memory* get_memory_info(struct gpu_info* gpu) {
  struct memory* mem = (struct memory*) emalloc(sizeof(struct memory));

  return mem;
}

int64_t get_peak_performance(struct gpu_info* gpu) {
  return 1000;
}

struct gpu_info* get_gpu_info() {
  struct gpu_info* gpu = (struct gpu_info*) emalloc(sizeof(struct gpu_info));

  printf("Waiting for CUDA driver to start...\n");
  int dev = 0;
  cudaSetDevice(dev);
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, dev);

  gpu->vendor = GPU_VENDOR_NVIDIA;
  gpu->name = (char *) emalloc(sizeof(char) * (strlen(deviceProp.name) + 1));
  strcpy(gpu->name, deviceProp.name);
  gpu->freq = 10000;

  gpu->arch = get_uarch_from_cuda(gpu);
  gpu->cach = get_cache_info(gpu);
  gpu->topo = get_topology_info(gpu);
  gpu->peak_performance = get_peak_performance(gpu);

  return gpu;
}

char* get_str_sm(struct gpu_info* gpu) {
  return NULL;
}

char* get_str_cores_sm(struct gpu_info* gpu) {
  return NULL;
}

char* get_str_cuda_cores(struct gpu_info* gpu) {
  return NULL;
}

