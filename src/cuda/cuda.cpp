#include <helper_cuda.h>
#include <cuda_runtime.h>

#include "cuda.hpp"
#include "nvmlb.hpp"
#include "uarch.hpp"
#include "../common/global.hpp"

struct cache* get_cache_info(struct gpu_info* gpu) {
  struct cache* cach = (struct cache*) emalloc(sizeof(struct cache));

  return cach;
}

struct topology* get_topology_info(struct gpu_info* gpu, cudaDeviceProp prop) {
  struct topology* topo = (struct topology*) emalloc(sizeof(struct topology));

  topo->streaming_mp = prop.multiProcessorCount;
  topo->cores_per_mp = _ConvertSMVer2Cores(prop.major, prop.minor);
  topo->cuda_cores = topo->streaming_mp * topo->cores_per_mp;

  return topo;
}

struct memory* get_memory_info(struct gpu_info* gpu) {
  struct memory* mem = (struct memory*) emalloc(sizeof(struct memory));

  return mem;
}

int64_t get_peak_performance(struct gpu_info* gpu) {
  return gpu->freq * 1000000 * gpu->topo->cuda_cores * 2;
}

struct gpu_info* get_gpu_info() {
  struct gpu_info* gpu = (struct gpu_info*) emalloc(sizeof(struct gpu_info));
  gpu->pci = NULL;

  printf("Waiting for CUDA driver to start...\n");
  int dev = 0;
  cudaSetDevice(dev);
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, dev);

  gpu->freq = deviceProp.clockRate * 1e-3f;
  gpu->vendor = GPU_VENDOR_NVIDIA;
  gpu->name = (char *) emalloc(sizeof(char) * (strlen(deviceProp.name) + 1));
  strcpy(gpu->name, deviceProp.name);

  gpu->nvmld = nvml_init();
  if(nvml_get_pci_info(dev, gpu->nvmld)) {
    gpu->pci = get_pci_from_nvml(gpu->nvmld);
  }

  gpu->arch = get_uarch_from_cuda(gpu);
  gpu->cach = get_cache_info(gpu);
  gpu->topo = get_topology_info(gpu, deviceProp);
  gpu->peak_performance = get_peak_performance(gpu);

  return gpu;
}

int32_t get_str_sm(struct gpu_info* gpu) {
  return gpu->topo->streaming_mp;
}

int32_t get_str_cores_sm(struct gpu_info* gpu) {
  return gpu->topo->cores_per_mp;
}

int32_t get_str_cuda_cores(struct gpu_info* gpu) {
  return gpu->topo->cuda_cores;
}

