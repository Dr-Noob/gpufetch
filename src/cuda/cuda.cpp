#include <helper_cuda.h>
#include <cuda_runtime.h>

#include "cuda.hpp"
#include "nvmlb.hpp"
#include "uarch.hpp"
#include "../common/global.hpp"

struct cache* get_cache_info(struct gpu_info* gpu, cudaDeviceProp prop) {
  struct cache* cach = (struct cache*) emalloc(sizeof(struct cache));

  cach->L2 = (struct cach*) emalloc(sizeof(struct cach));
  cach->L2->size = prop.l2CacheSize;
  cach->L2->num_caches = 1;
  cach->L2->exists = true;

  return cach;
}

struct topology* get_topology_info(struct gpu_info* gpu, cudaDeviceProp prop) {
  struct topology* topo = (struct topology*) emalloc(sizeof(struct topology));

  topo->streaming_mp = prop.multiProcessorCount;
  topo->cores_per_mp = _ConvertSMVer2Cores(prop.major, prop.minor);
  topo->cuda_cores = topo->streaming_mp * topo->cores_per_mp;

  return topo;
}

MEMTYPE guess_memory_type(struct memory* mem, struct gpu_info* gpu) {
  // 1. Guess data rate
  int32_t data_rate = -1;
  int32_t dr8 = abs((mem->freq/8) - gpu->freq);
  int32_t dr4 = abs((mem->freq/4) - gpu->freq);
  int32_t dr2 = abs((mem->freq/2) - gpu->freq);
  int32_t dr1 = abs((mem->freq/1) - gpu->freq);

  int32_t min = mem->freq;
  if(min > dr8) { data_rate = 8; min = dr8; }
  if(min > dr4) { data_rate = 4; min = dr4; }
  if(min > dr2) { data_rate = 2; min = dr2; }
  if(min > dr1) { data_rate = 1; min = dr1; }

  printf("data_rate=%d\n", data_rate);
  return MEMTYPE_GDDR6;
}

struct memory* get_memory_info(struct gpu_info* gpu, cudaDeviceProp prop) {
  struct memory* mem = (struct memory*) emalloc(sizeof(struct memory));

  mem->size_bytes = (unsigned long long) prop.totalGlobalMem;
  mem->freq = prop.memoryClockRate * 1e-3f;
  mem->bus_width = prop.memoryBusWidth;
  mem->type = guess_memory_type(mem, gpu);

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
  gpu->cach = get_cache_info(gpu, deviceProp);
  gpu->mem = get_memory_info(gpu, deviceProp);
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

