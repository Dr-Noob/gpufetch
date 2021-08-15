#include <helper_cuda.h>
#include <cuda_runtime.h>

#include "cuda.hpp"
#include "nvmlb.hpp"
#include "uarch.hpp"
#include "../common/global.hpp"

struct cache* get_cache_info(cudaDeviceProp prop) {
  struct cache* cach = (struct cache*) emalloc(sizeof(struct cache));

  cach->L2 = (struct cach*) emalloc(sizeof(struct cach));
  cach->L2->size = prop.l2CacheSize;
  cach->L2->num_caches = 1;
  cach->L2->exists = true;

  return cach;
}

struct topology* get_topology_info(cudaDeviceProp prop) {
  struct topology* topo = (struct topology*) emalloc(sizeof(struct topology));

  topo->streaming_mp = prop.multiProcessorCount;
  topo->cores_per_mp = _ConvertSMVer2Cores(prop.major, prop.minor);
  topo->cuda_cores = topo->streaming_mp * topo->cores_per_mp;

  return topo;
}

int32_t guess_clock_multipilier(struct gpu_info* gpu, struct memory* mem) {
  // Guess clock multiplier
  int32_t clk_mul = -1;

  int32_t clk8 = abs((mem->freq/8) - gpu->freq);
  int32_t clk4 = abs((mem->freq/4) - gpu->freq);
  int32_t clk2 = abs((mem->freq/2) - gpu->freq);
  int32_t clk1 = abs((mem->freq/1) - gpu->freq);

  int32_t min = mem->freq;
  if(min > clk8) { clk_mul = 8; min = clk8; }
  if(min > clk4) { clk_mul = 4; min = clk4; }
  if(min > clk2) { clk_mul = 2; min = clk2; }
  if(min > clk1) { clk_mul = 1; min = clk1; }

  return clk_mul;
}

struct memory* get_memory_info(struct gpu_info* gpu, cudaDeviceProp prop) {
  struct memory* mem = (struct memory*) emalloc(sizeof(struct memory));

  mem->size_bytes = (unsigned long long) prop.totalGlobalMem;
  mem->freq = prop.memoryClockRate * 0.001f;
  mem->bus_width = prop.memoryBusWidth;
  mem->clk_mul = guess_clock_multipilier(gpu, mem);
  mem->type = guess_memtype_from_cmul_and_uarch(mem->clk_mul, gpu->arch);

  // Fix frequency returned from CUDA to show real frequency
  mem->freq = mem->freq  / mem->clk_mul;

  return mem;
}

int64_t get_peak_performance(struct gpu_info* gpu) {
  return gpu->freq * 1000000 * gpu->topo->cuda_cores * 2;
}

struct gpu_info* get_gpu_info() {
  struct gpu_info* gpu = (struct gpu_info*) emalloc(sizeof(struct gpu_info));
  gpu->pci = NULL;

  printf("Waiting for CUDA driver to start...");
  fflush(stdout);
  int dev = 0;
  cudaSetDevice(dev);
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, dev);
  printf("\r");

  gpu->freq = deviceProp.clockRate * 1e-3f;
  gpu->vendor = GPU_VENDOR_NVIDIA;
  gpu->name = (char *) emalloc(sizeof(char) * (strlen(deviceProp.name) + 1));
  strcpy(gpu->name, deviceProp.name);

  gpu->nvmld = nvml_init();
  if(nvml_get_pci_info(dev, gpu->nvmld)) {
    gpu->pci = get_pci_from_nvml(gpu->nvmld);
  }

  gpu->arch = get_uarch_from_cuda(gpu);
  gpu->cach = get_cache_info(deviceProp);
  gpu->mem = get_memory_info(gpu, deviceProp);
  gpu->topo = get_topology_info(deviceProp);
  gpu->peak_performance = get_peak_performance(gpu);

  return gpu;
}

char* get_str_sm(struct gpu_info* gpu) {
  uint32_t max_size = 10;
  char* dummy = (char *) ecalloc(max_size, sizeof(char));
  snprintf(dummy, max_size, "%d", gpu->topo->streaming_mp);
  return dummy;
}

char* get_str_cores_sm(struct gpu_info* gpu) {
  uint32_t max_size = 10;
  char* dummy = (char *) ecalloc(max_size, sizeof(char));
  snprintf(dummy, max_size, "%d", gpu->topo->cores_per_mp);
  return dummy;
}

char* get_str_cuda_cores(struct gpu_info* gpu) {
  uint32_t max_size = 10;
  char* dummy = (char *) ecalloc(max_size, sizeof(char));
  snprintf(dummy, max_size, "%d", gpu->topo->cuda_cores);
  return dummy;
}

