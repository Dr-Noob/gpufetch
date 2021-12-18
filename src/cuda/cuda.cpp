#include <helper_cuda.h>
#include <cuda_runtime.h>

#include "cuda.hpp"
#include "uarch.hpp"
#include "../common/pci.hpp"
#include "../common/global.hpp"

bool print_gpu_cuda(struct gpu_info* gpu) {
  char* cc = get_str_cc(gpu->arch);
  printf("%s (Compute Capability %s)\n", gpu->name, cc);
  free(cc);

  return true;
}

struct cache* get_cache_info(cudaDeviceProp prop) {
  struct cache* cach = (struct cache*) emalloc(sizeof(struct cache));

  cach->L2 = (struct cach*) emalloc(sizeof(struct cach));
  cach->L2->size = prop.l2CacheSize;
  cach->L2->num_caches = 1;
  cach->L2->exists = true;

  return cach;
}

int get_tensor_cores(int sm, int major) {
  if(major == 7) return sm * 8;
  else if(major == 8) return sm * 4;
  else return 0;
}

struct topology* get_topology_info(cudaDeviceProp prop) {
  struct topology* topo = (struct topology*) emalloc(sizeof(struct topology));

  topo->streaming_mp = prop.multiProcessorCount;
  topo->cores_per_mp = _ConvertSMVer2Cores(prop.major, prop.minor);
  topo->cuda_cores = topo->streaming_mp * topo->cores_per_mp;
  topo->tensor_cores = get_tensor_cores(topo->streaming_mp, prop.major);

  return topo;
}

int32_t guess_clock_multipilier(struct gpu_info* gpu, struct memory* mem) {
  // Guess clock multiplier
  int32_t clk_mul = 1;

  int32_t clk8 = abs((mem->freq/8) - gpu->freq);
  int32_t clk4 = abs((mem->freq/4) - gpu->freq);
  int32_t clk2 = abs((mem->freq/2) - gpu->freq);
  int32_t clk1 = abs((mem->freq/1) - gpu->freq);

  int32_t min = mem->freq;
  if(clkm_possible_for_uarch(8, gpu->arch) && min > clk8) { clk_mul = 8; min = clk8; }
  if(clkm_possible_for_uarch(4, gpu->arch) && min > clk4) { clk_mul = 4; min = clk4; }
  if(clkm_possible_for_uarch(2, gpu->arch) && min > clk2) { clk_mul = 2; min = clk2; }
  if(clkm_possible_for_uarch(1, gpu->arch) && min > clk1) { clk_mul = 1; min = clk1; }

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

// Compute peak performance when using CUDA cores
int64_t get_peak_performance_cuda(struct gpu_info* gpu) {
  return gpu->freq * 1000000 * gpu->topo->cuda_cores * 2;
}

// Compute peak performance when using tensor cores
int64_t get_peak_performance_tcu(struct gpu_info* gpu) {
  return gpu->freq * 1000000 * 4 * 4 * 8 * gpu->topo->tensor_cores;
}

struct gpu_info* get_gpu_info_cuda(int gpu_idx) {
  struct gpu_info* gpu = (struct gpu_info*) emalloc(sizeof(struct gpu_info));
  gpu->pci = NULL;
  gpu->idx = gpu_idx;

  if(gpu->idx < 0) {
    printErr("GPU index must be equal or greater than zero");
    return NULL;
  }

  if(gpu_idx == 0) {
    printf("Waiting for CUDA driver to start...");
    fflush(stdout);
  }

  int num_gpus = -1;
  cudaError_t err = cudaSuccess;
  if ((err = cudaGetDeviceCount(&num_gpus)) != cudaSuccess) {
    printErr("%s: %s", cudaGetErrorName(err), cudaGetErrorString(err));
    return NULL;
  }

  if(gpu_idx == 0) {
    printf("\r");
  }

  if(num_gpus <= 0) {
    printErr("No CUDA capable devices found!");
    return NULL;
  }

  if(gpu->idx+1 > num_gpus) {
    // Master is trying to query an invalid GPU
    return NULL;
  }

  cudaDeviceProp deviceProp;
  if ((err = cudaGetDeviceProperties(&deviceProp, gpu->idx)) != cudaSuccess) {
    printErr("%s: %s", cudaGetErrorName(err), cudaGetErrorString(err));
    return NULL;
  }

  gpu->freq = deviceProp.clockRate * 1e-3f;
  gpu->vendor = GPU_VENDOR_NVIDIA;
  gpu->name = (char *) emalloc(sizeof(char) * (strlen(deviceProp.name) + 1));
  strcpy(gpu->name, deviceProp.name);

  struct pci_dev *devices = get_pci_devices_from_pciutils();
  gpu->pci = get_pci_from_pciutils(devices, PCI_VENDOR_ID_NVIDIA);
  gpu->arch = get_uarch_from_cuda(gpu);
  gpu->cach = get_cache_info(deviceProp);
  gpu->mem = get_memory_info(gpu, deviceProp);
  gpu->topo = get_topology_info(deviceProp);
  gpu->peak_performance = get_peak_performance_cuda(gpu);
  gpu->peak_performance_tcu = get_peak_performance_tcu(gpu);

  return gpu;
}

char* get_str_sm(struct gpu_info* gpu) {
  return get_str_generic(gpu->topo->streaming_mp);
}

char* get_str_cores_sm(struct gpu_info* gpu) {
  return get_str_generic(gpu->topo->cores_per_mp);
}

char* get_str_cuda_cores(struct gpu_info* gpu) {
  return get_str_generic(gpu->topo->cuda_cores);
}

char* get_str_tensor_cores(struct gpu_info* gpu) {
  return get_str_generic(gpu->topo->tensor_cores);
}

